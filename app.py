from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
)
import os
from pathlib import Path
import traceback
import uuid
import cv2

from image_solver import solve_image
from digit_reader import _visual_occupied
from sudoku_recovery import (
    solution_is_valid,
    solution_preserves_clues,
)


# ==========================================
# Paths - อ้างอิงจากตำแหน่ง app.py เสมอ
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "output"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ==========================================
# Flask
# ==========================================

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
)

app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


# ==========================================
# หน้าแรก
# ==========================================

@app.route("/")
def index():
    return render_template("index.html")


# ==========================================
# เปิด Output
# ==========================================

@app.route("/output/<path:filename>")
def output_file(filename):
    return send_from_directory(
        OUTPUT_FOLDER,
        filename,
    )


# ==========================================
# ตรวจ Solution เพิ่มก่อนบันทึก
# ==========================================

def _validate_final_result(
    board,
    solution,
):
    if solution is None:
        return False, "ไม่พบคำตอบ Sudoku"

    if not solution_is_valid(solution):
        return False, "คำตอบ Sudoku ไม่ถูกต้อง"

    if not solution_preserves_clues(
        board,
        solution,
    ):
        return False, "คำตอบเปลี่ยนเลขโจทย์เดิม"

    return True, None


# ==========================================
# วาด Solution
# ==========================================

def _draw_solution(
    sudoku_image,
    occupied,
    solution,
):
    """วาดเฉพาะช่องที่ภาพต้นฉบับไม่มีเลขอยู่แล้ว"""
    output = sudoku_image.copy()

    # ตรวจจากภาพจริง ไม่ใช้ board ที่ recovery อาจแก้ค่า OCR แล้ว
    visual_occupied = _visual_occupied(sudoku_image)

    height, width = output.shape[:2]
    cell_w = width / 9.0
    cell_h = height / 9.0

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    green = (0, 180, 0)
    drawn = 0

    for row in range(9):
        for col in range(9):
            # ถ้ามีเลขอยู่ในภาพจริง ห้ามวาดทับ
            if occupied[row][col] or visual_occupied[row][col]:
                continue

            number = solution[row][col]
            if number not in range(1, 10):
                continue

            text = str(number)
            font_scale = min(cell_w, cell_h) / 45.0
            text_size = cv2.getTextSize(
                text, font, font_scale, thickness
            )[0]

            center_x = int(col * cell_w + cell_w / 2)
            center_y = int(
                row * cell_h + cell_h / 2 + text_size[1] / 2
            )
            x = int(center_x - text_size[0] / 2)

            cv2.putText(
                output, text, (x, center_y), font,
                font_scale, green, thickness, cv2.LINE_AA
            )
            drawn += 1

    return output, drawn


# ==========================================
# /solve
# ==========================================

@app.route(
    "/solve",
    methods=["POST"],
)
def solve_route():

    files = request.files.getlist("images")

    # บาง HTML ใช้ชื่อ image[]
    if not files:
        files = request.files.getlist("image")

    if not files:
        return render_template(
            "index.html",
            error="ไม่พบไฟล์ภาพ",
        )

    results = []

    for file in files:

        original_name = (
            file.filename or "unnamed.png"
        )

        if not original_name.strip():
            continue

        extension = Path(
            original_name
        ).suffix.lower()

        allowed = {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
        }

        if extension not in allowed:
            results.append({
                "filename": original_name,
                "success": False,
                "error": (
                    "รองรับ PNG, JPG, JPEG และ WEBP"
                ),
            })
            continue

        request_id = uuid.uuid4().hex

        input_path = (
            UPLOAD_FOLDER
            / f"{request_id}{extension}"
        )

        output_filename = (
            f"{request_id}_solved.png"
        )

        output_path = (
            OUTPUT_FOLDER
            / output_filename
        )

        try:
            # ==================================
            # บันทึก Input
            # ==================================
            file.save(str(input_path))

            if not input_path.exists():
                raise RuntimeError(
                    "บันทึกไฟล์ Input ไม่สำเร็จ"
                )

            # ==================================
            # Solve
            # ==================================
            print(
                f"\n========== {original_name} =========="
            )

            result = solve_image(
                str(input_path)
            )

            if result is None:
                results.append({
                    "filename": original_name,
                    "success": False,
                    "error": (
                        "Solver ไม่สามารถหา "
                        "คำตอบที่ยืนยันได้"
                    ),
                })
                continue

            (
                sudoku_image,
                board,
                occupied,
                candidates,
                confidence,
                solution,
            ) = result

            # ==================================
            # Final validation
            # ==================================
            valid, error = _validate_final_result(
                board,
                solution,
            )

            if not valid:
                results.append({
                    "filename": original_name,
                    "success": False,
                    "error": error,
                })
                continue

            # ==================================
            # วาดคำตอบ
            # ==================================
            output, drawn = _draw_solution(
                sudoku_image,
                occupied,
                solution,
            )

            if output is None:
                raise RuntimeError(
                    "สร้างภาพผลลัพธ์ไม่สำเร็จ"
                )

            # ==================================
            # บันทึก Output
            # ==================================
            if not cv2.imwrite(
                str(output_path),
                output,
            ):
                raise RuntimeError(
                    "cv2.imwrite() บันทึก Output ไม่สำเร็จ"
                )

            if not output_path.exists():
                raise RuntimeError(
                    "ไม่พบไฟล์ Output หลังบันทึก"
                )

            print(
                f"SUCCESS: {original_name} "
                f"-> วาดเพิ่ม {drawn} ช่อง"
            )

            results.append({
                "filename": original_name,
                "success": True,
                "output": output_filename,
            })

        except Exception as error:

            print(
                f"ERROR: {original_name}: {error}"
            )

            traceback.print_exc()

            results.append({
                "filename": original_name,
                "success": False,
                "error": str(error),
            })

        finally:
            # ลบ Input ชั่วคราว
            try:
                if input_path.exists():
                    input_path.unlink()
            except OSError:
                pass

    return render_template(
        "index.html",
        results=results,
    )


# ==========================================
# Error 413
# ==========================================

@app.errorhandler(413)
def request_too_large(error):
    return render_template(
        "index.html",
        error="ไฟล์ใหญ่เกิน 32 MB",
    ), 413


# ==========================================
# Run
# ==========================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        debug=False,
        host="0.0.0.0",
        port=port,
    )