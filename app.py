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
import numpy as np

from image_solver import solve_image
from digit_reader import _visual_occupied
from sudoku_recovery import (
    solution_is_valid,
    solution_preserves_clues,
)


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "output"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
)

app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/output/<path:filename>")
def output_file(filename):
    return send_from_directory(
        OUTPUT_FOLDER,
        filename,
    )


def _validate_final_result(board, solution):
    if solution is None:
        return False, "ไม่พบคำตอบ Sudoku"

    if not solution_is_valid(solution):
        return False, "คำตอบ Sudoku ไม่ถูกต้อง"

    if not solution_preserves_clues(board, solution):
        return False, "คำตอบเปลี่ยนเลขโจทย์เดิม"

    return True, None


def _cell_has_visible_mark(image, row, col):
    """ตรวจหมึกที่อยู่ในพื้นที่กลาง Cell โดยไม่พึ่ง OCR

    ใช้เป็น safety guard ตอนวาดคำตอบเท่านั้น:
    ถ้ามีตัวเลขจริงอยู่ในภาพ จะไม่วาดทับ แม้ OCR จะพลาดช่องนั้น
    """
    if image is None:
        return False

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    if gray.shape[:2] != (450, 450):
        gray = cv2.resize(
            gray,
            (450, 450),
            interpolation=cv2.INTER_CUBIC,
        )

    # ตัดบริเวณใกล้เส้น Grid ออก
    y1 = row * 50 + 9
    y2 = (row + 1) * 50 - 9
    x1 = col * 50 + 9
    x2 = (col + 1) * 50 - 9

    cell = gray[y1:y2, x1:x2]

    if cell.size == 0:
        return False

    # พื้นหลังประมาณจากมุมของ Cell
    h, w = cell.shape
    corner_size = max(3, min(h, w) // 5)
    corners = np.concatenate([
        cell[:corner_size, :corner_size].ravel(),
        cell[:corner_size, -corner_size:].ravel(),
        cell[-corner_size:, :corner_size].ravel(),
        cell[-corner_size:, -corner_size:].ravel(),
    ])

    background = float(np.median(corners))

    # ตัวเลข/หมึกต้องแตกต่างจากพื้นหลังพอสมควร
    difference = np.abs(
        cell.astype(np.int16) - int(round(background))
    )

    ink = difference >= 35

    # ตัด pixel ที่รวมกันเป็นเส้นเล็ก ๆ ออก
    mask = (ink.astype(np.uint8) * 255)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8),
    )

    count = cv2.countNonZero(mask)

    # ตัวเลข Sudoku ปกติจะมีพื้นที่หมึกมากกว่านี้
    return count >= 35


def _draw_solution(sudoku_image, occupied, solution):
    """วาดเฉพาะช่องว่างจริง และห้ามเขียนทับเลขที่มองเห็นในภาพ"""
    output = sudoku_image.copy()

    # Detector เดิมยังใช้ได้เป็นหนึ่งใน safety checks
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
            # ถ้า OCR หรือ visual detector เห็นเลขแล้ว ห้ามวาด
            if occupied[row][col] or visual_occupied[row][col]:
                continue

            # Safety guard ตัวที่สองจาก pixel จริง
            if _cell_has_visible_mark(
                sudoku_image,
                row,
                col,
            ):
                print(
                    f"Draw guard: ข้าม R{row + 1}C{col + 1} "
                    "เพราะพบหมึกในภาพ"
                )
                continue

            number = solution[row][col]
            if number not in range(1, 10):
                continue

            text = str(number)
            font_scale = min(cell_w, cell_h) / 45.0
            text_size = cv2.getTextSize(
                text,
                font,
                font_scale,
                thickness,
            )[0]

            center_x = int(
                col * cell_w + cell_w / 2
            )
            center_y = int(
                row * cell_h
                + cell_h / 2
                + text_size[1] / 2
            )
            x = int(
                center_x - text_size[0] / 2
            )

            cv2.putText(
                output,
                text,
                (x, center_y),
                font,
                font_scale,
                green,
                thickness,
                cv2.LINE_AA,
            )
            drawn += 1

    return output, drawn


@app.route(
    "/solve",
    methods=["POST"],
)
def solve_route():

    files = request.files.getlist("images")

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
            file.save(str(input_path))

            if not input_path.exists():
                raise RuntimeError(
                    "บันทึกไฟล์ Input ไม่สำเร็จ"
                )

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

            output, drawn = _draw_solution(
                sudoku_image,
                occupied,
                solution,
            )

            if output is None:
                raise RuntimeError(
                    "สร้างภาพผลลัพธ์ไม่สำเร็จ"
                )

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
            try:
                if input_path.exists():
                    input_path.unlink()
            except OSError:
                pass

    return render_template(
        "index.html",
        results=results,
    )


@app.errorhandler(413)
def request_too_large(error):
    return render_template(
        "index.html",
        error="ไฟล์ใหญ่เกิน 32 MB",
    ), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        debug=False,
        host="0.0.0.0",
        port=port,
    )
