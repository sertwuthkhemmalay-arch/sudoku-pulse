import cv2
import os


INPUT_FILE = "sudoku_450.png"

OUTPUT_DIR = "cells"

SIZE = 450
CELL_SIZE = 50


def extract_cells(image_path=INPUT_FILE):

    print()
    print("==============================")
    print("Sudoku Cell Extractor")
    print("==============================")

    # ----------------------------------
    # อ่านภาพ
    # ----------------------------------

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"ไม่พบไฟล์: {image_path}"
        )

    height, width = image.shape[:2]

    print(f"ภาพ: {width} x {height}")

    # ----------------------------------
    # ตรวจขนาด
    # ----------------------------------

    if width != SIZE or height != SIZE:

        raise ValueError(
            "ภาพต้องมีขนาด 450 x 450"
        )

    # ----------------------------------
    # สร้างโฟลเดอร์
    # ----------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ----------------------------------
    # ลบ Cell เก่า
    # ----------------------------------

    for filename in os.listdir(OUTPUT_DIR):

        if filename.endswith(".png"):

            os.remove(
                os.path.join(
                    OUTPUT_DIR,
                    filename
                )
            )

    cells = []

    # ----------------------------------
    # แบ่ง 9 × 9
    # ----------------------------------

    for row in range(9):

        for col in range(9):

            x1 = col * CELL_SIZE
            y1 = row * CELL_SIZE

            x2 = (col + 1) * CELL_SIZE
            y2 = (row + 1) * CELL_SIZE

            cell = image[
                y1:y2,
                x1:x2
            ]

            # ----------------------------------
            # ตัดเส้นตารางออก
            # ----------------------------------

            margin = 5

            cell = cell[
                margin:CELL_SIZE - margin,
                margin:CELL_SIZE - margin
            ]

            # ----------------------------------
            # ขยาย Cell
            # ----------------------------------

            cell = cv2.resize(
                cell,
                (200, 200),
                interpolation=cv2.INTER_CUBIC
            )

            cells.append(cell)

            # ----------------------------------
            # บันทึก
            # ----------------------------------

            index = row * 9 + col

            filename = f"cell_{index:02d}.png"

            path = os.path.join(
                OUTPUT_DIR,
                filename
            )

            cv2.imwrite(
                path,
                cell
            )

    print()
    print(f"สร้าง Cells: {len(cells)}")

    if len(cells) != 81:

        raise RuntimeError(
            "สร้าง Cell ไม่ครบ 81 ช่อง"
        )

    print()
    print("แบ่ง Sudoku สำเร็จ! ✅")

    return cells


if __name__ == "__main__":

    extract_cells()