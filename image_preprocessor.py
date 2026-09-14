import cv2
import os
import numpy as np
import uuid


# ==========================================
# ตั้งค่าพื้นฐาน
# ==========================================

INPUT_FILE = "sudoku.png"
OUTPUT_FILE = "sudoku_450.png"

SIZE = 450


# ==========================================
# ฟังก์ชันหาเส้น Sudoku
# ==========================================

def find_grid_bounds(image):

    # --------------------------------------
    # เปลี่ยนภาพสีเป็นภาพขาวดำ
    # --------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------
    # หาเส้นขอบของภาพ
    # --------------------------------------

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    # --------------------------------------
    # หาเส้นตรง
    # --------------------------------------

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=int(min(gray.shape) * 0.25),
        minLineLength=int(min(gray.shape) * 0.50),
        maxLineGap=12
    )

    # สร้างรายการเก็บเส้นแนวนอน
    horizontal = []

    # สร้างรายการเก็บเส้นแนวตั้ง
    vertical = []

    # ถ้าหาเส้นไม่ได้
    if lines is None:

        return None

    # --------------------------------------
    # ตรวจสอบเส้นทีละเส้น
    # --------------------------------------

    for line in lines:

        # ดึง x1, y1, x2, y2
        coords = np.asarray(line).reshape(-1)
        if coords.size != 4:
            continue
        x1, y1, x2, y2 = coords.astype(int)

        # ระยะในแนวนอน
        dx = abs(x2 - x1)

        # ระยะในแนวตั้ง
        dy = abs(y2 - y1)

        # ----------------------------------
        # ตรวจว่าเป็นเส้นแนวนอน
        # ----------------------------------

        if dx > dy * 5:

            y = (y1 + y2) / 2

            horizontal.append(y)

        # ----------------------------------
        # ตรวจว่าเป็นเส้นแนวตั้ง
        # ----------------------------------

        elif dy > dx * 5:

            x = (x1 + x2) / 2

            vertical.append(x)

    # --------------------------------------
    # ตรวจว่ามีเส้นเพียงพอหรือไม่
    # --------------------------------------

    if len(horizontal) < 2:

        return None

    if len(vertical) < 2:

        return None

    # --------------------------------------
    # หาเส้นขอบซ้ายและขวา
    # --------------------------------------

    x1 = int(min(vertical))

    x2 = int(max(vertical))

    # --------------------------------------
    # หาเส้นขอบบนและล่าง
    # --------------------------------------

    y1 = int(min(horizontal))

    y2 = int(max(horizontal))

    # --------------------------------------
    # ตรวจขนาดของ Sudoku
    # --------------------------------------

    width = x2 - x1

    height = y2 - y1

    # ถ้ากรอบเล็กเกินไป
    if width < image.shape[1] * 0.50:

        return None

    if height < image.shape[0] * 0.50:

        return None

    # ส่งค่ากรอบกลับ
    return x1, y1, x2, y2


def _projection_grid_bounds(image):
    """หาเส้น Grid จาก Edge Projection เมื่อ Hough จับเส้นด้านในผิด"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    col_score = (edges > 0).sum(axis=0)
    row_score = (edges > 0).sum(axis=1)

    def clusters(score, limit):
        idx = np.where(score > limit * 0.50)[0]
        groups = []
        for value in idx:
            if not groups or value - groups[-1][-1] > 8:
                groups.append([value])
            else:
                groups[-1].append(value)
        return [float(sum(g) / len(g)) for g in groups]

    vx = clusters(col_score, image.shape[0])
    hy = clusters(row_score, image.shape[1])

    def make_bounds(lines, size):
        if len(lines) < 8:
            return None

        # เลือกเส้น 8 เส้นภายในที่มีระยะใกล้เคียงกันที่สุด
        lines = sorted(lines)
        if len(lines) > 10:
            best = None
            best_err = float("inf")
            for start in range(len(lines) - 9):
                part = lines[start:start + 10]
                diffs = np.diff(part)
                err = float(np.std(diffs))
                if err < best_err:
                    best_err = err
                    best = part
            lines = best

        # ถ้ามี 8 เส้นภายใน ให้เติมขอบนอกด้วยระยะ median
        if len(lines) == 8:
            step = float(np.median(np.diff(lines)))
            lines = [lines[0] - step] + lines + [lines[-1] + step]
        elif len(lines) == 9:
            # กรณีมีขอบด้านหนึ่งหาย
            step = float(np.median(np.diff(lines)))
            left_gap = lines[0] - 0
            right_gap = size - 1 - lines[-1]
            if left_gap < right_gap:
                lines = lines + [lines[-1] + step]
            else:
                lines = [lines[0] - step] + lines
        elif len(lines) != 10:
            return None

        lines = np.asarray(lines, dtype=float)
        lines[0] = max(0.0, lines[0])
        lines[-1] = min(float(size - 1), lines[-1])

        if len(lines) != 10:
            return None

        diffs = np.diff(lines)
        if np.min(diffs) < size * 0.04:
            return None

        # ต้องมีความสม่ำเสมอพอสมควร
        if np.std(diffs) > np.mean(diffs) * 0.18:
            return None

        return int(round(lines[0])), int(round(lines[-1]))

    xb = make_bounds(vx, image.shape[1])
    yb = make_bounds(hy, image.shape[0])

    if xb is None or yb is None:
        return None

    x1, x2 = xb
    y1, y2 = yb

    if (x2 - x1) < image.shape[1] * 0.75:
        return None
    if (y2 - y1) < image.shape[0] * 0.75:
        return None

    return x1, y1, x2, y2


# ==========================================
# ฟังก์ชันเตรียมภาพ
# ==========================================

def preprocess(input_path=INPUT_FILE):

    print()

    print("==============================")

    print("Sudoku Image Preprocessor")

    print("==============================")


    # ======================================
    # อ่านภาพ
    # ======================================

    image = cv2.imread(
        input_path
    )

    # ตรวจสอบว่าอ่านภาพได้หรือไม่
    if image is None:

        raise FileNotFoundError(
            f"ไม่พบไฟล์: {input_path}"
        )


    # --------------------------------------
    # อ่านขนาดภาพ
    # --------------------------------------

    height, width = image.shape[:2]

    print(
        f"ภาพต้นฉบับ: {width} x {height}"
    )


    # ======================================
    # หา Sudoku Grid
    # ======================================

    print()

    print(
        "กำลังหา Sudoku Grid..."
    )


    bounds = find_grid_bounds(
        image
    )


    # ======================================
    # เลือกกรอบ Grid ที่เชื่อถือได้
    # ======================================

    use_bounds = bounds

    if bounds is not None:

        x1, y1, x2, y2 = bounds

        detected_width = x2 - x1 + 1
        detected_height = y2 - y1 + 1

        width_ratio = detected_width / width
        height_ratio = detected_height / height

        # ถ้า Hough จับเส้นด้านใน เช่นเส้น 3x3
        # ให้ใช้ Projection Grid แทน
        if width_ratio < 0.85 or height_ratio < 0.85:
            print(
                "Hough จับกรอบด้านใน -> ใช้ Projection Grid"
            )
            use_bounds = _projection_grid_bounds(image)

    else:

        print(
            "ไม่พบกรอบจาก Hough -> ใช้ Projection Grid"
        )
        use_bounds = _projection_grid_bounds(image)


    if use_bounds is not None:

        x1, y1, x2, y2 = use_bounds

        print(
            "พบกรอบ Sudoku ✅"
        )

        image = image[
            y1:y2 + 1,
            x1:x2 + 1
        ]

    else:

        print(
            "ไม่สามารถหากรอบ Grid ที่เชื่อถือได้"
        )

        print(
            "ใช้ภาพต้นฉบับแทน"
        )


    # ======================================
    # ตรวจสอบขนาดใหม่
    # ======================================

    h, w = image.shape[:2]


    # ======================================
    # ทำให้ภาพเป็นสี่เหลี่ยม
    # ======================================

    if w != h:

        print(
            "กำลัง crop ให้เป็นสี่เหลี่ยม..."
        )

        # เลือกด้านที่สั้นกว่า
        side = min(
            w,
            h
        )

        # หาตำแหน่งเริ่มต้นในแนวนอน
        x = (w - side) // 2

        # หาตำแหน่งเริ่มต้นในแนวตั้ง
        y = (h - side) // 2

        # crop ภาพ
        image = image[
            y:y + side,
            x:x + side
        ]


    # ======================================
    # Resize เป็น 450 x 450
    # ======================================

    image = cv2.resize(
        image,
        (SIZE, SIZE),
        interpolation=cv2.INTER_CUBIC
    )


    # ======================================
    # บันทึกภาพ processed แบบไม่ชนกัน
    # ======================================

    output_dir = os.path.dirname(os.path.abspath(input_path))
    output_file = os.path.join(
        output_dir,
        f".sudoku_processed_{uuid.uuid4().hex}.png"
    )

    if not cv2.imwrite(output_file, image):
        raise IOError(
            f"ไม่สามารถบันทึกภาพ processed: {output_file}"
        )


    # ======================================
    # แสดงผล
    # ======================================

    print()

    print(
        f"สร้างไฟล์: {output_file}"
    )

    print(
        f"ขนาดใหม่: {SIZE} x {SIZE}"
    )

    print()

    print(
        "Preprocess สำเร็จ! ✅"
    )


    # ส่งชื่อไฟล์ processed ที่ไม่ซ้ำกันกลับไป
    return output_file


# ==========================================
# เริ่มโปรแกรม
# ==========================================

if __name__ == "__main__":

    preprocess()