import cv2
import numpy as np
import pytesseract
import os
import shutil


DIGITS = "123456789"
DEBUG_DIR = "debug_digits"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def _clues_are_valid(board):
    """ตรวจว่าเลขที่ OCR อ่านได้ยังไม่ขัดแย้งกันใน Sudoku

    0 หมายถึงช่องว่างและไม่ถือว่าเป็นความขัดแย้ง
    ฟังก์ชันนี้ใช้ตรวจเฉพาะ clues ที่ OCR อ่านมา ไม่ได้ตรวจว่า
    กระดานต้องมีคำตอบหรือมีคำตอบเดียว
    """
    if board is None or len(board) != 9:
        return False

    for row in board:
        if len(row) != 9:
            return False
        for value in row:
            if not isinstance(value, (int, np.integer)) or value < 0 or value > 9:
                return False

    # ตรวจซ้ำในแต่ละแถว/คอลัมน์/บล็อก 3x3
    for i in range(9):
        row = [v for v in board[i] if v != 0]
        if len(row) != len(set(row)):
            return False

        col = [board[r][i] for r in range(9) if board[r][i] != 0]
        if len(col) != len(set(col)):
            return False

    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            block = [
                board[r][c]
                for r in range(br, br + 3)
                for c in range(bc, bc + 3)
                if board[r][c] != 0
            ]
            if len(block) != len(set(block)):
                return False

    return True


# ==========================================
# ตั้งค่า Tesseract
# ==========================================

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
elif shutil.which("tesseract"):
    pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")


# ==========================================
# เตรียมภาพ Sudoku
# ==========================================

def prepare_board(image):

    if isinstance(image, str):
        image = cv2.imread(image)

    if image is None:
        raise FileNotFoundError(
            "ไม่สามารถเปิดภาพ Sudoku ได้"
        )

    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image.copy()

    if gray.shape != (450, 450):
        gray = cv2.resize(
            gray,
            (450, 450),
            interpolation=cv2.INTER_CUBIC
        )

    return gray


# ==========================================
# ตรวจว่าภาพมีเส้นทแยงหรือไม่
# ==========================================

def has_diagonal_lines(gray):

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=20,
        minLineLength=20,
        maxLineGap=8
    )

    if lines is None:
        return False

    diagonal_count = 0

    for line in lines:

        coords = np.asarray(line).reshape(-1)
        if coords.size != 4:
            continue
        x1, y1, x2, y2 = coords.astype(int)

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dx == 0:
            continue

        angle = abs(
            np.degrees(
                np.arctan2(dy, dx)
            )
        )

        if 30 <= angle <= 60:
            diagonal_count += 1

    return diagonal_count >= 2


# ==========================================
# ลบเส้น Sudoku ออกจากภาพ
# ==========================================

def clean_board(gray):

    background = np.median(
        np.concatenate([
            gray[:20, :].flatten(),
            gray[-20:, :].flatten(),
            gray[:, :20].flatten(),
            gray[:, -20:].flatten()
        ])
    )

    # พื้นขาว = ตัวเลขมืด
    if background > 128:

        diagonal_image = has_diagonal_lines(gray)

        threshold = 128 if diagonal_image else 180

        binary = cv2.threshold(
            gray,
            threshold,
            255,
            cv2.THRESH_BINARY_INV
        )[1]

        if diagonal_image:
            # ภาพที่มีเส้นทแยง ใช้ตำแหน่ง Grid 450x450
            # ที่แน่นอน เพื่อไม่ให้เส้นตารางรวมกับตัวเลข
            for x in range(0, 451, 50):
                cv2.rectangle(
                    binary,
                    (max(0, x - 1), 0),
                    (min(449, x + 1), 449),
                    0,
                    -1
                )

            for y in range(0, 451, 50):
                cv2.rectangle(
                    binary,
                    (0, max(0, y - 1)),
                    (449, min(449, y + 1)),
                    0,
                    -1
                )

            yy, xx = np.indices((450, 450))
            diagonal = (
                (np.abs(yy - xx) <= 3)
                |
                (np.abs(yy - (449 - xx)) <= 3)
            )
            binary[diagonal] = 0

        else:
            horizontal_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (40, 1)
            )

            vertical_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (1, 40)
            )

            horizontal = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                horizontal_kernel
            )

            vertical = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                vertical_kernel
            )

            lines = cv2.bitwise_or(
                horizontal,
                vertical
            )

            binary = cv2.subtract(
                binary,
                lines
            )

        return cv2.bitwise_not(binary)

    # --------------------------------------
    # พื้นมืด = ตัวเลขสว่าง
    # --------------------------------------

    binary = cv2.threshold(
        gray,
        128,
        255,
        cv2.THRESH_BINARY
    )[1]

    # Purple / Dark image ต้องลบเส้นตาราง
    # โดยตรง เพราะเส้นกับตัวเลขมีสีสว่างเหมือนกัน
    for x in range(0, 451, 50):

        cv2.rectangle(
            binary,
            (
                max(0, x - 3),
                0
            ),
            (
                min(449, x + 3),
                449
            ),
            0,
            -1
        )

    for y in range(0, 451, 50):

        cv2.rectangle(
            binary,
            (
                0,
                max(0, y - 3)
            ),
            (
                449,
                min(449, y + 3)
            ),
            0,
            -1
        )

    return cv2.bitwise_not(binary)


# ==========================================
# OCR ทั้งกระดาน
# ==========================================

def ocr_boxes(image, psm=6):

    config = (
        f"--psm {psm} --oem 1 "
        "-c tessedit_char_whitelist=123456789"
    )

    try:
        text = pytesseract.image_to_boxes(
            image,
            config=config,
            timeout=5
        )
    except RuntimeError:
        return []

    boxes = []

    for line in text.splitlines():

        parts = line.split()

        if len(parts) < 5:
            continue

        char = parts[0]

        if char not in DIGITS:
            continue

        try:
            x1 = int(parts[1])
            y1 = int(parts[2])
            x2 = int(parts[3])
            y2 = int(parts[4])
        except ValueError:
            continue

        boxes.append(
            (
                int(char),
                x1,
                y1,
                x2,
                y2
            )
        )

    return boxes


# ==========================================
# อ่าน Sudoku
# ==========================================

def read_whole_board(image):

    gray = prepare_board(image)
    clean = clean_board(gray)

    boxes = ocr_boxes(
        clean,
        psm=6
    )

    # ถ้า OCR อ่านได้น้อยผิดปกติ
    # ให้ลอง PSM 11 เพิ่ม
    if len(boxes) < 15:

        boxes_11 = ocr_boxes(
            clean,
            psm=11
        )

        boxes.extend(boxes_11)

    board = [
        [0 for _ in range(9)]
        for _ in range(9)
    ]

    occupied_board = [
        [False for _ in range(9)]
        for _ in range(9)
    ]

    candidates_board = [
        [set(range(1, 10)) for _ in range(9)]
        for _ in range(9)
    ]

    confidence_board = [
        [0.0 for _ in range(9)]
        for _ in range(9)
    ]

    for number, x1, y1, x2, y2 in boxes:

        center_x = (x1 + x2) / 2
        center_y = 450 - (y1 + y2) / 2

        col = int(center_x / 50)
        row = int(center_y / 50)

        if not (
            0 <= row < 9
            and 0 <= col < 9
        ):
            continue

        # ถ้ามี OCR ซ้ำในช่องเดียวกัน
        # ไม่เขียนทับค่าเดิม
        if board[row][col] != 0:
            continue

        board[row][col] = number
        occupied_board[row][col] = True
        candidates_board[row][col] = {number}
        confidence_board[row][col] = 1.0

    return (
        board,
        occupied_board,
        candidates_board,
        confidence_board
    )


def is_gold_style(image):
    """ตรวจว่าภาพเป็น Sudoku พื้นทอง/น้ำตาลหรือไม่"""
    if isinstance(image, str):
        image = cv2.imread(image)
    if image is None:
        return False
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    hue = hsv[:, :, 0]
    colored = saturation > 60
    if not np.any(colored):
        return False
    median_hue = float(np.median(hue[colored]))
    median_sat = float(np.median(saturation[colored]))
    return 5 <= median_hue <= 35 and median_sat >= 60


def prepare_cell(cell):

    # เปลี่ยนเป็น Grayscale
    gray = cv2.cvtColor(
        cell,
        cv2.COLOR_BGR2GRAY
    )

    # อ่านขนาดภาพ
    h, w = gray.shape

    # ตัดขอบ Cell ออก
    margin = int(min(h, w) * 0.10)

    gray = gray[
        margin:h - margin,
        margin:w - margin
    ]

    # ขยายภาพ
    gray = cv2.resize(
        gray,
        (200, 200),
        interpolation=cv2.INTER_CUBIC
    )

    return gray


# ==========================================
# 2. หาค่าสีพื้นหลัง
# ==========================================

def get_background(gray):

    # ใช้บริเวณขอบของภาพ
    top = gray[:20, :]
    bottom = gray[-20:, :]
    left = gray[:, :20]
    right = gray[:, -20:]

    # รวม pixel ทั้งหมด
    border = np.concatenate([
        top.flatten(),
        bottom.flatten(),
        left.flatten(),
        right.flatten()
    ])

    # ใช้ค่ามัธยฐานเป็นสีพื้นหลัง
    background = np.median(border)

    return background


# ==========================================
# 3. สร้างภาพ Binary
# ==========================================

def make_masks(gray):

    background = get_background(gray)

    masks = []

    # --------------------------------------
    # พื้นสว่าง → ตัวเลขสีเข้ม
    # --------------------------------------

    if background > 128:

        thresholds = [
            80,
            110,
            140
        ]

        for threshold in thresholds:

            _, mask = cv2.threshold(
                gray,
                threshold,
                255,
                cv2.THRESH_BINARY_INV
            )

            masks.append(mask)

    # --------------------------------------
    # พื้นมืด → ตัวเลขสีสว่าง
    # --------------------------------------

    else:

        thresholds = [
            120,
            160,
            200
        ]

        for threshold in thresholds:

            _, mask = cv2.threshold(
                gray,
                threshold,
                255,
                cv2.THRESH_BINARY
            )

            masks.append(mask)

    return masks


# ==========================================
# 4. ลบ Noise
# ==========================================

def remove_noise(mask):

    # สร้าง Kernel ขนาดเล็ก
    kernel = np.ones(
        (2, 2),
        np.uint8
    )

    # ลบจุดเล็ก ๆ
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # หา Connected Components
    number, labels, stats, centers = \
        cv2.connectedComponentsWithStats(
            mask,
            8
        )

    clean = np.zeros_like(mask)

    h, w = mask.shape

    # ตรวจสอบ component ทีละตัว
    for i in range(1, number):

        width = stats[i, cv2.CC_STAT_WIDTH]

        height = stats[i, cv2.CC_STAT_HEIGHT]

        area = stats[i, cv2.CC_STAT_AREA]

        # คำนวณสัดส่วนความยาว
        ratio = max(width, height) / max(
            min(width, height),
            1
        )

        # ----------------------------------
        # ตัด component เล็กมาก
        # ----------------------------------

        if area < 40:
            continue

        # ----------------------------------
        # ตัดเส้นยาวและบาง
        # เช่น เส้นทแยง
        # ----------------------------------

        if ratio > 8:
            continue

        # ----------------------------------
        # ตัด component ที่ใหญ่เกินไป
        # ----------------------------------

        if width > w * 0.90:
            continue

        if height > h * 0.90:
            continue

        # เก็บ component ที่เหลือ
        clean[labels == i] = 255

    return clean


# ==========================================
# 5. ตรวจว่ามีตัวเลขหรือไม่
# ==========================================

def has_digit(mask):

    # นับจำนวน pixel สีขาว
    white_pixels = cv2.countNonZero(
        mask
    )

    # คำนวณสัดส่วน
    ratio = white_pixels / mask.size

    # ถ้าน้อยเกินไป = ไม่มีเลข
    if ratio < 0.01:

        return False

    # ถ้ามากเกินไป = น่าจะเป็น noise
    if ratio > 0.45:

        return False

    # หา contour
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # ถ้าไม่มี contour
    if len(contours) == 0:

        return False

    return True


# ==========================================
# 6. ตัดเฉพาะตัวเลข
# ==========================================

def crop_digit(mask):

    # หา pixel สีขาว
    points = cv2.findNonZero(mask)

    if points is None:

        return None

    # หา Bounding Box
    x, y, w, h = cv2.boundingRect(
        points
    )

    # ตรวจขนาด
    if w < 8 or h < 12:

        return None

    # เพิ่มพื้นที่รอบตัวเลข
    padding = 8

    x1 = max(
        0,
        x - padding
    )

    y1 = max(
        0,
        y - padding
    )

    x2 = min(
        mask.shape[1],
        x + w + padding
    )

    y2 = min(
        mask.shape[0],
        y + h + padding
    )

    digit = mask[
        y1:y2,
        x1:x2
    ]

    return digit


# ==========================================
# 7. จัดตัวเลขให้อยู่ตรงกลาง
# ==========================================

def normalize_digit(digit):

    # กลับสี
    digit = cv2.bitwise_not(
        digit
    )

    # สร้างพื้นหลังสีขาว
    canvas = np.full(
        (180, 180),
        255,
        dtype=np.uint8
    )

    h, w = digit.shape

    # กำหนดขนาดสูงสุด
    max_size = 120

    # คำนวณอัตราการย่อ/ขยาย
    scale = min(
        max_size / w,
        max_size / h
    )

    new_w = max(
        1,
        int(w * scale)
    )

    new_h = max(
        1,
        int(h * scale)
    )

    # Resize
    digit = cv2.resize(
        digit,
        (new_w, new_h),
        interpolation=cv2.INTER_CUBIC
    )

    # คำนวณตำแหน่งกึ่งกลาง
    x = (180 - new_w) // 2
    y = (180 - new_h) // 2

    # วางตัวเลขตรงกลาง
    canvas[
        y:y + new_h,
        x:x + new_w
    ] = digit

    return canvas


# ==========================================
# 8. OCR ตัวเลข
# ==========================================

def _ocr_data(image, psm):

    config = (
        f"--psm {psm} "
        "--oem 1 "
        "-c tessedit_char_whitelist=123456789 "
        "-c user_defined_dpi=300"
    )

    data = pytesseract.image_to_data(
        image,
        config=config,
        output_type=pytesseract.Output.DICT
    )

    results = []

    for text, confidence in zip(
        data["text"],
        data["conf"]
    ):

        text = text.strip()

        try:
            confidence = float(confidence)
        except ValueError:
            continue

        if (
            len(text) == 1
            and text in DIGITS
            and confidence >= 0
        ):
            results.append(
                (int(text), confidence)
            )

    return results


def ocr_digit(image):

    # --------------------------------------
    # วิธีที่ 1: OCR ภาพปกติ
    # --------------------------------------

    results = []

    for psm in [6, 10, 13]:
        results.extend(
            _ocr_data(image, psm)
        )

    # --------------------------------------
    # วิธีที่ 2: เพิ่มขอบสีขาว
    # --------------------------------------
    # บางตัวเลข เช่น 9 อาจชิดขอบมากเกินไป
    # ทำให้ Tesseract อ่านไม่ได้

    if len(results) == 0:

        padded = cv2.copyMakeBorder(
            image,
            20,
            20,
            20,
            20,
            cv2.BORDER_CONSTANT,
            value=255
        )

        padded = cv2.resize(
            padded,
            (300, 300),
            interpolation=cv2.INTER_CUBIC
        )

        for psm in [6, 10, 13]:
            results.extend(
                _ocr_data(padded, psm)
            )

    # --------------------------------------
    # ถ้ายังอ่านไม่ได้
    # --------------------------------------

    if len(results) == 0:
        return None, 0.0

    # --------------------------------------
    # รวมคะแนนของแต่ละเลข
    # --------------------------------------

    scores = {}

    for number, confidence in results:

        if number not in scores:
            scores[number] = []

        scores[number].append(
            confidence
        )

    # ใช้ค่าเฉลี่ย Confidence แทนการบวกคะแนนทั้งหมด
    # เพราะการบวกอาจทำให้ผล OCR ที่ผิดซ้ำหลาย Mask ชนะ
    # ผลที่ถูกต้องจาก Mask เดียวได้

    average_scores = {}

    for number in scores:
        average_scores[number] = sum(
            scores[number]
        ) / len(scores[number])

    best_number = max(
        average_scores,
        key=average_scores.get
    )

    best_confidence = max(
        scores[best_number]
    )

    return (
        best_number,
        best_confidence
    )


# ==========================================
# 9. อ่าน Cell หนึ่งช่อง
# ==========================================

def read_cell(
    cell,
    debug_name=None
):

    # เตรียมภาพ
    gray = prepare_cell(
        cell
    )

    # สร้าง Binary หลายแบบ
    masks = make_masks(
        gray
    )

    observations = []
    digit_detected = False

    # ทดลองทีละ Mask
    for index, mask in enumerate(masks):

        # ลบ Noise
        clean = remove_noise(
            mask
        )

        # ตรวจว่ามีเลขหรือไม่
        if not has_digit(clean):

            continue

        # พบลักษณะตัวเลขในภาพแล้ว
        digit_detected = True

        # ตัดเฉพาะตัวเลข
        digit = crop_digit(
            clean
        )

        if digit is None:

            continue

        # Normalize
        digit = normalize_digit(
            digit
        )

        # บันทึกภาพสำหรับ Debug
        if debug_name is not None:

            os.makedirs(
                DEBUG_DIR,
                exist_ok=True
            )

            filename = (
                f"{debug_name}_"
                f"mask{index}.png"
            )

            path = os.path.join(
                DEBUG_DIR,
                filename
            )

            cv2.imwrite(
                path,
                digit
            )

        # OCR
        number, confidence = ocr_digit(
            digit
        )

        if number is not None:

            observations.append(
                (
                    number,
                    confidence
                )
            )

    # --------------------------------------
    # ไม่มีเลข
    # --------------------------------------

    if len(observations) == 0:

        return {
            "occupied": digit_detected,
            "digit": None,
            "confidence": 0.0,
            "candidates": set(range(1, 10))
        }


    # --------------------------------------
    # รวมผล OCR
    # --------------------------------------

    scores = {}

    for number, confidence in observations:

        if number not in scores:
            scores[number] = []

        scores[number].append(
            confidence
        )


    # ใช้ค่าเฉลี่ย Confidence ของแต่ละเลข
    # เพื่อไม่ให้ผลผิดที่เกิดซ้ำหลาย Mask
    # ชนะผลที่ถูกต้องจาก Mask อื่น
    average_scores = {}

    for number in scores:
        average_scores[number] = sum(
            scores[number]
        ) / len(scores[number])


    # เลือกเลขที่มีค่าเฉลี่ยสูงสุด
    best_number = max(
        average_scores,
        key=average_scores.get
    )


    # หา Confidence สูงสุดของเลขนั้น
    best_confidence = max(
        confidence
        for number, confidence
        in observations
        if number == best_number
    )


    return {
        "occupied": True,
        "digit": best_number,
        "confidence": best_confidence / 100,
        "candidates": {
            best_number
        }
    }


# ==========================================
# 10. หา Cell ที่มีเลขซ้ำ
# ==========================================

def find_conflicts(board):

    conflicts = []


    # --------------------------------------
    # ตรวจแถว
    # --------------------------------------

    for row in range(9):

        positions = {}

        for col in range(9):

            number = board[row][col]

            if number == 0:
                continue

            if number not in positions:

                positions[number] = []

            positions[number].append(
                (row, col)
            )


        for number in positions:

            cells = positions[number]

            if len(cells) > 1:

                conflicts.append(
                    cells
                )


    # --------------------------------------
    # ตรวจคอลัมน์
    # --------------------------------------

    for col in range(9):

        positions = {}

        for row in range(9):

            number = board[row][col]

            if number == 0:
                continue

            if number not in positions:

                positions[number] = []

            positions[number].append(
                (row, col)
            )


        for number in positions:

            cells = positions[number]

            if len(cells) > 1:

                conflicts.append(
                    cells
                )


    # --------------------------------------
    # ตรวจกล่อง 3x3
    # --------------------------------------

    for start_row in range(
        0,
        9,
        3
    ):

        for start_col in range(
            0,
            9,
            3
        ):

            positions = {}


            for row in range(
                start_row,
                start_row + 3
            ):

                for col in range(
                    start_col,
                    start_col + 3
                ):

                    number = board[row][col]

                    if number == 0:
                        continue

                    if number not in positions:

                        positions[number] = []

                    positions[number].append(
                        (row, col)
                    )


            for number in positions:

                cells = positions[number]

                if len(cells) > 1:

                    conflicts.append(
                        cells
                    )


    return conflicts


# ==========================================
# 11. อ่าน Sudoku ทั้งกระดาน
# ==========================================

def read_gold_cells_board(cells):

    board = []

    occupied_board = []

    candidates_board = []

    confidence_board = []


    # --------------------------------------
    # OCR ทั้ง 81 ช่อง
    # --------------------------------------

    for row in range(9):

        board_row = []

        occupied_row = []

        candidates_row = []

        confidence_row = []


        for col in range(9):

            index = row * 9 + col

            result = read_cell(
                cells[index],
                f"r{row + 1}c{col + 1}"
            )


            # เก็บตัวเลข
            if (
                result["occupied"]
                and result["digit"] is not None
            ):

                board_row.append(
                    result["digit"]
                )

            else:

                board_row.append(0)


            # เก็บสถานะ
            occupied_row.append(
                result["occupied"]
            )


            # เก็บ Candidates
            candidates_row.append(
                result["candidates"]
            )


            # เก็บ Confidence
            confidence_row.append(
                result["confidence"]
            )


        board.append(
            board_row
        )

        occupied_board.append(
            occupied_row
        )

        candidates_board.append(
            candidates_row
        )

        confidence_board.append(
            confidence_row
        )


    # ======================================
    # ตรวจ OCR ที่ขัดแย้งกัน
    # ======================================

    while True:

        conflicts = find_conflicts(
            board
        )

        # ไม่มีเลขซ้ำแล้ว
        if len(conflicts) == 0:

            break


        print()
        print(
            "พบ OCR ที่ขัดแย้งกัน ⚠️"
        )


        # ----------------------------------
        # หา Cell ที่ Confidence ต่ำที่สุด
        # ----------------------------------

        worst_cell = None

        worst_confidence = 999


        for conflict in conflicts:

            for row, col in conflict:

                confidence = confidence_board[
                    row
                ][
                    col
                ]


                if confidence < worst_confidence:

                    worst_confidence = confidence

                    worst_cell = (
                        row,
                        col
                    )


        # ----------------------------------
        # ลบเลขที่น่าสงสัย
        # ----------------------------------

        if worst_cell is not None:

            row, col = worst_cell

            print(
                f"ลบ OCR ที่น่าสงสัย "
                f"R{row + 1}C{col + 1}"
            )

            print(
                f"Confidence: "
                f"{worst_confidence:.2f}"
            )


            board[row][col] = 0

            # ช่องนี้ยังมีเลขอยู่ในภาพจริง
            # จึงไม่เปลี่ยน occupied เป็น False

            confidence_board[row][col] = 0.0

            # ให้ Solver เลือกเลขใหม่
            candidates_board[row][col] = set(
                range(1, 10)
            )

        else:

            break


    return (
        board,
        occupied_board,
        candidates_board,
        confidence_board
    )





# ==========================================
# แสดง Candidates
# ==========================================

def print_candidates(candidates_board):

    print()
    print("OCR CANDIDATES")
    print("-" * 70)

    for row in candidates_board:

        values = []

        for candidates in row:

            if candidates:
                text = "".join(
                    str(x)
                    for x in sorted(candidates)
                )
            else:
                text = "."

            values.append(
                f"{text:9}"
            )

        print(" ".join(values))

    print("-" * 70)


# ==========================================
# แสดง Board
# ==========================================

def print_board(board):

    print()
    print("=" * 50)
    print("OCR BOARD")
    print("=" * 50)

    for row in board:

        print(
            " ".join(
                str(x) if x != 0 else "."
                for x in row
            )
        )

# =========================================================
# ROBUST CELL OCR FALLBACK
# =========================================================

def _detect_grid_lines(gray):
    """หาเส้นตารางจริง เพื่อแบ่ง Cell ให้ตรงกับภาพ"""
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=45,
        minLineLength=int(min(gray.shape) * 0.25),
        maxLineGap=15
    )

    vertical = []
    horizontal = []

    if lines is not None:
        for line in lines:
            coords = np.asarray(line).reshape(-1)
            if coords.size != 4:
                continue
            x1, y1, x2, y2 = coords.astype(int)
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            if dy > dx * 5:
                vertical.append((x1 + x2) / 2)
            elif dx > dy * 5:
                horizontal.append((y1 + y2) / 2)

    def make_positions(values):
        values = sorted(values)
        groups = []

        for value in values:
            if not groups or value - groups[-1][-1] > 7:
                groups.append([value])
            else:
                groups[-1].append(value)

        positions = [sum(group) / len(group) for group in groups]

        # รวมเส้นขอบหนา 2 ด้านให้เป็นเส้นเดียว
        merged = []
        for value in positions:
            if not merged or value - merged[-1] > 12:
                merged.append(value)
            else:
                merged[-1] = (merged[-1] + value) / 2

        # ถ้าหายไป 1-2 เส้น ให้เติมจากระยะเฉลี่ย
        if len(merged) == 8:
            step = float(np.median(np.diff(merged)))
            merged = [merged[0] - step] + merged + [merged[-1] + step]

        if len(merged) == 9:
            step = float(np.median(np.diff(merged)))
            merged = [merged[0] - step] + merged + [merged[-1] + step]

        if len(merged) != 10:
            return np.linspace(0, 449, 10)

        merged[0] = max(0, merged[0])
        merged[-1] = min(449, merged[-1])

        return np.array(merged, dtype=float)

    return make_positions(vertical), make_positions(horizontal)


def _ocr_clean_cell(cell, psm=6):
    """OCR ช่องเดียวแบบทนต่อเลขสีเขียว/ดำและเส้นทแยง"""
    if cell is None or cell.size == 0:
        return None, 0.0

    if len(cell.shape) == 3:
        color = cell.copy()
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
    else:
        gray = cell.copy()
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

    gray = cv2.resize(gray, (200, 200), interpolation=cv2.INTER_CUBIC)
    hsv = cv2.resize(hsv, (200, 200), interpolation=cv2.INTER_CUBIC)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    colored = ((saturation >= 60) & (value >= 70) & (value <= 255)).astype(np.uint8) * 255
    colored = cv2.morphologyEx(colored, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    colored_ratio = cv2.countNonZero(colored) / colored.size

    def ocr(binary, current_psm):
        if binary is None or cv2.countNonZero(binary) < 15:
            return None
        binary = cv2.copyMakeBorder(
            binary, 30, 30, 30, 30,
            cv2.BORDER_CONSTANT, value=255
        )
        binary = cv2.resize(binary, (300, 300), interpolation=cv2.INTER_CUBIC)
        config = (
            f"--psm {current_psm} --oem 1 "
            "-c tessedit_char_whitelist=123456789 "
            "-c user_defined_dpi=300"
        )
        try:
            text = pytesseract.image_to_string(binary, config=config, timeout=2).strip()
        except (RuntimeError, pytesseract.TesseractError):
            return None
        return int(text) if len(text) == 1 and text in DIGITS else None

    # --------------------------------------------------
    # สี: ใช้เป็นทางเลือกแรก เพราะเส้นทแยง/เส้น Grid เป็นสีเทา
    # --------------------------------------------------
    if colored_ratio >= 0.015:
        # ตัดเฉพาะ component ที่ใหญ่ที่สุดของตัวเลขสี
        # เพื่อไม่ให้เส้นทแยง/เส้น Grid รบกวน Tesseract
        n, labels, stats, _ = cv2.connectedComponentsWithStats(colored, 8)
        best = None
        best_area = 0
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area >= 15 and area > best_area and w < 180 and h < 190:
                best = (x, y, w, h)
                best_area = area

        colored_crop = colored
        if best is not None:
            x, y, w, h = best
            pad = 8
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(colored.shape[1], x + w + pad)
            y2 = min(colored.shape[0], y + h + pad)
            colored_crop = colored[y1:y2, x1:x2]

        value_read = ocr(colored_crop, 10)
        if value_read is not None:
            return value_read, 0.98
        value_read = ocr(colored_crop, 13)
        if value_read is not None:
            return value_read, 0.95

    # --------------------------------------------------
    # ตัวเลขดำ
    # --------------------------------------------------
    background = float(np.median(gray))
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 9
    )
    otsu = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    for binary, current_psm, score in [
        (otsu, 6, 0.90),
        (adaptive, 6, 0.85),
        (otsu, 13, 0.80),
    ]:
        value_read = ocr(binary, current_psm)
        if value_read is not None:
            return value_read, score

    return None, 0.0


def _read_robust_cells(image):
    """อ่าน Sudoku แบบ Cell-by-Cell โดยรักษาข้อมูลสีไว้สำหรับ OCR"""
    gray = prepare_board(image)

    if isinstance(image, str):
        color_board = cv2.imread(image)
    else:
        color_board = image.copy() if image is not None else None

    if color_board is None:
        color_board = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif color_board.shape[:2] != (450, 450):
        color_board = cv2.resize(color_board, (450, 450), interpolation=cv2.INTER_CUBIC)

    xs, ys = _detect_grid_lines(gray)

    # ถ้าตารางอยู่เต็มภาพและช่องประมาณ 50px
    # ใช้ตำแหน่งมาตรฐาน จะรักษาเลขตัวใหญ่ได้ดีกว่า
    spacing = float(np.median(np.diff(xs)))
    use_fixed_grid = spacing >= 48

    hsv = cv2.cvtColor(
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        cv2.COLOR_BGR2HSV
    )
    saturation = float(np.mean(hsv[:, :, 1]))

    # ใช้ PSM 13 เป็นหลัก เพราะแต่ละ Cell มีตัวเลขเดี่ยว
    # และทำงานได้ดีกับฟอนต์/ขนาด/สีที่แตกต่างกัน
    primary_psm = 6
    secondary_psm = 10

    board = [
        [0 for _ in range(9)]
        for _ in range(9)
    ]

    occupied = [
        [False for _ in range(9)]
        for _ in range(9)
    ]

    candidates = [
        [set(range(1, 10)) for _ in range(9)]
        for _ in range(9)
    ]

    confidence = [
        [0.0 for _ in range(9)]
        for _ in range(9)
    ]

    pending = []

    for row in range(9):
        for col in range(9):

            if use_fixed_grid:
                x1 = col * 50 + 5
                x2 = (col + 1) * 50 - 5
                y1 = row * 50 + 5
                y2 = (row + 1) * 50 - 5
            else:
                x1 = int(round(xs[col] + 5))
                x2 = int(round(xs[col + 1] - 5))
                y1 = int(round(ys[row] + 5))
                y2 = int(round(ys[row + 1] - 5))

            gray_cell = gray[y1:y2, x1:x2]
            ocr_cell = color_board[y1:y2, x1:x2]

            # ตรวจทั้งหมึกสีดำและตัวเลขสี
            background = float(np.median(gray_cell))
            hsv_cell = cv2.cvtColor(ocr_cell, cv2.COLOR_BGR2HSV)
            colored_ratio = float(np.mean(
                (hsv_cell[:, :, 1] >= 60) &
                (hsv_cell[:, :, 2] >= 70)
            ))

            if background > 128:
                ink_ratio = float(np.mean(gray_cell < 120))
            else:
                ink_ratio = float(np.mean(gray_cell > 150))

            # ช่องที่มีเลขสีอาจไม่มืดใน grayscale
            if max(ink_ratio, colored_ratio) < 0.025:
                continue

            digit, score = _ocr_clean_cell(
                ocr_cell,
                primary_psm
            )

            if digit is None:
                pending.append((row, col, ocr_cell))
                continue

            board[row][col] = digit
            occupied[row][col] = True
            candidates[row][col] = {digit}
            confidence[row][col] = score

    # ลอง OCR สำรองเฉพาะช่องที่ตรวจพบหมึก แต่ OCR รอบแรกอ่านไม่ได้
    for row, col, cell in pending:
        digit, score = _ocr_clean_cell(
            cell,
            secondary_psm
        )

        if digit is not None:
            board[row][col] = digit
            occupied[row][col] = True
            candidates[row][col] = {digit}
            confidence[row][col] = score

    # ถ้ายังมีช่องที่มีเลขชัดเจนแต่ OCR ไม่ได้
    # ลองใช้ Cell จากตำแหน่ง Grid จริงอีกครั้ง
    for row in range(9):
        for col in range(9):
            if board[row][col] != 0:
                continue

            # ใช้รอบนี้เฉพาะภาพที่ช่องประมาณ 50px
            # เพราะภาพช่องเล็กกว่า 48px มีความแม่นยำจากรอบแรกดีกว่า
            if not use_fixed_grid:
                continue

            x1 = int(round(xs[col] + 5))
            x2 = int(round(xs[col + 1] - 5))
            y1 = int(round(ys[row] + 5))
            y2 = int(round(ys[row + 1] - 5))
            gray_cell = gray[y1:y2, x1:x2]
            ocr_cell = color_board[y1:y2, x1:x2]

            background = float(np.median(gray_cell))
            hsv_cell = cv2.cvtColor(ocr_cell, cv2.COLOR_BGR2HSV)
            colored_ratio = float(np.mean(
                (hsv_cell[:, :, 1] >= 60) &
                (hsv_cell[:, :, 2] >= 70)
            ))

            if background > 128:
                ink_ratio = float(np.mean(gray_cell < 120))
            else:
                ink_ratio = float(np.mean(gray_cell > 150))

            if max(ink_ratio, colored_ratio) < 0.025:
                continue

            digit, score = _ocr_clean_cell(
                ocr_cell,
                13
            )

            if digit is not None:
                board[row][col] = digit
                occupied[row][col] = True
                candidates[row][col] = {digit}
                confidence[row][col] = score

    return (
        board,
        occupied,
        candidates,
        confidence
    )


def _board_is_valid_for_fallback(board):
    """ตรวจว่า OCR Board ไม่มีเลขซ้ำ"""
    for row in board:
        values = [x for x in row if x != 0]
        if len(values) != len(set(values)):
            return False

    for col in range(9):
        values = [
            board[row][col]
            for row in range(9)
            if board[row][col] != 0
        ]
        if len(values) != len(set(values)):
            return False

    for start_row in range(0, 9, 3):
        for start_col in range(0, 9, 3):
            values = []
            for row in range(start_row, start_row + 3):
                for col in range(start_col, start_col + 3):
                    value = board[row][col]
                    if value != 0:
                        values.append(value)

            if len(values) != len(set(values)):
                return False

    return True



def _ocr_component_digit(binary_component):
    """อ่าน Component ที่เป็นเลข 1 ตัว"""
    if binary_component is None or binary_component.size == 0:
        return None, 0.0

    digit = cv2.copyMakeBorder(
        binary_component, 30, 30, 30, 30,
        cv2.BORDER_CONSTANT, value=0
    )
    digit = cv2.resize(digit, (200, 200), interpolation=cv2.INTER_CUBIC)

    observations = []

    # PSM 6 มักแม่นกว่าสำหรับตัวเลขฟอนต์มาตรฐาน
    # PSM 13 ใช้เป็นตัวสำรองสำหรับเลขเดี่ยว
    for psm in [6, 13]:
        config = (
            f"--psm {psm} --oem 1 "
            "-c tessedit_char_whitelist=123456789 "
            "-c user_defined_dpi=300"
        )

        text = pytesseract.image_to_string(
            digit,
            config=config,
            timeout=3
        ).strip()

        if len(text) == 1 and text in DIGITS:
            observations.append(int(text))

    if not observations:
        return None, 0.0

    # ถ้า PSM ทั้งสองอ่านตรงกัน ให้ถือว่าน่าเชื่อถือมาก
    counts = {}
    for value in observations:
        counts[value] = counts.get(value, 0) + 1

    best = max(
        counts,
        key=lambda value: (counts[value], -observations.index(value))
    )

    confidence = 1.0 if counts[best] >= 2 else 0.85
    return best, confidence

def _read_component_board(image):
    """อ่าน Sudoku โดยหา Connected Components หลังลบเส้น Grid"""
    gray = prepare_board(image)
    background = float(np.median(np.concatenate([
        gray[:20, :].flatten(),
        gray[-20:, :].flatten(),
        gray[:, :20].flatten(),
        gray[:, -20:].flatten()
    ])))

    if background > 128:
        binary = cv2.threshold(
            gray, 180, 255, cv2.THRESH_BINARY_INV
        )[1]
    else:
        binary = cv2.threshold(
            gray, 128, 255, cv2.THRESH_BINARY
        )[1]

    # ลบเส้น Grid ตามตำแหน่ง 9x9 ของภาพที่ preprocess แล้ว
    # ใช้ 5 pixel เพื่อรองรับเส้นหนา
    for x in range(0, 451, 50):
        binary[:, max(0, x - 2):min(450, x + 3)] = 0

    for y in range(0, 451, 50):
        binary[max(0, y - 2):min(450, y + 3), :] = 0

    # ลบเส้นทแยงเฉพาะเมื่อเป็นเส้นยาวจริง ๆ
    # เพื่อไม่ให้ตัวเลขที่มีเส้นเฉียงถูกลบโดยเข้าใจผิดว่าเป็น Grid
    edges = cv2.Canny(gray, 50, 150)
    diagonal_count = 0
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=150,
        maxLineGap=15
    )

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(-1)
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dx == 0 or dy == 0:
                continue
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            length = float(np.hypot(dx, dy))
            if 35 <= angle <= 55 and length >= 150:
                diagonal_count += 1

    if diagonal_count >= 2:
        yy, xx = np.indices((450, 450))
        diagonal = (
            (np.abs(yy - xx) <= 3)
            |
            (np.abs(yy - (449 - xx)) <= 3)
        )
        binary[diagonal] = 0

    number, labels, stats, centers = cv2.connectedComponentsWithStats(
        binary, 8
    )

    board = [[0 for _ in range(9)] for _ in range(9)]
    occupied = [[False for _ in range(9)] for _ in range(9)]
    candidates = [[set(range(1, 10)) for _ in range(9)] for _ in range(9)]
    confidence = [[0.0 for _ in range(9)] for _ in range(9)]

    # แต่ละเลขควรเป็น Component แยกออกมา
    components = []
    for i in range(1, number):
        x, y, w, h, area = stats[i]

        if area < 70:
            continue
        if w < 5 or h < 10:
            continue
        if w > 45 or h > 48:
            continue

        # ตัวเลข 1 อาจแคบ แต่เส้น Grid ที่หลงเหลือจะกว้างมาก
        ratio = max(w, h) / max(min(w, h), 1)
        if ratio > 8:
            continue

        center_x, center_y = centers[i]
        row = int(center_y // 50)
        col = int(center_x // 50)

        if not (0 <= row < 9 and 0 <= col < 9):
            continue

        components.append((i, row, col, area, x, y, w, h))

    for i, row, col, area, x, y, w, h in components:
        component = np.zeros((h, w), dtype=np.uint8)
        component[labels[y:y+h, x:x+w] == i] = 255

        digit, score = _ocr_component_digit(component)
        if digit is None:
            continue

        # ถ้ามี OCR ซ้ำใน Cell เดียว ให้เก็บผลที่ confidence สูงกว่า
        if not occupied[row][col] or score > confidence[row][col]:
            board[row][col] = digit
            occupied[row][col] = True
            candidates[row][col] = {digit}
            confidence[row][col] = score

    # Component OCR สำเร็จเมื่อมีเลขอย่างน้อยพอสมควร
    count = sum(
        1 for row in range(9)
        for col in range(9)
        if occupied[row][col]
    )

    return board, occupied, candidates, confidence, count


def _board_is_valid_for_fallback(board):
    """ตรวจว่า Board ไม่มีเลขซ้ำตามกฎ Sudoku"""
    for row in board:
        values = [x for x in row if x != 0]
        if len(values) != len(set(values)):
            return False

    for col in range(9):
        values = [
            board[row][col]
            for row in range(9)
            if board[row][col] != 0
        ]
        if len(values) != len(set(values)):
            return False

    for start_row in range(0, 9, 3):
        for start_col in range(0, 9, 3):
            values = []
            for row in range(start_row, start_row + 3):
                for col in range(start_col, start_col + 3):
                    if board[row][col] != 0:
                        values.append(board[row][col])
            if len(values) != len(set(values)):
                return False

    return True


# =========================================================
# Final robust reader
# =========================================================

def _visual_occupied(image):
    """ตรวจช่องที่มีตัวเลขจริงจากภาพ โดยไม่พึ่ง OCR"""
    gray = prepare_board(image)
    if isinstance(image, str):
        color = cv2.imread(image)
    else:
        color = image.copy() if image is not None else None
    if color is None:
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif color.shape[:2] != (450, 450):
        color = cv2.resize(color, (450, 450), interpolation=cv2.INTER_CUBIC)

    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    diagonal_found = _has_diagonal_from_gray(gray)
    dark_bg = float(np.median(gray)) < 80
    result = [[False] * 9 for _ in range(9)]

    # สร้าง mask ของ "หมึก" โดยตัดขอบ Cell ออกแล้ว
    for r in range(9):
        for c in range(9):
            y1, y2 = r * 50 + 8, (r + 1) * 50 - 8
            x1, x2 = c * 50 + 8, (c + 1) * 50 - 8
            g = gray[y1:y2, x1:x2]
            h = hsv[y1:y2, x1:x2]

            if dark_bg:
                mask = g > 120
                # เส้นทแยงของภาพสีม่วง/ดำไม่ใช่ตัวเลข
                if diagonal_found:
                    yy, xx = np.indices(g.shape)
                    gy, gx = yy + y1, xx + x1
                    d = ((np.abs(gy - gx) <= 4) |
                         (np.abs(gy - (449 - gx)) <= 4))
                    mask[d] = False
            else:
                # ตัวเลขดำ/เทา
                dark = g < 110
                # ตัวเลขสี เช่น เขียว/น้ำเงิน
                colored = (h[:, :, 1] > 60) & (h[:, :, 2] < 250)
                mask = dark | colored

            # ลบ component เล็ก ๆ
            m = (mask.astype(np.uint8) * 255)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
            good_area = 0
            for i in range(1, n):
                x, y, w, hh, area = stats[i]
                if area < 8:
                    continue
                if w > 0.85 * m.shape[1] or hh > 0.85 * m.shape[0]:
                    continue
                ratio = max(w, hh) / max(1, min(w, hh))
                if ratio > 10:
                    continue
                good_area += area

            result[r][c] = good_area >= 20

    return result


def _strong_visual_occupied(image):
    """ตรวจเลข/หมึกจริงในพื้นที่กลาง Cell ด้วย pixel difference

    ใช้เป็น safety check เพิ่มจาก _visual_occupied เพราะภาพบางชุด
    มีเลขชิดขอบหรือมีสี/พื้นหลังที่ทำให้ detector เดิมพลาด
    """
    gray = prepare_board(image)

    if isinstance(image, str):
        color = cv2.imread(image)
    else:
        color = image.copy() if image is not None else None

    if color is None:
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif color.shape[:2] != (450, 450):
        color = cv2.resize(
            color,
            (450, 450),
            interpolation=cv2.INTER_CUBIC,
        )

    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    result = [[False] * 9 for _ in range(9)]

    for r in range(9):
        for c in range(9):
            y1, y2 = r * 50 + 9, (r + 1) * 50 - 9
            x1, x2 = c * 50 + 9, (c + 1) * 50 - 9

            cell = gray[y1:y2, x1:x2]
            hcell = hsv[y1:y2, x1:x2]

            if cell.size == 0:
                continue

            h, w = cell.shape
            k = max(3, min(h, w) // 5)

            corners = np.concatenate([
                cell[:k, :k].ravel(),
                cell[:k, -k:].ravel(),
                cell[-k:, :k].ravel(),
                cell[-k:, -k:].ravel(),
            ])

            background = float(np.median(corners))
            diff = np.abs(
                cell.astype(np.int16)
                - int(round(background))
            )

            # ใช้ทั้งความต่างของความสว่างและสี
            gray_ink = diff >= 30
            colored = (
                (hcell[:, :, 1] >= 45)
                & (np.abs(
                    hcell[:, :, 2].astype(np.int16)
                    - int(round(background))
                ) >= 20)
            )

            mask = (
                gray_ink | colored
            ).astype(np.uint8) * 255

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                np.ones((2, 2), np.uint8),
            )

            n, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask,
                8,
            )

            good_area = 0
            for i in range(1, n):
                x, y, ww, hh, area = stats[i]

                if area < 12:
                    continue

                if ww > 0.85 * mask.shape[1]:
                    continue

                if hh > 0.85 * mask.shape[0]:
                    continue

                ratio = max(ww, hh) / max(
                    1,
                    min(ww, hh),
                )

                if ratio > 10:
                    continue

                good_area += area

            result[r][c] = good_area >= 28

    return result


def _repair_visible_clues(image, board, occupied, candidates, confidence):
    """เติม clue ที่ภาพมีจริงแต่ Whole-board OCR พลาด

    ถ้าพบหมึกใน Cell ที่ board ยังเป็น 0 จะอ่าน Cell นั้นซ้ำโดยตรง
    และห้ามปล่อยให้ Solver เดาแทนเลขโจทย์ที่มองเห็นได้
    """
    strong = _strong_visual_occupied(image)
    repaired = 0
    unresolved = []

    for r in range(9):
        for c in range(9):
            if not strong[r][c]:
                continue

            if board[r][c] != 0:
                occupied[r][c] = True
                continue

            x1 = c * 50 + 5
            x2 = (c + 1) * 50 - 5
            y1 = r * 50 + 5
            y2 = (r + 1) * 50 - 5

            cell = image[y1:y2, x1:x2] if not isinstance(image, str) else None

            if cell is None:
                full = cv2.imread(image)
                if full is None:
                    unresolved.append((r, c))
                    continue
                cell = full[y1:y2, x1:x2]

            digit = None
            score = 0.0

            for psm in (6, 10, 13):
                value, current_score = _ocr_clean_cell(
                    cell,
                    psm,
                )
                if value is not None:
                    digit = value
                    score = current_score
                    break

            if digit is None:
                # read_cell มี preprocessing อีกชุดหนึ่ง จึงใช้เป็นรอบสุดท้าย
                result = read_cell(cell)
                digit = result.get("digit")
                score = float(result.get("confidence", 0.0))

            if digit is None:
                unresolved.append((r, c))
                continue

            board[r][c] = digit
            occupied[r][c] = True
            candidates[r][c] = {digit}
            confidence[r][c] = score
            repaired += 1

            print(
                f"Visible clue repair: "
                f"R{r + 1}C{c + 1} = {digit}"
            )

    if unresolved:
        print(
            "Visible clues ที่ OCR อ่านไม่ได้:",
            [(r + 1, c + 1) for r, c in unresolved],
        )

    return repaired, unresolved


def _has_diagonal_from_gray(gray):
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=50, minLineLength=120, maxLineGap=15
    )
    count = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(-1)
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            if dx == 0 or dy == 0:
                continue
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            if 35 <= angle <= 55 and np.hypot(dx, dy) >= 120:
                count += 1
    return count >= 2


def _read_standard_board(image):
    """อ่าน Whole-board OCR โดยไม่ลบ clue อัตโนมัติ"""
    board, occupied, candidates, confidence = read_whole_board(image)

    visual = _visual_occupied(image)
    for r in range(9):
        for c in range(9):
            if visual[r][c]:
                occupied[r][c] = True

    return board, occupied, candidates, confidence


def _read_gold_fast(image):
    """Gold/น้ำตาล: อ่านเฉพาะ Cell ที่มีหมึกสีดำ ลดเวลาจาก 81 Cell OCR เหลือเฉพาะช่องที่มีเลข"""
    gray = prepare_board(image)
    board = [[0] * 9 for _ in range(9)]
    occupied = [[False] * 9 for _ in range(9)]
    candidates = [[set(range(1, 10)) for _ in range(9)] for _ in range(9)]
    confidence = [[0.0] * 9 for _ in range(9)]

    for r in range(9):
        for c in range(9):
            cell = gray[r*50+6:(r+1)*50-6, c*50+6:(c+1)*50-6]
            # ช่องทองว่างไม่มีหมึกดำจำนวนมาก
            if np.mean(cell < 90) < 0.025:
                continue

            votes = []
            for threshold in (90, 130, 170):
                mask = (cell < threshold).astype(np.uint8) * 255
                mask = cv2.copyMakeBorder(mask, 20, 20, 20, 20,
                                          cv2.BORDER_CONSTANT, value=255)
                mask = cv2.resize(mask, (200, 200), interpolation=cv2.INTER_CUBIC)
                config = (
                    '--psm 10 --oem 1 '
                    '-c tessedit_char_whitelist=123456789 '
                    '-c user_defined_dpi=300'
                )
                try:
                    text = pytesseract.image_to_string(mask, config=config, timeout=2).strip()
                except RuntimeError:
                    continue
                if len(text) == 1 and text in DIGITS:
                    votes.append(int(text))

            occupied[r][c] = True
            if votes:
                counts = {n: votes.count(n) for n in set(votes)}
                digit = max(counts, key=lambda n: (counts[n], n))
                board[r][c] = digit
                candidates[r][c] = {digit}
                confidence[r][c] = counts[digit] / len(votes)

    return board, occupied, candidates, confidence


def is_blue_style(image):
    """ตรวจว่ากระดานใช้เลขสีน้ำเงินเป็น clue หรือไม่"""
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image.copy() if image is not None else None

    if img is None:
        return False

    if img.shape[:2] != (450, 450):
        img = cv2.resize(img, (450, 450), interpolation=cv2.INTER_CUBIC)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, sat, val = cv2.split(hsv)
    blue = (h >= 115) & (h <= 125) & (sat >= 70) & (val >= 80)
    ratio = float(np.mean(blue))

    if not np.any(blue):
        return False

    # กระดานสีน้ำเงินของโจทย์นี้มีหมึกสีน้ำเงินประมาณ 2-4% ของภาพ
    # ภาพพื้นหลังม่วงมีพื้นที่สีช่วงนี้มากกว่ามาก จึงกันไว้ไม่ให้
    # ถูกเข้าโหมด Blue OCR โดยไม่ตั้งใจ
    hue_median = float(np.median(h[blue]))
    return 0.01 <= ratio <= 0.08 and 118 <= hue_median <= 122


def _read_blue_fast(image):
    """อ่านกระดานที่ clue เป็นสีน้ำเงินด้วย Cell OCR โดยตรง

    ภาพประเภทนี้มีสีของ clue แยกจากเส้นตารางชัดเจน
    จึงไม่ควรใช้ Whole-board OCR ที่เคยทำให้ตำแหน่ง/ตัวเลขเพี้ยน
    """
    if isinstance(image, str):
        color = cv2.imread(image)
    else:
        color = image.copy() if image is not None else None

    if color is None:
        raise ValueError("ไม่สามารถอ่านภาพได้")

    if color.shape[:2] != (450, 450):
        color = cv2.resize(color, (450, 450), interpolation=cv2.INTER_CUBIC)

    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    blue_mask = (
        (hsv[:, :, 0] >= 105) &
        (hsv[:, :, 0] <= 135) &
        (hsv[:, :, 1] >= 70) &
        (hsv[:, :, 2] >= 80)
    ).astype(np.uint8) * 255

    board = [[0] * 9 for _ in range(9)]
    occupied = [[False] * 9 for _ in range(9)]
    candidates = [[set(range(1, 10)) for _ in range(9)] for _ in range(9)]
    confidence = [[0.0] * 9 for _ in range(9)]

    for r in range(9):
        for c in range(9):
            # เว้นขอบ cell เล็กน้อยเพื่อไม่ให้เส้น grid เข้า OCR
            x1, x2 = c * 50 + 2, (c + 1) * 50 - 2
            y1, y2 = r * 50 + 2, (r + 1) * 50 - 2
            cell = color[y1:y2, x1:x2]
            mask = blue_mask[y1:y2, x1:x2]

            ratio = cv2.countNonZero(mask) / float(mask.size)
            if ratio < 0.008:
                continue

            digit, score = _ocr_clean_cell(cell, 10)
            if digit is None:
                digit, score = _ocr_clean_cell(cell, 6)
            if digit is None:
                digit, score = _ocr_clean_cell(cell, 13)

            # มีหมึกสีน้ำเงินจริง แต่ OCR อ่านไม่ได้: อย่าปล่อยให้
            # reader อื่นเดาเลขดำ/เลขจากเส้น grid แทน
            occupied[r][c] = True

            if digit is not None:
                board[r][c] = digit
                candidates[r][c] = {digit}
                confidence[r][c] = score

    # ตรวจว่า OCR ไม่เกิด conflict จากเลขสีน้ำเงิน
    return board, occupied, candidates, confidence

def read_board(image, force_cell=False):
    """อ่าน Sudoku และยืนยัน clue ที่มองเห็นก่อนส่งให้ Solver."""

    def finalize(result, label):
        board, occupied, candidates, confidence = result

        repaired, unresolved = _repair_visible_clues(
            image,
            board,
            occupied,
            candidates,
            confidence,
        )

        if unresolved:
            print(
                f"{label}: ยังมี clue ที่อ่านไม่ได้ "
                "-> ไม่เลือก board นี้"
            )
            return None

        if not _clues_are_valid(board):
            print(
                f"{label}: OCR มีเลขขัดแย้ง "
                "-> ไม่เลือก board นี้"
            )
            return None

        if repaired:
            print(
                f"{label}: ซ่อม clue ที่ OCR พลาด {repaired} ช่อง"
            )

        print(f"OK: {label}")
        return board, occupied, candidates, confidence

    if is_gold_style(image):
        return finalize(
            _read_gold_fast(image),
            "Gold Cell OCR",
        )

    if is_blue_style(image):
        print("Blue Sudoku -> Blue Cell OCR")
        return finalize(
            _read_blue_fast(image),
            "Blue Cell OCR",
        )

    visual = _visual_occupied(image)

    if not force_cell:
        try:
            result = _read_standard_board(image)

            missing = [
                (r, c)
                for r in range(9)
                for c in range(9)
                if visual[r][c] and result[0][r][c] == 0
            ]

            if missing:
                print(
                    "Whole-board OCR มี clue ที่อ่านไม่ครบ "
                    f"({len(missing)} ช่อง) -> ตรวจซ้ำด้วย Cell OCR"
                )
            else:
                finalized = finalize(
                    result,
                    "Whole-board OCR",
                )
                if finalized is not None:
                    return finalized

        except Exception as e:
            print(f"Whole-board OCR fallback: {e}")

    try:
        result = _read_robust_cells(image)
        finalized = finalize(
            result,
            "Cell-by-cell OCR",
        )
        if finalized is not None:
            return finalized

        print(
            "Cell OCR ยังไม่ผ่าน -> Component OCR"
        )

    except Exception as e:
        print(f"Cell OCR fallback: {e}")

    board, occupied, candidates, confidence, count = _read_component_board(image)

    finalized = finalize(
        (board, occupied, candidates, confidence),
        f"Component OCR ({count} cells)",
    )

    if finalized is None:
        raise ValueError(
            "OCR อ่าน clue ที่มองเห็นไม่ครบหรือเกิดเลขขัดแย้ง"
        )

    return finalized

