import cv2

from image_preprocessor import preprocess

from digit_reader import (
    read_board,
    print_candidates,
    read_cell,
)

from sudoku_recovery import (
    solve_with_candidates,
    solution_is_valid,
    solution_preserves_clues
)


def _print_board(title, board):

    print(title)

    for row in board:
        print(
            " ".join(
                str(x) if x else "."
                for x in row
            )
        )


def _verify_recovered_solution_with_cells(processed, original_board, solution):
    """ตรวจ clue ที่ Recovery เปลี่ยนด้วย OCR ของ Cell โดยตรง"""
    image = cv2.imread(processed)
    if image is None:
        return False

    changed = []
    for r in range(9):
        for c in range(9):
            if (
                original_board[r][c] != 0
                and original_board[r][c] != solution[r][c]
            ):
                changed.append((r, c))

    if not changed:
        return False

    print(
        "Recovery changed cells:",
        [(r + 1, c + 1) for r, c in changed],
    )

    for r, c in changed:
        x1 = c * 50 + 5
        x2 = (c + 1) * 50 - 5
        y1 = r * 50 + 5
        y2 = (r + 1) * 50 - 5

        result = read_cell(image[y1:y2, x1:x2])
        detected = result.get("digit")

        print(
            f"Recovery verify R{r + 1}C{c + 1}: "
            f"original={original_board[r][c]}, "
            f"solution={solution[r][c]}, "
            f"cell_ocr={detected}"
        )

        if detected != solution[r][c]:
            return False

    return True


def solve_image(image_path):

    processed = None

    try:

        print("[1/4] กำลังเตรียมภาพ...")

        processed = preprocess(image_path)

        # ======================================
        # รอบที่ 1: Whole-board / normal OCR
        # ======================================

        print(
            "[2/4] กำลังอ่าน Sudoku "
            "ทั้งกระดาน..."
        )

        (
            board,
            occupied,
            candidates,
            confidence
        ) = read_board(processed)

        print_candidates(candidates)

        _print_board(
            "[3/4] OCR Board",
            board
        )

        # เก็บ clue จากภาพไว้แบบ immutable
        # Recovery อาจแก้ board สำหรับค้นหาคำตอบได้
        # แต่ห้ามใช้ board ที่ถูก Recovery แก้แล้วมาตัดสินว่า
        # clue ในภาพต้นฉบับถูกต้องหรือไม่
        original_board = [
            row.copy()
            for row in board
        ]

        # ======================================
        # Solver + OCR Recovery
        # ======================================

        print("[4/4] กำลังแก้ Sudoku...")

        solution = solve_with_candidates(
            board,
            candidates,
            confidence
        )

        # ======================================
        # ตรวจรอบแรกกับ clue ดั้งเดิม
        #
        # Whole-board OCR อาจอ่านเลขผิด แต่ยังสร้าง
        # Sudoku ที่มีคำตอบได้จาก Recovery
        # ถ้า solution ไม่รักษา clue ที่ OCR อ่านมา
        # ห้ามรับคำตอบนั้น ให้เปลี่ยนไปใช้ Cell OCR
        # ซึ่งอ่านทีละช่องละเอียดกว่า
        # ======================================

        first_solution_accepted = (
            solution is not None
            and solution_is_valid(solution)
            and solution_preserves_clues(
                original_board,
                solution,
            )
        )

        if not first_solution_accepted:

            if solution is not None:
                print(
                    "Whole-board OCR ได้ Solution "
                    "แต่ไม่รักษา clue เดิม -> ใช้ Cell OCR ตรวจซ้ำ"
                )
            else:
                print(
                    "Whole-board OCR / Recovery "
                    "ไม่สำเร็จ -> ใช้ Cell OCR ตรวจซ้ำ"
                )

            (
                board,
                occupied,
                candidates,
                confidence
            ) = read_board(
                processed,
                force_cell=True
            )

            print_candidates(candidates)

            _print_board(
                "OCR Board (Cell OCR)",
                board
            )

            # Cell OCR เป็น board ใหม่ จึงต้องบันทึก clue ชุดนี้ใหม่
            original_board = [
                row.copy()
                for row in board
            ]

            solution = solve_with_candidates(
                board,
                candidates,
                confidence
            )

            # ถ้า OCR รอบปกติยังหา solution ไม่ได้
            # ใช้ Verified Cell OCR เป็นรอบสุดท้าย โดยอ่านเฉพาะ
            # ช่องที่ภาพยืนยันว่ามี clue จริง
            if solution is None:
                print(
                    "Solver รอบปกติไม่สำเร็จ -> "
                    "ใช้ Verified Cell OCR รอบสุดท้าย"
                )

                (
                    verified_board,
                    verified_occupied,
                    verified_candidates,
                    verified_confidence
                ) = _read_verified_cell_board(processed)

                _print_board(
                    "OCR Board (Verified Cell OCR)",
                    verified_board
                )

                verified_solution = solve_with_candidates(
                    verified_board,
                    verified_candidates,
                    verified_confidence
                )

                if verified_solution is not None:
                    board = verified_board
                    occupied = verified_occupied
                    candidates = verified_candidates
                    confidence = verified_confidence
                    original_board = [
                        row.copy()
                        for row in board
                    ]
                    solution = verified_solution

        # ======================================
        # ไม่มี Solution
        # ======================================

        if solution is None:

            print(
                "Solution validation failed: "
                "ไม่มีคำตอบ"
            )

            return None

        # ======================================
        # ตรวจ Solution
        # ======================================

        if not solution_is_valid(solution):

            print(
                "Solution validation failed: "
                "Sudoku ไม่ถูกต้อง"
            )

            return None

        # ======================================
        # ตรวจ Solution กับ clue ดั้งเดิมจากภาพ
        # ห้ามใช้ board หลัง OCR Recovery เพราะ Recovery อาจเปลี่ยน clue
        # ทำให้คำตอบปลอมดูเหมือนรักษา clue ได้
        # ======================================

        if not solution_preserves_clues(
            original_board,
            solution
        ):
            if not _verify_recovered_solution_with_cells(
                processed,
                original_board,
                solution,
            ):
                print(
                    "Solution validation failed: "
                    "Recovery ไม่ได้รับการยืนยันจากภาพ"
                )
                return None

        sudoku_image = cv2.imread(
            processed
        )

        if sudoku_image is None:
            return None

        print("แก้ Sudoku สำเร็จ! ✅")

        return (
            sudoku_image,
            board,
            occupied,
            candidates,
            confidence,
            solution,
        )

    finally:

        if processed:

            try:
                import os

                if os.path.isfile(processed):
                    os.remove(processed)

            except OSError:
                pass
