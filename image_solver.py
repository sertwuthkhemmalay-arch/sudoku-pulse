import cv2

from image_preprocessor import preprocess

from digit_reader import (
    read_board,
    print_candidates
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

        # เก็บ clue ที่ OCR ยืนยันจากภาพไว้แยกต่างหาก
        # Recovery อาจแก้ board ภายในของมันเอง แต่ห้ามแก้เลขโจทย์จริง
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
        # ถ้ารอบแรกไม่ได้ -> Cell OCR
        # ======================================

        if solution is None:

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

            original_board = [
                row.copy()
                for row in board
            ]

            solution = solve_with_candidates(
                board,
                candidates,
                confidence
            )

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
        # ตรวจว่า clue ที่เหลืออยู่ไม่ถูกเปลี่ยน
        # ======================================

        if not solution_preserves_clues(
            original_board,
            solution
        ):

            print(
                "Solution validation failed: "
                "clue ถูกเปลี่ยน"
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
            solution
        )

    finally:

        if processed:

            try:
                import os

                if os.path.isfile(processed):
                    os.remove(processed)

            except OSError:
                pass
