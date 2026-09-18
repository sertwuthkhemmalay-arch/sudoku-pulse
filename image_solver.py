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

        corrected_cells = []

        for r in range(9):
            for c in range(9):
                if original_board[r][c] != board[r][c]:
                    corrected_cells.append((r, c))

        if corrected_cells:
            print(
                "OCR Recovery แก้ clue:",
                [
                    (
                        r + 1,
                        c + 1,
                        original_board[r][c],
                        board[r][c],
                        solution[r][c],
                    )
                    for r, c in corrected_cells
                ],
            )

        print("แก้ Sudoku สำเร็จ! ✅")

        return (
            sudoku_image,
            board,
            occupied,
            candidates,
            confidence,
            solution,
            corrected_cells,
        )

    finally:

        if processed:

            try:
                import os

                if os.path.isfile(processed):
                    os.remove(processed)

            except OSError:
                pass
