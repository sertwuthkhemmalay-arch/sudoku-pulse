def board_is_valid(board):

    if board is None or len(board) != 9:
        return False

    if any(
        len(row) != 9
        for row in board
    ):
        return False

    for row in board:

        values = [
            x
            for x in row
            if x != 0
        ]

        if any(
            x not in range(1, 10)
            for x in values
        ):
            return False

        if len(values) != len(set(values)):
            return False

    for col in range(9):

        values = [
            board[r][col]
            for r in range(9)
            if board[r][col] != 0
        ]

        if len(values) != len(set(values)):
            return False

    for sr in range(0, 9, 3):

        for sc in range(0, 9, 3):

            values = [
                board[r][c]

                for r in range(
                    sr,
                    sr + 3
                )

                for c in range(
                    sc,
                    sc + 3
                )

                if board[r][c] != 0
            ]

            if len(values) != len(set(values)):
                return False

    return True


def _possible(
    board,
    row,
    col
):

    used = set(
        board[row]
    )

    used.update(
        board[r][col]
        for r in range(9)
    )

    sr = (
        row // 3
    ) * 3

    sc = (
        col // 3
    ) * 3

    used.update(
        board[r][c]

        for r in range(
            sr,
            sr + 3
        )

        for c in range(
            sc,
            sc + 3
        )
    )

    return [
        n
        for n in range(1, 10)
        if n not in used
    ]


def solve(board):

    best_cell = None
    best_options = None

    for row in range(9):

        for col in range(9):

            if board[row][col] != 0:
                continue

            options = _possible(
                board,
                row,
                col
            )

            if not options:
                return False

            if (
                best_options is None
                or
                len(options)
                <
                len(best_options)
            ):

                best_cell = (
                    row,
                    col
                )

                best_options = options

                if len(options) == 1:
                    break

        if (
            best_options is not None
            and
            len(best_options) == 1
        ):
            break

    if best_cell is None:
        return True

    row, col = best_cell

    for number in best_options:

        board[row][col] = number

        if solve(board):
            return True

        board[row][col] = 0

    return False


def _solve_copy(board):

    test = [
        row.copy()
        for row in board
    ]

    if not board_is_valid(
        test
    ):
        return None

    if solve(test):
        return test

    return None


def solution_is_valid(
    solution
):

    if (
        solution is None
        or
        len(solution) != 9
    ):
        return False

    for row in solution:

        if (
            len(row) != 9
            or
            set(row)
            !=
            set(range(1, 10))
        ):
            return False

    for col in range(9):

        if (
            set(
                solution[r][col]
                for r in range(9)
            )
            !=
            set(range(1, 10))
        ):
            return False

    for sr in range(0, 9, 3):

        for sc in range(0, 9, 3):

            values = [
                solution[r][c]

                for r in range(
                    sr,
                    sr + 3
                )

                for c in range(
                    sc,
                    sc + 3
                )
            ]

            if (
                set(values)
                !=
                set(range(1, 10))
            ):
                return False

    return True


def solution_preserves_clues(
    board,
    solution
):

    if solution is None:
        return False

    for r in range(9):

        for c in range(9):

            if (
                board[r][c] != 0
                and
                board[r][c]
                != solution[r][c]
            ):
                return False

    return True


def _recover_single_digit_ocr(
    board,
    confidence=None,
    max_cells=4
):

    import itertools

    if board is None:
        return None

    conflict_cells = set()

    def add_conflicts(values):

        positions = {}

        for r, c, value in values:

            if value == 0:
                continue

            positions.setdefault(
                value,
                []
            ).append(
                (r, c)
            )

        for cells in positions.values():

            if len(cells) > 1:

                conflict_cells.update(
                    cells
                )

    for r in range(9):

        add_conflicts([
            (
                r,
                c,
                board[r][c]
            )

            for c in range(9)
        ])

    for c in range(9):

        add_conflicts([
            (
                r,
                c,
                board[r][c]
            )

            for r in range(9)
        ])

    for sr in range(0, 9, 3):

        for sc in range(0, 9, 3):

            add_conflicts([
                (
                    r,
                    c,
                    board[r][c]
                )

                for r in range(
                    sr,
                    sr + 3
                )

                for c in range(
                    sc,
                    sc + 3
                )
            ])

    if not conflict_cells:
        return None

    if confidence is not None:

        ordered = sorted(
            conflict_cells,
            key=lambda p:
                confidence[
                    p[0]
                ][
                    p[1]
                ]
        )

    else:

        ordered = list(
            conflict_cells
        )

    ordered = ordered[
        :max_cells
    ]

    # ======================================
    # เปลี่ยน 1 ช่อง
    # ======================================

    for r, c in ordered:

        original = board[r][c]

        for number in range(1, 10):

            if number == original:
                continue

            test = [
                row.copy()
                for row in board
            ]

            test[r][c] = number

            if not board_is_valid(
                test
            ):
                continue

            solution = _solve_copy(
                test
            )

            if solution is not None:

                print(
                    f"OCR Recovery: "
                    f"R{r + 1}C{c + 1} "
                    f"{original} -> {number}"
                )

                return (
                    solution,
                    test
                )

    # ======================================
    # เปลี่ยน 2 ช่อง
    # ======================================

    if len(ordered) >= 2:

        for cells in itertools.combinations(
            ordered,
            2
        ):

            (
                (r1, c1),
                (r2, c2)
            ) = cells

            old1 = board[r1][c1]
            old2 = board[r2][c2]

            for n1 in range(1, 10):

                if n1 == old1:
                    continue

                for n2 in range(1, 10):

                    if n2 == old2:
                        continue

                    test = [
                        row.copy()
                        for row in board
                    ]

                    test[r1][c1] = n1
                    test[r2][c2] = n2

                    if not board_is_valid(
                        test
                    ):
                        continue

                    solution = _solve_copy(
                        test
                    )

                    if solution is not None:

                        print(
                            "OCR Recovery: "
                            f"R{r1 + 1}C{c1 + 1} "
                            f"{old1}->{n1}, "
                            f"R{r2 + 1}C{c2 + 1} "
                            f"{old2}->{n2}"
                        )

                        return (
                            solution,
                            test
                        )

    return None



def _recover_unsolved_ocr(
    board,
    confidence=None,
    max_cells=4
):
    """
    Recovery สำหรับกรณี OCR อ่านเลขผิด
    แต่เลขที่ผิดยังไม่ทำให้เกิด conflict
    ทำให้ board_is_valid() เป็น True แต่ Sudoku ไม่มีคำตอบ

    วิธีทำ:
    1. เรียง clue ตาม confidence ต่ำ -> สูง
    2. ทดลองลบ clue ที่ไม่น่าเชื่อถือ 1-4 ช่อง
    3. ให้ Sudoku solver หา solution ใหม่
    4. ถ้าพบ solution ให้คืนทั้ง solution และ board ที่ใช้จริง
    """

    import itertools

    if board is None:
        return None

    cells = []

    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                continue

            score = 0.0

            if confidence is not None:
                try:
                    score = float(confidence[r][c])
                except Exception:
                    score = 0.0

            cells.append((score, r, c))

    if not cells:
        return None

    # ทดลอง clue ที่ confidence ต่ำที่สุดก่อน
    cells.sort(key=lambda item: item[0])
    cells = cells[:max_cells]

    print("OCR Recovery: เริ่มตรวจ clue confidence ต่ำ")
    print(
        "Recovery cells:",
        [
            (
                r + 1,
                c + 1,
                round(score, 3)
            )
            for score, r, c in cells
        ]
    )

    # ลบ 1, 2, 3, ... ช่อง แล้วลอง solve
    for count in range(1, len(cells) + 1):

        print(
            f"OCR Recovery: ลองลบ {count} ช่อง"
        )

        for combination in itertools.combinations(cells, count):

            test = [
                row.copy()
                for row in board
            ]

            changed = []

            for score, r, c in combination:
                old = test[r][c]
                changed.append((r, c, old, score))
                test[r][c] = 0

            if not board_is_valid(test):
                continue

            solution = _solve_copy(test)

            if solution is not None:

                print("OCR Recovery SUCCESS:")

                for r, c, old, score in changed:
                    print(
                        f"  R{r + 1}C{c + 1}: "
                        f"{old} -> ? "
                        f"(confidence={score:.3f})"
                    )

                return solution, test

    return None


def _ocr_alternative_map():
    # Tesseract ที่ใช้กับฟอนต์ของภาพชุดนี้มี 3/6 -> 8
    # เป็นความคลาดเคลื่อนที่พบจริงในภาพ images (4).png
    return {
        3: (8,),
        6: (8,),
        4: (5,),
    }


def _conflict_cells(board):
    cells = set()

    def collect(values):
        positions = {}
        for r, c, value in values:
            if value == 0:
                continue
            positions.setdefault(value, []).append((r, c))
        for group in positions.values():
            if len(group) > 1:
                cells.update(group)

    for r in range(9):
        collect([(r, c, board[r][c]) for c in range(9)])

    for c in range(9):
        collect([(r, c, board[r][c]) for r in range(9)])

    for sr in range(0, 9, 3):
        for sc in range(0, 9, 3):
            collect([
                (r, c, board[r][c])
                for r in range(sr, sr + 3)
                for c in range(sc, sc + 3)
            ])

    return cells


def _recover_by_ocr_alternatives(board, candidates=None, confidence=None, max_changes=4):
    """กู้ OCR ที่อ่านผิดแบบไม่เกิดเลขซ้ำหรือเกิดเลขซ้ำ

    จะลองเฉพาะ alternative ที่กำหนดไว้ และให้ cell ที่เป็น conflict
    ถูกตรวจเป็นอันดับแรก เพื่อไม่ให้ solver เปลี่ยน clue ที่ถูกโดยไม่จำเป็น
    """
    if board is None:
        return None

    alternatives = _ocr_alternative_map()
    conflict = _conflict_cells(board)
    cells = []

    for r in range(9):
        for c in range(9):
            value = board[r][c]
            if value == 0 or value not in alternatives:
                continue

            opts = set(alternatives[value])
            if candidates is not None:
                try:
                    opts.update(candidates[r][c])
                except Exception:
                    pass
            opts.discard(value)
            if not opts:
                continue

            score = 0.0
            if confidence is not None:
                try:
                    score = float(confidence[r][c])
                except Exception:
                    score = 0.0

            cells.append((0 if (r, c) in conflict else 1, score, r, c, tuple(sorted(opts))))

    # Conflict ก่อน และ confidence ต่ำก่อน; row/col เป็น tie-break
    cells.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    cells = cells[:12]

    if not cells:
        return None

    print('OCR Recovery: alternative search')
    print('Alternative cells:', [(r + 1, c + 1, board[r][c], opts)
                                 for _, _, r, c, opts in cells])

    # DFS แบบ best-first: ทดลองคำตอบเดิมก่อนในกรณีที่ยังใช้ได้
    # แต่สำหรับ conflict จะลอง alternative ก่อน เพื่อแก้ conflict ทันที
    def search(index, changes, working):
        exact = _solve_copy(working)
        if exact is not None:
            return exact, [row.copy() for row in working]

        if changes >= max_changes or index >= len(cells):
            return None

        _, _, r, c, opts = cells[index]
        original = working[r][c]

        for value in opts:
            test = [row.copy() for row in working]
            test[r][c] = value
            if not board_is_valid(test):
                continue
            result = search(index + 1, changes + 1, test)
            if result is not None:
                print(f'OCR Recovery: R{r + 1}C{c + 1} {original} -> {value}')
                return result

        # ข้าม cell นี้ได้
        result = search(index + 1, changes, working)
        if result is not None:
            return result
        return None

    return search(0, 0, [row.copy() for row in board])


def solve_with_candidates(board, candidates=None, confidence=None):
    if board is None:
        return None

    # 1) ถ้า OCR board ถูกกฎ ให้ลอง solve ตรง ๆ ก่อนเสมอ
    if board_is_valid(board):
        solution = _solve_copy(board)
        if solution is not None:
            return solution

    # 2) ถ้า board ไม่ conflict แต่ Sudoku ไม่มีคำตอบ
    # อาจเกิดจาก OCR อ่านเลขผิดเป็นเลขอื่นที่ยังไม่เกิดเลขซ้ำ
    # ให้ใช้ recovery ตาม confidence ก่อน
    recovered = _recover_unsolved_ocr(
        board,
        confidence=confidence,
        max_cells=4,
    )

    if recovered is not None:
        solution, recovered_board = recovered
        board[:] = [row.copy() for row in recovered_board]

        if solution_is_valid(solution) and solution_preserves_clues(
            board,
            solution,
        ):
            print("OCR Recovery: unsolved recovery สำเร็จ")
            return solution

    # 3) ถ้ามีเลขซ้ำจาก OCR ให้ลองแก้ conflict
    recovered = _recover_single_digit_ocr(
        board,
        confidence=confidence,
        max_cells=4,
    )

    if recovered is not None:
        solution, recovered_board = recovered
        board[:] = [row.copy() for row in recovered_board]

        if solution_is_valid(solution) and solution_preserves_clues(
            board,
            solution,
        ):
            print("OCR Recovery: conflict recovery สำเร็จ")
            return solution

    # 4) ใช้ mapping ของ OCR ที่รู้จากภาพชุดนี้
    recovered = _recover_by_ocr_alternatives(
        board,
        candidates=candidates,
        confidence=confidence,
        max_changes=4,
    )

    if recovered is None:
        print('OCR Recovery ไม่สำเร็จ')
        return None

    solution, recovered_board = recovered

    board[:] = [row.copy() for row in recovered_board]

    if not solution_is_valid(solution):
        return None

    if not solution_preserves_clues(board, solution):
        return None

    return solution


def print_board(board):

    for row in board:
        print(row)