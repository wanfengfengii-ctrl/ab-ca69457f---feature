"""求解器单元测试：4×4 全量暴力枚举对照 + 边界情形。

运行：cd api && python -m pytest tests/ -q
（需要 pip install ortools pytest，仅用于开发，不进入镜像。）
"""

from __future__ import annotations

import itertools
import random

import pytest

from app.solver import solve_reconstruction


def projections_of(grid):
    rows, cols = len(grid), len(grid[0])
    row_sums = [sum(row) for row in grid]
    col_sums = [sum(grid[r][c] for r in range(rows)) for c in range(cols)]
    diag = [0] * (rows + cols - 1)
    anti = [0] * (rows + cols - 1)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c]:
                diag[r - c + (cols - 1)] += 1
                anti[r + c] += 1
    return row_sums, col_sums, diag, anti


def is_connected(bits, rows, cols):
    """值为 1 的单元是否仅经上下左右相邻的 1 单元互达（对角不算）。"""
    ones = [i for i, b in enumerate(bits) if b]
    if len(ones) <= 1:
        return True
    seen = {ones[0]}
    stack = [ones[0]]
    while stack:
        i = stack.pop()
        r, c = divmod(i, cols)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                j = nr * cols + nc
                if bits[j] and j not in seen:
                    seen.add(j)
                    stack.append(j)
    return len(seen) == len(ones)


def brute_force(rows, cols, row_sums, col_sums, diag, anti, known=(), connected=False):
    known_map = {(r, c): v for r, c, v in known}
    sols = []
    for bits in itertools.product((0, 1), repeat=rows * cols):
        if any(bits[r * cols + c] != v for (r, c), v in known_map.items()):
            continue
        if connected and not is_connected(bits, rows, cols):
            continue
        grid = [list(bits[r * cols : (r + 1) * cols]) for r in range(rows)]
        if projections_of(grid) == (row_sums, col_sums, diag, anti):
            sols.append(list(bits))
    return sols


def check(rows, cols, grid, known=()):
    rs, cs, dg, an = projections_of(grid)
    status, sols = solve_reconstruction(rows, cols, rs, cs, dg, an, list(known))
    expected = brute_force(rows, cols, rs, cs, dg, an, known)
    exp_status = (
        "no_solution" if not expected else "unique" if len(expected) == 1 else "multiple"
    )
    assert status == exp_status
    assert sols == expected[:2]


def test_exhaustive_4x4_sample():
    """对 4×4 均匀抽样（含全 0/全 1）与暴力枚举逐一比对。"""
    rng = random.Random(2026)
    samples = [0, 1, 2, 65535, 65534, 32768] + [rng.randrange(65536) for _ in range(60)]
    for code in samples:
        grid = [[(code >> (r * 4 + c)) & 1 for c in range(4)] for r in range(4)]
        check(4, 4, grid)


def test_known_cells_4x4():
    rng = random.Random(7)
    for _ in range(20):
        grid = [[rng.randint(0, 1) for _ in range(4)] for _ in range(4)]
        known = set()
        for _ in range(rng.randint(0, 5)):
            r, c = rng.randrange(4), rng.randrange(4)
            known.add((r, c, grid[r][c]))
        check(4, 4, grid, sorted(known))


def test_known_cells_infeasible():
    grid = [[0] * 4 for _ in range(4)]
    rs, cs, dg, an = projections_of(grid)
    status, sols = solve_reconstruction(4, 4, rs, cs, dg, an, [(0, 0, 1)])
    assert status == "no_solution"
    assert sols == []


def test_lex_order_multiple():
    # 已知 4×4 歧义实例：两解必须按 0<1 字典序升序。
    status, sols = solve_reconstruction(
        4, 4, [1, 1, 1, 1], [1, 1, 1, 1],
        [0, 1, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 1, 0], [],
    )
    assert status == "multiple"
    assert len(sols) == 2
    assert sols[0] < sols[1]


@pytest.mark.parametrize("side", [5, 6, 8])
def test_rectangular_and_larger(side):
    rng = random.Random(side)
    rows, cols = side, side + 2
    grid = [[1 if rng.random() < 0.5 else 0 for _ in range(cols)] for _ in range(rows)]
    rs, cs, dg, an = projections_of(grid)
    status, sols = solve_reconstruction(rows, cols, rs, cs, dg, an, [])
    assert status in ("unique", "multiple")
    for sol in sols:
        rebuilt = [sol[r * cols : (r + 1) * cols] for r in range(rows)]
        assert projections_of(rebuilt) == (rs, cs, dg, an)
    if status == "multiple":
        assert sols[0] < sols[1]


# ------------------------------------------------------------------
# 连续夹杂体约束（require_connected）
# ------------------------------------------------------------------

# 4×5 混合实例：无约束时两个解，字典序最小解断开、次小解连通。
CONN_4x5 = dict(
    rows=4,
    cols=5,
    row_sums=[1, 4, 3, 1],
    col_sums=[1, 3, 3, 1, 1],
    diag_sums=[0, 1, 1, 2, 2, 2, 1, 0],
    antidiag_sums=[0, 1, 2, 2, 2, 2, 0, 0],
)
CONN_4x5_FIRST = [int(b) for b in "00100111010111001000"]  # 断开
CONN_4x5_SECOND = [int(b) for b in "01000011111110000100"]  # 连通

# 4×5 双连通实例：两个解都是连通团簇。
CONN2_4x5 = dict(
    rows=4,
    cols=5,
    row_sums=[1, 3, 3, 1],
    col_sums=[1, 3, 3, 1, 0],
    diag_sums=[0, 0, 1, 2, 2, 2, 1, 0],
    antidiag_sums=[0, 1, 2, 2, 2, 1, 0, 0],
)


def test_connected_constraint_4x4_sample():
    """启用约束后与"暴力枚举 + 连通性过滤"逐一比对（含已知单元）。"""
    rng = random.Random(2026)
    samples = [0, 1, 2, 65535, 65534, 32768, 4369, 13107] + [
        rng.randrange(65536) for _ in range(40)
    ]
    for code in samples:
        grid = [[(code >> (r * 4 + c)) & 1 for c in range(4)] for r in range(4)]
        rs, cs, dg, an = projections_of(grid)
        known = []
        if rng.random() < 0.5:
            r, c = rng.randrange(4), rng.randrange(4)
            known = [(r, c, grid[r][c])]
        status, sols = solve_reconstruction(
            4, 4, rs, cs, dg, an, known, require_connected=True
        )
        expected = brute_force(4, 4, rs, cs, dg, an, known, connected=True)
        exp_status = (
            "no_solution" if not expected else "unique" if len(expected) == 1 else "multiple"
        )
        assert status == exp_status, (code, status, exp_status)
        assert sols == expected[:2], code


def test_connected_witness_comes_from_same_solve():
    """约束后的见证必须来自同一次完备求解，而不是旧见证筛除。

    无约束时字典序最小的见证是断开的；启用约束后应直接在求解中
    跳过它，返回连通解（本例中即原次小解），判定由 multiple 变 unique。
    """
    p = CONN_4x5
    status, sols = solve_reconstruction(
        p["rows"], p["cols"], p["row_sums"], p["col_sums"],
        p["diag_sums"], p["antidiag_sums"], [],
    )
    assert status == "multiple"
    assert sols[0] == CONN_4x5_FIRST and not is_connected(sols[0], 4, 5)
    assert sols[1] == CONN_4x5_SECOND and is_connected(sols[1], 4, 5)

    status, sols = solve_reconstruction(
        p["rows"], p["cols"], p["row_sums"], p["col_sums"],
        p["diag_sums"], p["antidiag_sums"], [], require_connected=True,
    )
    assert status == "unique"
    assert sols == [CONN_4x5_SECOND]
    assert is_connected(sols[0], 4, 5)


def test_connected_all_disconnected_gives_no_solution():
    """全部投影解都断开时，启用约束返回无解与空解列表。"""
    args = ([1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 1, 0])
    status, sols = solve_reconstruction(4, 4, *args, [])
    assert status == "multiple"  # 无约束时有两个解（均为 4 个孤立点）
    status, sols = solve_reconstruction(4, 4, *args, [], require_connected=True)
    assert status == "no_solution"
    assert sols == []


def test_connected_diagonal_touch_does_not_count():
    """对角接触不得连接：(0,0) 与 (1,1) 的唯一解在约束下无解。"""
    args = ([1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 2, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0])
    status, _ = solve_reconstruction(4, 4, *args, [])
    assert status == "unique"
    status, sols = solve_reconstruction(4, 4, *args, [], require_connected=True)
    assert status == "no_solution"
    assert sols == []


def test_connected_empty_grid_still_valid():
    """没有夹杂物的网格仍可成立：全零投影在约束下唯一可解。"""
    status, sols = solve_reconstruction(
        4, 4, [0] * 4, [0] * 4, [0] * 7, [0] * 7, [], require_connected=True
    )
    assert status == "unique"
    assert sols == [[0] * 16]


def test_connected_two_connected_witnesses_ordered():
    """两个解都连通时，约束下仍为多解且按 0<1 字典序升序。"""
    p = CONN2_4x5
    status, sols = solve_reconstruction(
        p["rows"], p["cols"], p["row_sums"], p["col_sums"],
        p["diag_sums"], p["antidiag_sums"], [], require_connected=True,
    )
    assert status == "multiple"
    assert len(sols) == 2 and sols[0] < sols[1]
    assert all(is_connected(s, 4, 5) for s in sols)


def test_connected_with_known_cells():
    """已知单元与连通约束在同一模型内共同生效。"""
    p = CONN_4x5
    # 钉 (0,1)=0 排除唯一的连通解 -> 约束下无解
    status, sols = solve_reconstruction(
        p["rows"], p["cols"], p["row_sums"], p["col_sums"],
        p["diag_sums"], p["antidiag_sums"], [(0, 1, 0)], require_connected=True,
    )
    assert status == "no_solution"
    assert sols == []
    # 钉 (0,1)=1 与连通解一致 -> 唯一连通解
    status, sols = solve_reconstruction(
        p["rows"], p["cols"], p["row_sums"], p["col_sums"],
        p["diag_sums"], p["antidiag_sums"], [(0, 1, 1)], require_connected=True,
    )
    assert status == "unique"
    assert sols == [CONN_4x5_SECOND]


def test_connected_default_matches_omitted():
    """未启用时省略参数与显式 False 结果一致（向后兼容）。"""
    p = CONN_4x5
    args = (p["rows"], p["cols"], p["row_sums"], p["col_sums"],
            p["diag_sums"], p["antidiag_sums"], [])
    assert solve_reconstruction(*args) == solve_reconstruction(
        *args, require_connected=False
    )


def test_connected_larger_random_blob():
    """8×8 连通团簇实例：约束下见证必须连通且满足四向投影。"""
    rng = random.Random(20260925)
    rows = cols = 8
    cells = {(4, 4)}
    while len(cells) < 30:
        r, c = rng.choice(tuple(cells))
        dr, dc = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            cells.add((nr, nc))
    grid = [[1 if (r, c) in cells else 0 for c in range(cols)] for r in range(rows)]
    rs, cs, dg, an = projections_of(grid)
    status, sols = solve_reconstruction(
        rows, cols, rs, cs, dg, an, [], require_connected=True
    )
    assert status in ("unique", "multiple")
    assert len(sols) >= 1
    for sol in sols:
        assert is_connected(sol, rows, cols)
        rebuilt = [sol[r * cols : (r + 1) * cols] for r in range(rows)]
        assert projections_of(rebuilt) == (rs, cs, dg, an)
    if status == "multiple":
        assert sols[0] < sols[1]
