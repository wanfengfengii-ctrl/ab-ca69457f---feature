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


def brute_force(rows, cols, row_sums, col_sums, diag, anti, known=(), connected=False):
    known_map = {(r, c): v for r, c, v in known}
    sols = []
    for bits in itertools.product((0, 1), repeat=rows * cols):
        if any(bits[r * cols + c] != v for (r, c), v in known_map.items()):
            continue
        grid = [list(bits[r * cols : (r + 1) * cols]) for r in range(rows)]
        if projections_of(grid) != (row_sums, col_sums, diag, anti):
            continue
        if connected and not _is_4connected(bits, rows, cols):
            continue
        sols.append(list(bits))
    return sols


def _is_4connected(bits, rows, cols):
    """全部 1 单元仅经上下左右相邻互达（对角不连接）；无 1 单元视为成立。"""
    ones = {i for i, b in enumerate(bits) if b}
    if not ones:
        return True
    seen = {next(iter(ones))}
    stack = list(seen)
    while stack:
        i = stack.pop()
        r, c = divmod(i, cols)
        for j in (
            i - cols if r > 0 else None,
            i + cols if r + 1 < rows else None,
            i - 1 if c > 0 else None,
            i + 1 if c + 1 < cols else None,
        ):
            if j is not None and j in ones and j not in seen:
                seen.add(j)
                stack.append(j)
    return seen == ones


def check(rows, cols, grid, known=(), connected=False):
    rs, cs, dg, an = projections_of(grid)
    status, sols = solve_reconstruction(
        rows, cols, rs, cs, dg, an, list(known), require_connected=connected
    )
    expected = brute_force(rows, cols, rs, cs, dg, an, known, connected=connected)
    exp_status = (
        "no_solution" if not expected else "unique" if len(expected) == 1 else "multiple"
    )
    assert status == exp_status
    assert sols == expected[:2]
    for bits in sols:
        if connected:
            assert _is_4connected(bits, rows, cols)


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


# ---------------------------------------------------------------------
# 连续夹杂体（四邻连通）约束
# ---------------------------------------------------------------------
def test_connected_exhaustive_4x4_sample():
    """启用连通约束后，4×4 抽样实例与暴力枚举逐一比对。"""
    rng = random.Random(2027)
    samples = [0, 1, 2, 65535, 65534, 32768] + [rng.randrange(65536) for _ in range(30)]
    for code in samples:
        grid = [[(code >> (r * 4 + c)) & 1 for c in range(4)] for r in range(4)]
        check(4, 4, grid, connected=True)


def test_connected_rectangular_4x5():
    rng = random.Random(31)
    for _ in range(6):
        grid = [[rng.randint(0, 1) for _ in range(5)] for _ in range(4)]
        check(4, 5, grid, connected=True)


def test_connected_diagonal_touch_is_disconnected():
    """两个仅对角接触的 1 单元：启用约束后无解（对角不得连接）。"""
    grid = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    rs, cs, dg, an = projections_of(grid)
    # 不加约束时该网格本身是唯一解。
    status, sols = solve_reconstruction(4, 4, rs, cs, dg, an, [])
    assert status == "unique" and sols[0] == [b for row in grid for b in row]
    # 启用约束后，若投影允许其它布局则只返回连通解；本例投影把两个 1
    # 钉死在对角线上（各线和均为 1 且交叉唯一），故无解。
    status, sols = solve_reconstruction(4, 4, rs, cs, dg, an, [], require_connected=True)
    assert status == "no_solution"
    assert sols == []


def test_connected_empty_grid_still_valid():
    """无夹杂网格在启用约束后仍成立。"""
    status, sols = solve_reconstruction(
        4, 4, [0] * 4, [0] * 4, [0] * 7, [0] * 7, [], require_connected=True
    )
    assert status == "unique"
    assert sols == [[0] * 16]


def test_connected_single_inclusion():
    """单个夹杂自然连通。"""
    grid = [[0] * 4 for _ in range(4)]
    grid[0][3] = 1
    check(4, 4, grid, connected=True)


def test_connected_with_known_cells():
    """连通约束与已知单元在同一模型中求解。"""
    rng = random.Random(99)
    for _ in range(8):
        grid = [[rng.randint(0, 1) for _ in range(4)] for _ in range(4)]
        known = set()
        for _ in range(rng.randint(0, 4)):
            r, c = rng.randrange(4), rng.randrange(4)
            known.add((r, c, grid[r][c]))
        check(4, 4, grid, sorted(known), connected=True)

    # 已知单元钉住两个对角点（投影允许）时，若不存在四邻桥接则无解。
    grid = [
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
    ]
    rs, cs, dg, an = projections_of(grid)
    status, _ = solve_reconstruction(
        4, 4, rs, cs, dg, an, [(0, 0, 1), (3, 3, 1)], require_connected=True
    )
    assert status == "no_solution"


def test_connected_lex_order_preserved():
    """启用约束后的多解仍按 0<1 字典序升序，且两解都连通。"""
    # 暴力枚举确认：该 4×4 实例恰好有 2 个可行网格且两者均连通。
    rs = [3, 3, 3, 2]
    cs = [2, 3, 3, 3]
    dg = [1, 1, 2, 4, 2, 1, 0]
    an = [1, 1, 2, 3, 2, 1, 1]
    status, sols = solve_reconstruction(
        4, 4, rs, cs, dg, an, [], require_connected=True
    )
    assert status == "multiple"
    assert len(sols) == 2
    assert sols[0] < sols[1]
    assert all(_is_4connected(s, 4, 4) for s in sols)
    # 与暴力枚举独立复核，确为连通可行域内最小的两个。
    expected = brute_force(4, 4, rs, cs, dg, an, connected=True)
    assert sols == expected[:2]


def test_connected_default_off_is_compatible():
    """省略连通参数时行为与旧接口完全一致。"""
    status_a, sols_a = solve_reconstruction(
        4, 4, [1, 1, 1, 1], [1, 1, 1, 1],
        [0, 1, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 1, 0], [],
    )
    status_b, sols_b = solve_reconstruction(
        4, 4, [1, 1, 1, 1], [1, 1, 1, 1],
        [0, 1, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 1, 0], [],
        require_connected=False,
    )
    assert status_a == status_b == "multiple"
    assert sols_a == sols_b
