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


def brute_force(rows, cols, row_sums, col_sums, diag, anti, known=()):
    known_map = {(r, c): v for r, c, v in known}
    sols = []
    for bits in itertools.product((0, 1), repeat=rows * cols):
        if any(bits[r * cols + c] != v for (r, c), v in known_map.items()):
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
