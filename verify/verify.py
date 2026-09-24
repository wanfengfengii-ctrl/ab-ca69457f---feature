"""一次性验收服务：对 API 与 Web 做端到端核验，以退出码报告结果。

核验要点：
  * 健康检查与 Web 页面可达；
  * 无解 / 唯一 / 多解 三种判定正确（4×4 用暴力枚举独立复核）；
  * 返回解按行优先位串（0<1）升序，且确为最小的一或两个；
  * 已知单元被遵守，重复/冲突/越界等非法输入返回可定位的 422；
  * 较大实例（8×8 / 12×12）返回的见证满足全部四向投影。
"""

from __future__ import annotations

import itertools
import os
import random
import sys
import time

import requests

API_URL = os.environ.get("API_URL", "http://api:8000")
WEB_URL = os.environ.get("WEB_URL", "http://web")
WAIT_SECONDS = float(os.environ.get("VERIFY_WAIT_SECONDS", "90"))

FAILURES: list[str] = []
PASSES = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSES
    if condition:
        PASSES += 1
        print(f"  PASS {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL {name} {detail}")


def wait_ready() -> None:
    deadline = time.time() + WAIT_SECONDS
    for url in (f"{API_URL}/api/health", WEB_URL):
        while True:
            try:
                resp = requests.get(url, timeout=3)
                if resp.status_code == 200:
                    print(f"ready: {url}")
                    break
            except requests.RequestException:
                pass
            if time.time() > deadline:
                print(f"FATAL: 等待 {url} 就绪超时")
                sys.exit(1)
            time.sleep(1.5)


def reconstruct(payload: dict) -> requests.Response:
    return requests.post(f"{API_URL}/api/reconstruct", json=payload, timeout=60)


def projections_of(grid: list[list[int]]) -> tuple:
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
    """独立暴力枚举全部解（仅供 4×4 使用），按位串升序返回。"""
    known_map = {(r, c): v for r, c, v in known}
    sols = []
    for bits in itertools.product((0, 1), repeat=rows * cols):
        if any(bits[r * cols + c] != v for (r, c), v in known_map.items()):
            continue
        grid = [list(bits[r * cols : (r + 1) * cols]) for r in range(rows)]
        if projections_of(grid) != (row_sums, col_sums, diag, anti):
            continue
        if connected and not is_4connected(grid):
            continue
        sols.append(list(bits))
    return sols


def bits_of(grid) -> str:
    return "".join(str(v) for row in grid for v in row)


def satisfies(grid, row_sums, col_sums, diag, anti, known) -> bool:
    if projections_of(grid) != (row_sums, col_sums, diag, anti):
        return False
    return all(grid[r][c] == v for r, c, v in known)


def is_4connected(grid) -> bool:
    """全部 1 单元仅经上下左右相邻互达（对角不连接）；无夹杂视为成立。"""
    rows, cols = len(grid), len(grid[0])
    ones = {(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1}
    if not ones:
        return True
    seen = {next(iter(ones))}
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (r + dr, c + dc)
            if q in ones and q not in seen:
                seen.add(q)
                stack.append(q)
    return seen == ones


MULTI_4x4 = dict(
    rows=4,
    cols=4,
    row_sums=[1, 1, 1, 1],
    col_sums=[1, 1, 1, 1],
    diag_sums=[0, 1, 1, 0, 1, 1, 0],
    antidiag_sums=[0, 1, 1, 0, 1, 1, 0],
    known_cells=[],
)

UNIQUE_4x4 = dict(
    rows=4,
    cols=4,
    row_sums=[0, 0, 3, 4],
    col_sums=[1, 2, 2, 2],
    diag_sums=[0, 0, 1, 2, 2, 1, 1],
    antidiag_sums=[0, 0, 0, 2, 2, 2, 1],
    known_cells=[],
)

MULTI_8x8 = dict(
    rows=8,
    cols=8,
    row_sums=[2, 5, 4, 6, 5, 4, 5, 1],
    col_sums=[4, 4, 5, 4, 4, 5, 4, 2],
    diag_sums=[1, 0, 1, 2, 4, 2, 3, 5, 3, 3, 3, 2, 2, 1, 0],
    antidiag_sums=[0, 0, 1, 2, 4, 5, 5, 5, 2, 3, 1, 2, 2, 0, 0],
    known_cells=[],
)


def test_health_and_web() -> None:
    print("[1] 健康检查与 Web 页面")
    resp = requests.get(f"{API_URL}/api/health", timeout=5)
    check("API /api/health 返回 ok", resp.status_code == 200 and resp.json().get("status") == "ok")
    resp = requests.get(WEB_URL, timeout=5)
    check(
        "Web 首页可访问且挂载 root 节点",
        resp.status_code == 200 and 'id="root"' in resp.text,
    )
    resp = requests.get(f"{WEB_URL}/api/health", timeout=5)
    check(
        "Web 反向代理 /api 可达",
        resp.status_code == 200 and resp.json().get("status") == "ok",
    )


def test_multiple_4x4() -> None:
    print("[2] 4×4 多解：与暴力枚举独立复核")
    resp = reconstruct(MULTI_4x4)
    check("请求成功", resp.status_code == 200, f"HTTP {resp.status_code}")
    body = resp.json()
    expected = brute_force(
        4, 4,
        MULTI_4x4["row_sums"], MULTI_4x4["col_sums"],
        MULTI_4x4["diag_sums"], MULTI_4x4["antidiag_sums"],
    )
    check("暴力枚举确认存在 ≥2 个解", len(expected) >= 2, f"实际 {len(expected)}")
    check("状态为 multiple", body.get("status") == "multiple", str(body.get("status")))
    got = [bits_of(s["grid"]) for s in body.get("solutions", [])]
    want = ["".join(map(str, b)) for b in expected[:2]]
    check("返回恰为最小的两个解", got == want, f"got={got} want={want}")
    check("两解按 0<1 字典序升序", len(got) == 2 and got[0] < got[1])
    check(
        "两解均满足四向投影",
        all(
            satisfies(s["grid"], MULTI_4x4["row_sums"], MULTI_4x4["col_sums"],
                      MULTI_4x4["diag_sums"], MULTI_4x4["antidiag_sums"], [])
            for s in body["solutions"]
        ),
    )
    diff = [i for i in range(16) if got[0][i] != got[1][i]]
    check("两见证存在差异单元（供前端高亮）", len(diff) > 0, f"diff={diff}")


def test_unique_4x4() -> None:
    print("[3] 4×4 唯一解：与暴力枚举独立复核")
    resp = reconstruct(UNIQUE_4x4)
    body = resp.json()
    expected = brute_force(
        4, 4,
        UNIQUE_4x4["row_sums"], UNIQUE_4x4["col_sums"],
        UNIQUE_4x4["diag_sums"], UNIQUE_4x4["antidiag_sums"],
    )
    check("暴力枚举确认唯一", len(expected) == 1, f"实际 {len(expected)}")
    check("状态为 unique", body.get("status") == "unique")
    got = [bits_of(s["grid"]) for s in body.get("solutions", [])]
    want = ["".join(map(str, b)) for b in expected]
    check("返回的正是该唯一解", got == want, f"got={got} want={want}")


def test_known_cells() -> None:
    print("[4] 已知单元约束")
    base = {k: v for k, v in MULTI_4x4.items() if k != "known_cells"}

    resp = reconstruct({**base, "known_cells": [{"row": 0, "col": 1, "value": 1}]})
    body = resp.json()
    check(
        "钉住 (0,1)=1 后唯一且为第二小解",
        body.get("status") == "unique"
        and bits_of(body["solutions"][0]["grid"]) == "0100000110000010",
        str(body.get("status")),
    )

    resp = reconstruct({**base, "known_cells": [{"row": 0, "col": 1, "value": 0}]})
    body = resp.json()
    check(
        "钉住 (0,1)=0 后唯一且为最小解",
        body.get("status") == "unique"
        and bits_of(body["solutions"][0]["grid"]) == "0010100000010100",
        str(body.get("status")),
    )

    resp = reconstruct({**base, "known_cells": [{"row": 0, "col": 0, "value": 1}]})
    check("钉住 (0,0)=1 后无解", resp.json().get("status") == "no_solution")


def test_no_solution() -> None:
    print("[5] 无解判定")
    payload = dict(
        rows=4,
        cols=4,
        row_sums=[1, 0, 0, 0],
        col_sums=[0, 0, 0, 0],
        diag_sums=[0] * 7,
        antidiag_sums=[0] * 7,
        known_cells=[],
    )
    body = reconstruct(payload).json()
    check("行列总和矛盾 -> no_solution", body.get("status") == "no_solution")
    check("无解时 solutions 为空", body.get("solutions") == [])

    payload2 = dict(
        rows=4,
        cols=4,
        row_sums=[1, 1, 1, 1],
        col_sums=[1, 1, 1, 1],
        diag_sums=[0, 1, 1, 0, 1, 1, 0],
        antidiag_sums=[0, 1, 1, 0, 1, 0, 1],
        known_cells=[],
    )
    body = reconstruct(payload2).json()
    check("结构矛盾（总和一致）-> no_solution", body.get("status") == "no_solution")


def test_validation() -> None:
    print("[6] 非法输入的可定位 422")
    valid = dict(MULTI_4x4)

    cases = [
        ("行数越界", {**valid, "rows": 3}, "rows"),
        ("行数超上限", {**valid, "rows": 13}, "rows"),
        ("行投影长度错误", {**valid, "row_sums": [1, 1, 1]}, "row_sums"),
        ("行投影超过列数", {**valid, "row_sums": [5, 1, 1, 1]}, "row_sums[0]"),
        ("列投影为负", {**valid, "col_sums": [-1, 1, 1, 1]}, "col_sums[0]"),
        ("对角投影长度错误", {**valid, "diag_sums": [0] * 6}, "diag_sums"),
        (
            "对角投影超过线长",
            {**valid, "diag_sums": [2, 1, 1, 0, 1, 1, 0]},
            "diag_sums[0]",
        ),
        (
            "副对角投影超过线长",
            {**valid, "antidiag_sums": [0, 1, 1, 0, 1, 1, 5]},
            "antidiag_sums[6]",
        ),
        (
            "已知单元行越界",
            {**valid, "known_cells": [{"row": 4, "col": 0, "value": 1}]},
            "known_cells[0].row",
        ),
        (
            "已知单元列越界",
            {**valid, "known_cells": [{"row": 0, "col": 9, "value": 1}]},
            "known_cells[0].col",
        ),
        (
            "已知单元取值非法",
            {**valid, "known_cells": [{"row": 0, "col": 0, "value": 2}]},
            "known_cells.0.value",
        ),
        (
            "已知单元重复",
            {
                **valid,
                "known_cells": [
                    {"row": 1, "col": 1, "value": 1},
                    {"row": 1, "col": 1, "value": 1},
                ],
            },
            "known_cells[1]",
        ),
        (
            "已知单元冲突",
            {
                **valid,
                "known_cells": [
                    {"row": 1, "col": 1, "value": 1},
                    {"row": 1, "col": 1, "value": 0},
                ],
            },
            "known_cells[1]",
        ),
        ("字段类型错误", {**valid, "row_sums": ["a", 1, 1, 1]}, "row_sums.0"),
    ]
    for name, payload, expect_loc in cases:
        resp = reconstruct(payload)
        if resp.status_code != 422:
            check(name, False, f"HTTP {resp.status_code}，期望 422")
            continue
        detail = resp.json().get("detail", [])
        locs = [e.get("loc", "") for e in detail]
        check(
            name,
            any(expect_loc in loc for loc in locs),
            f"期望 loc 含 {expect_loc!r}，实际 {locs}",
        )


def test_larger_instances() -> None:
    print("[7] 8×8 / 12×12 见证满足投影且有序")
    resp = reconstruct(MULTI_8x8)
    body = resp.json()
    check("8×8 状态为 multiple", body.get("status") == "multiple", str(body.get("status")))
    sols = body.get("solutions", [])
    check(
        "8×8 两见证均满足四向投影",
        len(sols) == 2
        and all(
            satisfies(s["grid"], MULTI_8x8["row_sums"], MULTI_8x8["col_sums"],
                      MULTI_8x8["diag_sums"], MULTI_8x8["antidiag_sums"], [])
            for s in sols
        ),
    )
    if len(sols) == 2:
        check(
            "8×8 见证按 0<1 字典序升序且不同",
            sols[0]["bits"] < sols[1]["bits"],
        )

    rng = random.Random(20260920)
    rows = cols = 12
    grid = [[1 if rng.random() < 0.5 else 0 for _ in range(cols)] for _ in range(rows)]
    rs, cs, dg, an = projections_of(grid)
    known = [(2, 3, grid[2][3]), (7, 9, grid[7][9]), (11, 0, grid[11][0])]
    payload = dict(
        rows=rows, cols=cols,
        row_sums=rs, col_sums=cs, diag_sums=dg, antidiag_sums=an,
        known_cells=[{"row": r, "col": c, "value": v} for r, c, v in known],
    )
    resp = reconstruct(payload)
    body = reconstruct(payload).json()
    check("12×12 请求成功", resp.status_code == 200, f"HTTP {resp.status_code}")
    sols = body.get("solutions", [])
    check(
        "12×12 见证满足投影与已知单元",
        len(sols) >= 1
        and all(satisfies(s["grid"], rs, cs, dg, an, known) for s in sols),
    )
    if body.get("status") == "multiple":
        check(
            "12×12 见证按 0<1 字典序升序且不同",
            len(sols) == 2 and sols[0]["bits"] < sols[1]["bits"],
        )
    again = reconstruct(payload).json()
    check("12×12 结果确定（两次调用一致）", again.get("solutions") == sols)


def test_connected_inclusion() -> None:
    print("[8] 连续夹杂体（四邻连通）约束")

    # ---- 兼容场景：省略字段与显式 false 等价，且响应回显 connected ----
    base = {k: v for k, v in MULTI_4x4.items()}
    omitted = reconstruct(base).json()
    explicit_false = reconstruct({**base, "connected": False}).json()
    check(
        "省略 connected 时响应回显 false",
        omitted.get("connected") is False and explicit_false.get("connected") is False,
    )
    check(
        "省略 connected 与显式 false 裁决完全一致",
        omitted.get("status") == explicit_false.get("status")
        and omitted.get("solutions") == explicit_false.get("solutions"),
    )

    # ---- 连通场景：单夹杂唯一解 ----
    single = dict(
        rows=4,
        cols=4,
        row_sums=[1, 0, 0, 0],
        col_sums=[0, 0, 0, 1],
        diag_sums=[1, 0, 0, 0, 0, 0, 0],
        antidiag_sums=[0, 0, 0, 1, 0, 0, 0],
        known_cells=[],
        connected=True,
    )
    body = reconstruct(single).json()
    check(
        "单夹杂启用约束 -> unique 且唯一 1 在 (0,3)",
        body.get("status") == "unique"
        and bits_of(body["solutions"][0]["grid"]) == "0001000000000000",
        str(body.get("status")),
    )
    check("响应回显 connected=true", body.get("connected") is True)
    check(
        "单夹杂见证四邻连通",
        all(is_4connected(s["grid"]) for s in body.get("solutions", [])),
    )

    # ---- 连通场景：暴力枚举确认恰有两个连通解，裁决与排序均正确 ----
    conn_multi = dict(
        rows=4,
        cols=4,
        row_sums=[3, 3, 3, 2],
        col_sums=[2, 3, 3, 3],
        diag_sums=[1, 1, 2, 4, 2, 1, 0],
        antidiag_sums=[1, 1, 2, 3, 2, 1, 1],
        known_cells=[],
        connected=True,
    )
    body = reconstruct(conn_multi).json()
    expected = brute_force(
        4, 4,
        conn_multi["row_sums"], conn_multi["col_sums"],
        conn_multi["diag_sums"], conn_multi["antidiag_sums"],
        connected=True,
    )
    check(
        "连通多解：暴力枚举确认 ≥2 个连通解",
        len(expected) >= 2 and all(is_4connected([b[r * 4 : (r + 1) * 4] for r in range(4)]) for b in expected),
        f"实际 {len(expected)}",
    )
    check("连通多解状态为 multiple", body.get("status") == "multiple", str(body.get("status")))
    got = [bits_of(s["grid"]) for s in body.get("solutions", [])]
    want = ["".join(map(str, b)) for b in expected[:2]]
    check("连通多解返回最小的两个连通见证", got == want, f"got={got} want={want}")
    check(
        "连通多解两见证均仅四邻连通（对角不算）",
        all(is_4connected(s["grid"]) for s in body.get("solutions", [])),
    )

    # ---- 断开场景：两个仅对角接触的夹杂，投影把位置钉死 ----
    diag_only = dict(
        rows=4,
        cols=4,
        row_sums=[1, 1, 0, 0],
        col_sums=[1, 1, 0, 0],
        diag_sums=[0, 0, 0, 2, 0, 0, 0],
        antidiag_sums=[1, 0, 1, 0, 0, 0, 0],
        known_cells=[],
    )
    body_off = reconstruct({**diag_only, "connected": False}).json()
    check(
        "不启用约束时对角断开网格可成立（unique，2 个团簇）",
        body_off.get("status") == "unique"
        and is_4connected(body_off["solutions"][0]["grid"]) is False,
        str(body_off.get("status")),
    )
    body_on = reconstruct({**diag_only, "connected": True}).json()
    check(
        "启用约束后对角断开 -> no_solution（非事后筛除）",
        body_on.get("status") == "no_solution",
        str(body_on.get("status")),
    )
    check("连通性无解时返回空网格列表", body_on.get("solutions") == [])

    # ---- 无夹杂场景：全 0 投影启用约束仍成立 ----
    empty = dict(
        rows=4,
        cols=4,
        row_sums=[0] * 4,
        col_sums=[0] * 4,
        diag_sums=[0] * 7,
        antidiag_sums=[0] * 7,
        known_cells=[],
        connected=True,
    )
    body = reconstruct(empty).json()
    check(
        "无夹杂网格启用约束仍 -> unique 且全 0",
        body.get("status") == "unique"
        and bits_of(body["solutions"][0]["grid"]) == "0" * 16,
        str(body.get("status")),
    )

    # ---- 已知单元与连通约束同模型：钉住两个对角点且无桥接 -> 无解 ----
    corners = dict(
        rows=4,
        cols=4,
        row_sums=[1, 0, 0, 1],
        col_sums=[1, 0, 0, 1],
        diag_sums=[0, 0, 0, 2, 0, 0, 0],
        antidiag_sums=[1, 0, 0, 0, 0, 0, 1],
        known_cells=[],
    )
    body_off = reconstruct({**corners, "connected": False}).json()
    check(
        "不启用约束时对角双角点网格可成立（unique）",
        body_off.get("status") == "unique"
        and bits_of(body_off["solutions"][0]["grid"]) == "1000000000000001",
        str(body_off.get("status")),
    )
    body_on = reconstruct(
        {**corners, "connected": True,
         "known_cells": [{"row": 0, "col": 0, "value": 1},
                         {"row": 3, "col": 3, "value": 1}]}
    ).json()
    check(
        "已知单元钉住 (0,0)/(3,3) 且启用连通 -> 无解、空网格",
        body_on.get("status") == "no_solution" and body_on.get("solutions") == [],
        str(body_on.get("status")),
    )

    # ---- 12×12 连通形：构造保证四邻连通的网格，见证须连通 ----
    rng = random.Random(20260924)
    rows = cols = 12
    g = [[0] * cols for _ in range(rows)]
    r, c = rng.randrange(rows), rng.randrange(cols)
    g[r][c] = 1
    placed = 1
    while placed < 46:
        nbrs = [
            (r + dr, c + dc)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if 0 <= r + dr < rows and 0 <= c + dc < cols
        ]
        nr, nc = rng.choice(nbrs)
        if g[nr][nc] == 0:
            g[nr][nc] = 1
            placed += 1
        r, c = nr, nc
    rs, cs, dg, an = projections_of(g)
    payload = dict(
        rows=rows, cols=cols,
        row_sums=rs, col_sums=cs, diag_sums=dg, antidiag_sums=an,
        known_cells=[], connected=True,
    )
    resp = reconstruct(payload)
    body = resp.json()
    check("12×12 连通实例请求成功", resp.status_code == 200, f"HTTP {resp.status_code}")
    sols = body.get("solutions", [])
    check(
        "12×12 连通实例有解且见证全部四邻连通",
        body.get("status") in ("unique", "multiple")
        and len(sols) >= 1
        and all(is_4connected(s["grid"]) for s in sols),
        str(body.get("status")),
    )
    check(
        "12×12 连通实例见证满足四向投影",
        all(satisfies(s["grid"], rs, cs, dg, an, []) for s in sols),
    )


def main() -> int:
    print(f"verify: API_URL={API_URL} WEB_URL={WEB_URL}")
    wait_ready()
    test_health_and_web()
    test_multiple_4x4()
    test_unique_4x4()
    test_known_cells()
    test_no_solution()
    test_validation()
    test_larger_instances()
    test_connected_inclusion()
    print(f"\n通过 {PASSES} 项，失败 {len(FAILURES)} 项")
    if FAILURES:
        for f in FAILURES:
            print(f"  FAILED: {f}")
        return 1
    print("验收全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
