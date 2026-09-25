"""一次性验收服务：对 API 与 Web 做端到端核验，以退出码报告结果。

核验要点：
  * 健康检查与 Web 页面可达；
  * 无解 / 唯一 / 多解 三种判定正确（4×4 用暴力枚举独立复核）；
  * 返回解按行优先位串（0<1）升序，且确为最小的一或两个；
  * 已知单元被遵守，重复/冲突/越界等非法输入返回可定位的 422；
  * 较大实例（8×8 / 12×12）返回的见证满足全部四向投影；
  * 连续夹杂体约束：连通见证在同一次求解中产生（非旧见证筛除）、
    全断开实例判无解、对角接触不算连通、无夹杂网格仍成立、
    省略 require_connected 字段时接口与裁决保持兼容。
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


def is_connected(bits, rows, cols) -> bool:
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
    """独立暴力枚举全部解（逐行枚举，供 4×4 / 4×5 使用），按位串升序返回。"""
    known_map = {(r, c): v for r, c, v in known}
    sols = []

    def rec(r, grid_rows):
        if r == rows:
            bits = [b for row in grid_rows for b in row]
            if connected and not is_connected(bits, rows, cols):
                return
            grid = [list(row) for row in grid_rows]
            if projections_of(grid) == (row_sums, col_sums, diag, anti):
                sols.append(bits)
            return
        for row_bits in itertools.product((0, 1), repeat=cols):
            if sum(row_bits) != row_sums[r]:
                continue
            if any(
                row_bits[c] != known_map[(r, c)]
                for c in range(cols)
                if (r, c) in known_map
            ):
                continue
            rec(r + 1, grid_rows + [row_bits])

    rec(0, [])
    return sols


def bits_of(grid) -> str:
    return "".join(str(v) for row in grid for v in row)


def satisfies(grid, row_sums, col_sums, diag, anti, known) -> bool:
    if projections_of(grid) != (row_sums, col_sums, diag, anti):
        return False
    return all(grid[r][c] == v for r, c, v in known)


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

# 4×5 混合实例：无约束时两个解，字典序最小解断开、次小解连通。
CONN_4x5 = dict(
    rows=4,
    cols=5,
    row_sums=[1, 4, 3, 1],
    col_sums=[1, 3, 3, 1, 1],
    diag_sums=[0, 1, 1, 2, 2, 2, 1, 0],
    antidiag_sums=[0, 1, 2, 2, 2, 2, 0, 0],
    known_cells=[],
)

# 4×5 双连通实例：两个解都是四向连通团簇。
CONN2_4x5 = dict(
    rows=4,
    cols=5,
    row_sums=[1, 3, 3, 1],
    col_sums=[1, 3, 3, 1, 0],
    diag_sums=[0, 0, 1, 2, 2, 2, 1, 0],
    antidiag_sums=[0, 1, 2, 2, 2, 1, 0, 0],
    known_cells=[],
)

# 4×4 对角接触实例：(0,0) 与 (1,1) 仅对角相邻，唯一解但不连通。
DIAG_4x4 = dict(
    rows=4,
    cols=4,
    row_sums=[1, 1, 0, 0],
    col_sums=[1, 1, 0, 0],
    diag_sums=[0, 0, 0, 2, 0, 0, 0],
    antidiag_sums=[1, 0, 1, 0, 0, 0, 0],
    known_cells=[],
)

# 4×4 无夹杂实例：全零投影。
EMPTY_4x4 = dict(
    rows=4,
    cols=4,
    row_sums=[0, 0, 0, 0],
    col_sums=[0, 0, 0, 0],
    diag_sums=[0] * 7,
    antidiag_sums=[0] * 7,
    known_cells=[],
)


def connected_blob_8x8():
    """确定性地生成一个 8×8 四向连通团簇实例（随机游走 30 格）。"""
    rng = random.Random(20260925)
    rows = cols = 8
    cells = {(4, 4)}
    while len(cells) < 30:
        r, c = rng.choice(sorted(cells))
        dr, dc = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            cells.add((nr, nc))
    grid = [[1 if (r, c) in cells else 0 for c in range(cols)] for r in range(rows)]
    rs, cs, dg, an = projections_of(grid)
    return rs, cs, dg, an


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


def test_connectivity_constraint() -> None:
    print("[8] 连续夹杂体约束（连通 / 断开 / 无夹杂 / 兼容）")
    base = {k: v for k, v in CONN_4x5.items() if k != "known_cells"}

    # 独立地面真值：逐行暴力枚举 + 连通性过滤
    all_sols = brute_force(
        4, 5, CONN_4x5["row_sums"], CONN_4x5["col_sums"],
        CONN_4x5["diag_sums"], CONN_4x5["antidiag_sums"],
    )
    conn_sols = brute_force(
        4, 5, CONN_4x5["row_sums"], CONN_4x5["col_sums"],
        CONN_4x5["diag_sums"], CONN_4x5["antidiag_sums"], connected=True,
    )
    check(
        "4×5 实例无约束有 2 解且最小解断开",
        len(all_sols) == 2 and not is_connected(all_sols[0], 4, 5),
    )
    check("4×5 实例恰有 1 个连通解", len(conn_sols) == 1, f"实际 {len(conn_sols)}")

    # 连通：约束见证必须来自同一次完备求解（判定 multiple -> unique）
    resp = reconstruct({**base, "require_connected": True})
    body = resp.json()
    check("启用约束后请求成功", resp.status_code == 200, f"HTTP {resp.status_code}")
    check(
        "启用约束后判定由多解变为唯一",
        body.get("status") == "unique",
        str(body.get("status")),
    )
    got = [bits_of(s["grid"]) for s in body.get("solutions", [])]
    want = ["".join(map(str, b)) for b in conn_sols[:2]]
    check("约束见证恰为暴力枚举的连通解", got == want, f"got={got} want={want}")
    check(
        "约束见证四向连通",
        len(got) >= 1 and all(is_connected([int(b) for b in g], 4, 5) for g in got),
    )
    check(
        "约束见证不同于无约束最小见证（非旧见证筛除）",
        len(got) >= 1 and got[0] != "".join(map(str, all_sols[0])),
    )

    # 断开：全部投影解都断开 -> 无解 + 空网格
    multi_base = {k: v for k, v in MULTI_4x4.items() if k != "known_cells"}
    body = reconstruct({**multi_base, "require_connected": True}).json()
    check("全部解断开时启用约束 -> no_solution", body.get("status") == "no_solution")
    check("约束无解时 solutions 为空", body.get("solutions") == [])

    # 对角接触不得连接
    diag_base = {k: v for k, v in DIAG_4x4.items() if k != "known_cells"}
    body = reconstruct({**diag_base, "require_connected": True}).json()
    check(
        "对角接触不算连通 -> no_solution",
        body.get("status") == "no_solution" and body.get("solutions") == [],
    )
    body = reconstruct(DIAG_4x4).json()
    check("同一实例不启用约束 -> unique", body.get("status") == "unique")

    # 无夹杂：全零网格仍可成立
    empty_base = {k: v for k, v in EMPTY_4x4.items() if k != "known_cells"}
    body = reconstruct({**empty_base, "require_connected": True}).json()
    check("无夹杂网格在约束下仍唯一可解", body.get("status") == "unique")
    check(
        "无夹杂解为全零网格",
        bool(body.get("solutions"))
        and all(v == 0 for row in body["solutions"][0]["grid"] for v in row),
    )

    # 两个连通见证：仍为多解且按 0<1 字典序升序
    conn2_base = {k: v for k, v in CONN2_4x5.items() if k != "known_cells"}
    body = reconstruct({**conn2_base, "require_connected": True}).json()
    sols = body.get("solutions", [])
    check(
        "双连通实例约束下仍为多解",
        body.get("status") == "multiple" and len(sols) == 2,
        str(body.get("status")),
    )
    check(
        "双连通见证均连通且按 0<1 升序",
        len(sols) == 2
        and sols[0]["bits"] < sols[1]["bits"]
        and all(is_connected([int(b) for b in s["bits"]], 4, 5) for s in sols),
    )

    # 已知单元与连通约束在同一模型内共同生效
    body = reconstruct(
        {**base, "known_cells": [{"row": 0, "col": 1, "value": 0}], "require_connected": True}
    ).json()
    check("已知单元排除连通解后约束下无解", body.get("status") == "no_solution")
    body = reconstruct(
        {**base, "known_cells": [{"row": 0, "col": 1, "value": 1}], "require_connected": True}
    ).json()
    check(
        "已知单元与连通解一致时唯一",
        body.get("status") == "unique"
        and bits_of(body["solutions"][0]["grid"]) == want[0],
    )

    # 兼容：省略字段 == 显式 false == 旧的暴力枚举裁决
    body_omit = reconstruct(CONN_4x5).json()
    body_false = reconstruct({**base, "require_connected": False}).json()
    check(
        "省略字段与显式 false 结果一致",
        body_omit.get("status") == body_false.get("status")
        and body_omit.get("solutions") == body_false.get("solutions"),
    )
    got = [bits_of(s["grid"]) for s in body_omit.get("solutions", [])]
    want_all = ["".join(map(str, b)) for b in all_sols[:2]]
    check(
        "省略字段时维持旧裁决（多解含断开见证）",
        body_omit.get("status") == "multiple" and got == want_all,
        f"got={got}",
    )

    # 8×8 连通团簇实例：约束下见证连通、满足投影、结果确定
    rs, cs, dg, an = connected_blob_8x8()
    payload = dict(
        rows=8, cols=8, row_sums=rs, col_sums=cs,
        diag_sums=dg, antidiag_sums=an, require_connected=True,
    )
    resp = reconstruct(payload)
    body = resp.json()
    sols = body.get("solutions", [])
    check(
        "8×8 连通约束请求成功且有解",
        resp.status_code == 200
        and body.get("status") in ("unique", "multiple")
        and len(sols) >= 1,
        f"HTTP {resp.status_code} status={body.get('status')}",
    )
    check(
        "8×8 约束见证均连通且满足四向投影",
        len(sols) >= 1
        and all(
            is_connected([int(b) for b in s["bits"]], 8, 8)
            and satisfies(s["grid"], rs, cs, dg, an, [])
            for s in sols
        ),
    )
    again = reconstruct(payload).json()
    check("8×8 约束结果确定（两次调用一致）", again.get("solutions") == sols)


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
    test_connectivity_constraint()
    print(f"\n通过 {PASSES} 项，失败 {len(FAILURES)} 项")
    if FAILURES:
        for f in FAILURES:
            print(f"  FAILED: {f}")
        return 1
    print("验收全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
