"""请求/响应模型与可定位输入校验。

校验错误统一形如 {"loc": "row_sums[2]", "msg": "..."}，
loc 指向出错的投影线或已知单元，便于前端定位反馈。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

MIN_SIDE = 4
MAX_SIDE = 12


class InputError(Exception):
    """携带可定位错误列表的输入非法异常。"""

    def __init__(self, errors: list[dict]) -> None:
        super().__init__(str(errors))
        self.errors = errors


class KnownCell(BaseModel):
    row: int = Field(ge=0, description="行号，从 0 开始")
    col: int = Field(ge=0, description="列号，从 0 开始")
    value: Literal[0, 1]


class ReconstructRequest(BaseModel):
    rows: int = Field(ge=MIN_SIDE, le=MAX_SIDE, description="行数 4~12")
    cols: int = Field(ge=MIN_SIDE, le=MAX_SIDE, description="列数 4~12")
    row_sums: list[int] = Field(description="行投影，自上而下，长度=rows")
    col_sums: list[int] = Field(description="列投影，自左向右，长度=cols")
    diag_sums: list[int] = Field(
        description="对角投影，按(行-列)差递增，长度=rows+cols-1"
    )
    antidiag_sums: list[int] = Field(
        description="副对角投影，按(行+列)和递增，长度=rows+cols-1"
    )
    known_cells: list[KnownCell] = Field(default_factory=list)
    require_connected: bool = Field(
        default=False,
        description=(
            "连续夹杂体约束：为 1 的单元必须仅经上下左右相邻的 1 单元互达"
            "（对角接触不算连通，全 0 网格仍合法）。可省略，默认不启用。"
        ),
    )


class SolutionOut(BaseModel):
    grid: list[list[int]] = Field(description="rows×cols 的 0/1 网格")
    bits: str = Field(description="行优先位串，如 '0101...'")


class ReconstructResponse(BaseModel):
    status: Literal["no_solution", "unique", "multiple"]
    rows: int
    cols: int
    solutions: list[SolutionOut]
    elapsed_ms: int


def _diag_length(rows: int, cols: int, d: int) -> int:
    """行列差为 d 的对角线长度。"""
    return sum(
        1 for r in range(rows) for c in range(cols) if r - c == d
    )


def _antidiag_length(rows: int, cols: int, s: int) -> int:
    """行列和为 s 的副对角线长度。"""
    return sum(
        1 for r in range(rows) for c in range(cols) if r + c == s
    )


def validate_request(req: ReconstructRequest) -> None:
    """对通过基本类型校验的请求做完整业务校验，失败抛 InputError。"""
    errors: list[dict] = []
    rows, cols = req.rows, req.cols

    def check_line(name: str, values: list[int], expected_len: int, cap) -> None:
        if len(values) != expected_len:
            errors.append(
                {
                    "loc": name,
                    "msg": f"长度应为 {expected_len}，实际为 {len(values)}",
                }
            )
            return
        for i, v in enumerate(values):
            limit = cap(i)
            if v < 0:
                errors.append(
                    {"loc": f"{name}[{i}]", "msg": f"值 {v} 不能为负数"}
                )
            elif v > limit:
                errors.append(
                    {
                        "loc": f"{name}[{i}]",
                        "msg": f"值 {v} 超过该线单元数上限 {limit}",
                    }
                )

    check_line("row_sums", req.row_sums, rows, lambda _i: cols)
    check_line("col_sums", req.col_sums, cols, lambda _i: rows)
    check_line(
        "diag_sums",
        req.diag_sums,
        rows + cols - 1,
        lambda i: _diag_length(rows, cols, i - (cols - 1)),
    )
    check_line(
        "antidiag_sums",
        req.antidiag_sums,
        rows + cols - 1,
        lambda i: _antidiag_length(rows, cols, i),
    )

    seen: dict[tuple[int, int], int] = {}
    for idx, cell in enumerate(req.known_cells):
        if cell.row >= rows:
            errors.append(
                {
                    "loc": f"known_cells[{idx}].row",
                    "msg": f"行号 {cell.row} 超出范围 0~{rows - 1}",
                }
            )
        if cell.col >= cols:
            errors.append(
                {
                    "loc": f"known_cells[{idx}].col",
                    "msg": f"列号 {cell.col} 超出范围 0~{cols - 1}",
                }
            )
        key = (cell.row, cell.col)
        if key in seen:
            if seen[key] == cell.value:
                msg = f"单元 ({cell.row}, {cell.col}) 的已知值重复出现"
            else:
                msg = (
                    f"单元 ({cell.row}, {cell.col}) 的已知值冲突："
                    f"{seen[key]} 与 {cell.value}"
                )
            errors.append({"loc": f"known_cells[{idx}]", "msg": msg})
        else:
            seen[key] = cell.value

    if errors:
        raise InputError(errors)
