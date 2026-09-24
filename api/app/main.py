"""FastAPI 入口：四向投影重建复核 API。"""

from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import (
    InputError,
    ReconstructRequest,
    ReconstructResponse,
    SolutionOut,
    validate_request,
)
from .solver import solve_reconstruction

app = FastAPI(
    title="复合材料射线检测四向投影重建复核台",
    version="1.0.0",
)

# 生产环境经 Web 端 nginx 同源反代；放开 CORS 便于本地联调。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(InputError)
async def input_error_handler(_request: Request, exc: InputError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": exc.errors})


@app.exception_handler(RequestValidationError)
async def pydantic_error_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    """把 pydantic 校验错误改写成与业务校验一致的可定位结构。"""
    detail = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"] if part != "body")
        detail.append({"loc": loc or "body", "msg": err["msg"]})
    return JSONResponse(status_code=422, content={"detail": detail})


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/reconstruct", response_model=ReconstructResponse)
def reconstruct(req: ReconstructRequest) -> ReconstructResponse:
    # 业务校验：长度、逐线上限、已知单元重复/冲突，全部可定位。
    validate_request(req)

    known = [(c.row, c.col, c.value) for c in req.known_cells]
    started = time.perf_counter()
    try:
        status, solutions = solve_reconstruction(
            req.rows,
            req.cols,
            req.row_sums,
            req.col_sums,
            req.diag_sums,
            req.antidiag_sums,
            known,
        )
    except RuntimeError as exc:  # 求解器超时等未决情形
        return JSONResponse(
            status_code=503, content={"detail": [{"loc": "solver", "msg": str(exc)}]}
        )
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    out = []
    for bits in solutions:
        grid = [
            bits[r * req.cols : (r + 1) * req.cols] for r in range(req.rows)
        ]
        out.append(SolutionOut(grid=grid, bits="".join(map(str, bits))))
    return ReconstructResponse(
        status=status,
        rows=req.rows,
        cols=req.cols,
        solutions=out,
        elapsed_ms=elapsed_ms,
    )
