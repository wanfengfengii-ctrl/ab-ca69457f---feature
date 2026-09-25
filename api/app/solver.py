"""四向投影二值网格重建求解器。

使用 OR-Tools CP-SAT 完备求解，分四个阶段（每阶段重建干净模型）：
  1. 可行性判定 -> 无解；
  2. 以 <=60 位为一块逐块最小化"行优先位串"对应的二进制整数，
     得到零先于一字典序下的最小解（每块的最优值唯一确定该块位型，
     因此结果与求解器内部启发式无关，完全确定）；
  3. 加入"至少一个单元不同"约束再次判定可行性 -> 唯一；
  4. 否则在同一排除约束下再次分块最小化，得到第二小的解 -> 多解。

不使用贪心、随机搜索，也不会"找到首解后推测唯一性"：
唯一性由 CP-SAT 对排除首解后的模型给出不可行证明来判定。

可选的"连续夹杂体"约束（require_connected=True）：任一可行网格中
所有值为 1 的单元必须仅经上下左右相邻的 1 单元互达（对角接触不算
连通；全 0 的无夹杂网格仍然合法）。该约束与四向投影、已知单元在
同一模型内建模——可行性判定、字典序分块最小化、唯一性证明都在
同一次完备求解中完成，不是先求出旧见证再筛除。
"""

from __future__ import annotations

from ortools.sat.python import cp_model

# 每块位数：2**60 的系数与和均在 int64 范围内，CP-SAT 可安全处理。
_CHUNK_BITS = 60
# 单次求解的兜底时间上限；12x12 规模正常远低于此值。
_MAX_TIME_SECONDS = 30.0


class ReconstructionSolver:
    """对一组四向投影 + 已知单元进行完备重建。"""

    def __init__(
        self,
        rows: int,
        cols: int,
        row_sums: list[int],
        col_sums: list[int],
        diag_sums: list[int],
        antidiag_sums: list[int],
        known_cells: list[tuple[int, int, int]],
        require_connected: bool = False,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.n = rows * cols
        self.row_sums = row_sums
        self.col_sums = col_sums
        self.diag_sums = diag_sums
        self.antidiag_sums = antidiag_sums
        self.known_cells = known_cells
        self.require_connected = require_connected

    # ------------------------------------------------------------------
    # 模型构建
    # ------------------------------------------------------------------
    def _build_model(
        self, exclude_bits: list[int] | None = None
    ) -> tuple[cp_model.CpModel, list[cp_model.IntVar]]:
        """构建投影约束模型；可选地排除某个给定位串。"""
        rows, cols = self.rows, self.cols
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x{i}") for i in range(self.n)]

        def at(r: int, c: int):
            return x[r * cols + c]

        # 行投影（自上而下）
        for r in range(rows):
            model.Add(sum(at(r, c) for c in range(cols)) == self.row_sums[r])
        # 列投影（自左向右）
        for c in range(cols):
            model.Add(sum(at(r, c) for r in range(rows)) == self.col_sums[c])
        # 对角投影：按 (行 - 列) 差递增，d 从 -(cols-1) 到 rows-1
        for k, d in enumerate(range(-(cols - 1), rows)):
            model.Add(
                sum(
                    at(r, c)
                    for r in range(rows)
                    for c in range(cols)
                    if r - c == d
                )
                == self.diag_sums[k]
            )
        # 副对角投影：按 (行 + 列) 和递增，s 从 0 到 rows+cols-2
        for k, s in enumerate(range(rows + cols - 1)):
            model.Add(
                sum(
                    at(r, c)
                    for r in range(rows)
                    for c in range(cols)
                    if r + c == s
                )
                == self.antidiag_sums[k]
            )
        # 已知单元
        for r, c, v in self.known_cells:
            model.Add(at(r, c) == v)
        # 连续夹杂体约束：所有 1 单元四向连通（与投影同模型求解）
        if self.require_connected:
            self._add_connectivity(model, x)
        # 排除位串：至少一个单元取值不同
        if exclude_bits is not None:
            model.Add(
                sum(
                    x[i] if bit == 0 else x[i].Not()
                    for i, bit in enumerate(exclude_bits)
                )
                >= 1
            )
        return model, x

    def _add_connectivity(
        self, model: cp_model.CpModel, x: list[cp_model.IntVar]
    ) -> None:
        """所有值为 1 的单元必须仅经上下左右相邻的 1 单元互达。

        用单源网络流编码：在 1 单元中恰选一个根，根注入 T 单位流
        （T = 1 单元总数，由投影等式固定），其余每个 1 单元净消耗
        1 单位，流只能经过 1 单元之间的四向边。
          * 若 1 单元四向连通：取任一生成树沿树边送流即可满足；
          * 若流可行：每个非根 1 单元都有来自根的流路径，路径上
            全为 1 单元，故全部 1 单元同属一个四向连通分量。
        两个方向都成立，因此该编码与"单团簇"完全等价。
        T <= 1（无夹杂或单个夹杂）时自然满足，无需建模。
        """
        rows, cols = self.rows, self.cols
        total = sum(self.row_sums)  # 1 单元总数（投影等式已固定）
        if total <= 1:
            return
        n = self.n

        def neighbors(idx: int):
            r, c = divmod(idx, cols)
            if r > 0:
                yield idx - cols
            if r + 1 < rows:
                yield idx + cols
            if c > 0:
                yield idx - 1
            if c + 1 < cols:
                yield idx + 1

        # 1 单元中恰选一个根
        root = [model.NewBoolVar(f"root{i}") for i in range(n)]
        for i in range(n):
            model.Add(root[i] <= x[i])
        model.AddExactlyOne(root)

        # 四向相邻对之间的有向流；流只能经过 1 单元
        flow: dict[tuple[int, int], cp_model.IntVar] = {}
        for i in range(n):
            for j in neighbors(i):
                f = model.NewIntVar(0, total, f"f{i}_{j}")
                flow[(i, j)] = f
                model.Add(f <= total * x[i])
                model.Add(f <= total * x[j])
        # 流守恒：非根 1 单元净消耗 1；根净注入 total-1（自身也消耗 1）
        for i in range(n):
            inflow = sum(flow[(j, i)] for j in neighbors(i))
            outflow = sum(flow[(i, j)] for j in neighbors(i))
            model.Add(inflow - outflow == x[i] - total * root[i])

    @staticmethod
    def _new_solver() -> cp_model.CpSolver:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = _MAX_TIME_SECONDS
        solver.parameters.random_seed = 20260920
        return solver

    # ------------------------------------------------------------------
    # 求解原语
    # ------------------------------------------------------------------
    @staticmethod
    def _feasible(model: cp_model.CpModel, solver: cp_model.CpSolver) -> bool:
        status = solver.Solve(model)
        if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            return True
        if status == cp_model.INFEASIBLE:
            return False
        raise RuntimeError(f"求解器未能在时限内给出结论（状态={status}）")

    def _lex_min_bits(
        self, model: cp_model.CpModel, x: list[cp_model.IntVar], solver: cp_model.CpSolver
    ) -> list[int]:
        """在（假定可行的）模型上求行优先位串 0<1 字典序最小解。

        将 n 位切成若干 <=60 位的块，逐块最小化"以该块首位为最高位
        的二进制整数"；块的最优整数值与位型一一对应。目标值经 double
        回读会丢失精度，因此最优块的位型直接从解中逐位读取并锁定，
        再处理下一块，最终得到整体字典序最小解。
        """
        bits: list[int] = []
        n = self.n
        for start in range(0, n, _CHUNK_BITS):
            end = min(start + _CHUNK_BITS, n)
            objective = sum(x[i] * (1 << (end - 1 - i)) for i in range(start, end))
            model.Minimize(objective)
            status = solver.Solve(model)
            if status != cp_model.OPTIMAL:
                raise RuntimeError(f"字典序最小化未收敛到最优（状态={status}）")
            model.ClearObjective()
            # 最优解中该块的位型即块最优值的二进制展开，逐位锁定前缀。
            for i in range(start, end):
                bit = solver.Value(x[i])
                model.Add(x[i] == bit)
                bits.append(bit)
        return bits

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def solve(self) -> tuple[str, list[list[int]]]:
        """返回 (status, solutions)。

        status ∈ {"no_solution", "unique", "multiple"}；
        solutions 为按行优先位串（0<1）升序排列的 0~2 个解，
        每个解是长度为 rows*cols 的 0/1 列表。
        """
        model, x = self._build_model()
        solver = self._new_solver()
        if not self._feasible(model, solver):
            return "no_solution", []

        model, x = self._build_model()
        solver = self._new_solver()
        first = self._lex_min_bits(model, x, solver)

        # 排除首解后判定是否还有其它解。
        model, x = self._build_model(exclude_bits=first)
        solver = self._new_solver()
        if not self._feasible(model, solver):
            return "unique", [first]

        second = self._lex_min_bits(model, x, solver)
        return "multiple", [first, second]


def solve_reconstruction(
    rows: int,
    cols: int,
    row_sums: list[int],
    col_sums: list[int],
    diag_sums: list[int],
    antidiag_sums: list[int],
    known_cells: list[tuple[int, int, int]],
    require_connected: bool = False,
) -> tuple[str, list[list[int]]]:
    """便捷入口：构建求解器并执行完备求解。"""
    solver = ReconstructionSolver(
        rows,
        cols,
        row_sums,
        col_sums,
        diag_sums,
        antidiag_sums,
        known_cells,
        require_connected,
    )
    return solver.solve()
