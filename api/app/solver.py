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

启用"连续夹杂体"约束（require_connected=True）时，连通性与四向投影、
已知单元、字典序排序在**同一个模型**中一起求解，不会先取旧见证再事后筛除：
以行优先最早的 1 单元为唯一根，根层数为 0；其余每个 1 单元必须有一个
上下左右相邻（对角不算）的 1 单元，其层数恰小 1，由此所有 1 单元互达。
全 0 网格（无夹杂）天然成立。
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
    def _add_connectedness(
        self,
        model: cp_model.CpModel,
        x: list[cp_model.IntVar],
    ) -> None:
        """加入"全部 1 单元四邻连通（对角不连接）"约束；全 0 网格仍可行。

        取行优先最早的 1 单元为唯一根：
          * 根 root[i] 恰好为"该单元为 1 且此前没有 1"；
          * 根层数 lvl=0，非根 1 单元层数 1..n-1；
          * 每个非根 1 单元至少有一个四邻 1 前驱，层数恰小 1
            （沿层数严格递减必能到达唯一的根，故所有 1 互达）。
        """
        rows, cols, n = self.rows, self.cols, self.n

        # 前缀或 pref[i] = 1 表示位置 i 之前（不含）已出现过 1。
        pref = [model.NewConstant(0)]
        for i in range(n - 1):
            p = model.NewBoolVar(f"pref{i}")
            model.AddMaxEquality(p, [pref[-1], x[i]])
            pref.append(p)

        root = [model.NewBoolVar(f"root{i}") for i in range(n)]
        for i in range(n):
            # root[i] = x[i] 且 pref[i]=0
            model.Add(root[i] == 1).only_enforce_if(x[i], pref[i].Not())
            model.Add(root[i] == 0).only_enforce_if(x[i].Not())
            model.Add(root[i] == 0).only_enforce_if(pref[i])
        # 有夹杂时恰好一个根；无夹杂时没有根（全 0 网格仍成立）。
        total = sum(x)
        has_ones = model.NewBoolVar("has_ones")
        model.Add(total >= 1).OnlyEnforceIf(has_ones)
        model.Add(total == 0).OnlyEnforceIf(has_ones.Not())
        model.Add(sum(root) == 1).OnlyEnforceIf(has_ones)
        model.Add(sum(root) == 0).OnlyEnforceIf(has_ones.Not())

        lvl = [model.NewIntVar(0, n - 1, f"lvl{i}") for i in range(n)]
        neighbors: list[list[int]] = [[] for _ in range(n)]
        for r in range(rows):
            for c in range(cols):
                i = r * cols + c
                if r > 0:
                    neighbors[i].append(i - cols)
                if r + 1 < rows:
                    neighbors[i].append(i + cols)
                if c > 0:
                    neighbors[i].append(i - 1)
                if c + 1 < cols:
                    neighbors[i].append(i + 1)

        for i in range(n):
            # 0 单元不参与层结构。
            model.Add(lvl[i] == 0).only_enforce_if(x[i].Not())
            # 根层数为 0。
            model.Add(lvl[i] == 0).only_enforce_if(root[i])
            # 非根的 1 单元层数 >=1，且必须存在一个层数恰小 1 的四邻 1 前驱。
            nonroot_one = model.NewBoolVar(f"nonroot_one{i}")
            model.Add(nonroot_one == 1).only_enforce_if(x[i], root[i].Not())
            model.Add(nonroot_one == 0).only_enforce_if(x[i].Not())
            model.Add(nonroot_one == 0).only_enforce_if(root[i])
            model.Add(lvl[i] >= 1).only_enforce_if(nonroot_one)
            preds = []
            for j in neighbors[i]:
                p = model.NewBoolVar(f"pred{j}_of{i}")
                # 前驱边：两端均为 1，且层数严格相差 1（从 j 指向 i）。
                # 蕴含式保证 0 单元上不可能存在前驱边，无需反向约束。
                model.Add(x[i] == 1).only_enforce_if(p)
                model.Add(x[j] == 1).only_enforce_if(p)
                model.Add(lvl[i] == lvl[j] + 1).only_enforce_if(p)
                preds.append(p)
            model.Add(sum(preds) >= 1).only_enforce_if(nonroot_one)

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
        # 连续夹杂体约束：与投影、已知单元同模型一起求解。
        if self.require_connected:
            self._add_connectedness(model, x)
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
        require_connected=require_connected,
    )
    return solver.solve()
