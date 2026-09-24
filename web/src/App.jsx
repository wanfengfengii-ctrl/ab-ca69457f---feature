import { Fragment, useEffect, useMemo, useState } from "react";

const MIN_SIDE = 4;
const MAX_SIDE = 12;

// 内置示例：6×6 多解（歧义）实例，两见证在 12 个单元上不同。
const EXAMPLE_MULTIPLE = {
  rows: 6,
  cols: 6,
  row_sums: [3, 2, 4, 2, 2, 4],
  col_sums: [1, 3, 2, 3, 3, 5],
  diag_sums: [1, 1, 2, 1, 3, 4, 2, 1, 1, 1, 0],
  antidiag_sums: [0, 1, 1, 2, 1, 3, 3, 1, 2, 2, 1],
};

// 内置示例：6×6 唯一解实例。
const EXAMPLE_UNIQUE = {
  rows: 6,
  cols: 6,
  row_sums: [5, 4, 5, 4, 2, 2],
  col_sums: [5, 4, 3, 5, 2, 3],
  diag_sums: [1, 1, 3, 2, 3, 4, 2, 2, 2, 1, 1],
  antidiag_sums: [1, 1, 3, 3, 4, 4, 2, 4, 0, 0, 0],
};

const LOC_NAMES = {
  row_sums: "行投影",
  col_sums: "列投影",
  diag_sums: "对角投影(r−c)",
  antidiag_sums: "副对角投影(r+c)",
  known_cells: "已知单元",
  rows: "行数",
  cols: "列数",
  solver: "求解器",
};

const toStrings = (arr) => arr.map(String);
const keyOf = (r, c) => `${r},${c}`;
const clampSide = (v) => Math.min(MAX_SIDE, Math.max(MIN_SIDE, v));

function resizeStrings(arr, n) {
  const next = arr.slice(0, n);
  while (next.length < n) next.push("");
  return next;
}

function translateLoc(loc) {
  if (typeof loc !== "string") return "输入";
  return loc.replace(/^[a-z_]+/, (m) => LOC_NAMES[m] || m);
}

function parseIntStrict(text) {
  const t = String(text).trim();
  if (t === "") return { ok: false, reason: "empty" };
  if (!/^-?\d+$/.test(t)) return { ok: false, reason: "nan", raw: t };
  return { ok: true, value: parseInt(t, 10) };
}

function computeLineSums(grid, R, C) {
  const row = Array(R).fill(0);
  const col = Array(C).fill(0);
  const diag = Array(R + C - 1).fill(0);
  const anti = Array(R + C - 1).fill(0);
  grid.forEach((cells, r) =>
    cells.forEach((v, c) => {
      if (v === 1) {
        row[r] += 1;
        col[c] += 1;
        diag[r - c + (C - 1)] += 1;
        anti[r + c] += 1;
      }
    })
  );
  return { row, col, diag, anti };
}

// 四邻连通团簇分析（对角接触不连接）。
// 返回夹杂数量、团簇数、首个非主团簇的行优先最早坐标
// （主团簇 = 含行优先最早 1 单元的团簇）。
function witnessStats(grid, R, C) {
  const comp = Array.from({ length: R }, () => Array(C).fill(-1));
  const starts = [];
  let count = 0;
  for (let r = 0; r < R; r += 1) {
    for (let c = 0; c < C; c += 1) {
      if (grid[r][c] !== 1 || comp[r][c] !== -1) continue;
      const id = starts.length;
      starts.push([r, c]);
      const stack = [[r, c]];
      comp[r][c] = id;
      while (stack.length > 0) {
        const [cr, cc] = stack.pop();
        count += 1;
        const nbrs = [
          cr > 0 ? [cr - 1, cc] : null,
          cr + 1 < R ? [cr + 1, cc] : null,
          cc > 0 ? [cr, cc - 1] : null,
          cc + 1 < C ? [cr, cc + 1] : null,
        ];
        nbrs.forEach((p) => {
          if (p && grid[p[0]][p[1]] === 1 && comp[p[0]][p[1]] === -1) {
            comp[p[0]][p[1]] = id;
            stack.push(p);
          }
        });
      }
    }
  }
  return {
    count,
    clusters: starts.length,
    firstNonPrimary: starts.length > 1 ? starts[1] : null,
  };
}

function Badge({ actual, targetText, compact = false }) {
  const parsed = parseIntStrict(targetText);
  if (!parsed.ok)
    return <span className="badge na">{compact ? `${actual}/?` : `实算 ${actual} / 目标 ?`}</span>;
  const ok = actual === parsed.value;
  return (
    <span className={`badge ${ok ? "ok" : "bad"}`}>
      {compact
        ? `${actual}/${parsed.value}${ok ? "✓" : "✗"}`
        : `实算 ${actual} / 目标 ${parsed.value} ${ok ? "✓" : "✗"}`}
    </span>
  );
}

export default function App() {
  const [rows, setRows] = useState(EXAMPLE_MULTIPLE.rows);
  const [cols, setCols] = useState(EXAMPLE_MULTIPLE.cols);
  const [rowSums, setRowSums] = useState(() => toStrings(EXAMPLE_MULTIPLE.row_sums));
  const [colSums, setColSums] = useState(() => toStrings(EXAMPLE_MULTIPLE.col_sums));
  const [diagSums, setDiagSums] = useState(() => toStrings(EXAMPLE_MULTIPLE.diag_sums));
  const [antiSums, setAntiSums] = useState(() => toStrings(EXAMPLE_MULTIPLE.antidiag_sums));
  const [known, setKnown] = useState({}); // {"r,c": 0|1}
  const [connected, setConnected] = useState(false); // 连续夹杂体约束
  const [view, setView] = useState("edit"); // "edit" | "result"
  const [result, setResult] = useState(null);
  const [witnessIdx, setWitnessIdx] = useState(0);
  const [errors, setErrors] = useState([]);
  const [loading, setLoading] = useState(false);

  const lineCount = rows + cols - 1;

  // 任一输入（含连续夹杂体开关）变化后，旧见证不再对应当前条件，立即清除，
  // 不等待下次请求；失败响应与无解同样不会保留旧网格（见 reconstruct）。
  useEffect(() => {
    setResult(null);
    setErrors([]);
    setWitnessIdx(0);
    setView("edit");
  }, [rows, cols, rowSums, colSums, diagSums, antiSums, known, connected]);

  const diagLen = (d) => {
    let n = 0;
    for (let r = 0; r < rows; r += 1) {
      const c = r - d;
      if (c >= 0 && c < cols) n += 1;
    }
    return n;
  };
  const antiLen = (s) => {
    let n = 0;
    for (let r = 0; r < rows; r += 1) {
      const c = s - r;
      if (c >= 0 && c < cols) n += 1;
    }
    return n;
  };

  // ---------- 状态维护 ----------
  function clearOutcome() {
    setResult(null);
    setErrors([]);
    setWitnessIdx(0);
  }

  function applySize(nextRows, nextCols) {
    const R = clampSide(nextRows);
    const C = clampSide(nextCols);
    setRows(R);
    setCols(C);
    setRowSums((a) => resizeStrings(a, R));
    setColSums((a) => resizeStrings(a, C));
    setDiagSums((a) => resizeStrings(a, R + C - 1));
    setAntiSums((a) => resizeStrings(a, R + C - 1));
    setKnown((k) =>
      Object.fromEntries(
        Object.entries(k).filter(([key]) => {
          const [r, c] = key.split(",").map(Number);
          return r < R && c < C;
        })
      )
    );
    clearOutcome();
    setView("edit");
  }

  function loadExample(ex) {
    setRows(ex.rows);
    setCols(ex.cols);
    setRowSums(toStrings(ex.row_sums));
    setColSums(toStrings(ex.col_sums));
    setDiagSums(toStrings(ex.diag_sums));
    setAntiSums(toStrings(ex.antidiag_sums));
    setKnown({});
    clearOutcome();
    setView("edit");
  }

  function randomInstance() {
    const grid = Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => (Math.random() < 0.45 ? 1 : 0))
    );
    const sums = computeLineSums(grid, rows, cols);
    setRowSums(toStrings(sums.row));
    setColSums(toStrings(sums.col));
    setDiagSums(toStrings(sums.diag));
    setAntiSums(toStrings(sums.anti));
    setKnown({});
    clearOutcome();
    setView("edit");
  }

  function cycleKnown(r, c) {
    const key = keyOf(r, c);
    setKnown((k) => {
      const next = { ...k };
      if (!(key in next)) next[key] = 1;
      else if (next[key] === 1) next[key] = 0;
      else delete next[key];
      return next;
    });
  }

  // ---------- 客户端校验（与后端规则一致，逐线可定位） ----------
  function collectInput() {
    const errs = [];
    const parseLine = (label, arr, expected, capOf) => {
      if (arr.length !== expected) {
        errs.push({ loc: label, msg: `长度应为 ${expected}，实际为 ${arr.length}` });
      }
      const nums = arr.map((s, i) => {
        const p = parseIntStrict(s);
        if (!p.ok) {
          errs.push({
            loc: `${label}[${i}]`,
            msg: p.reason === "empty" ? "不能为空" : `“${p.raw}”不是整数`,
          });
          return null;
        }
        return p.value;
      });
      if (arr.length === expected) {
        nums.forEach((v, i) => {
          if (v === null) return;
          const cap = capOf(i);
          if (v < 0) errs.push({ loc: `${label}[${i}]`, msg: `值 ${v} 不能为负数` });
          else if (v > cap)
            errs.push({ loc: `${label}[${i}]`, msg: `值 ${v} 超过该线单元数上限 ${cap}` });
        });
      }
      return nums;
    };

    const rowNums = parseLine("行投影", rowSums, rows, () => cols);
    const colNums = parseLine("列投影", colSums, cols, () => rows);
    const diagNums = parseLine("对角投影(r−c)", diagSums, lineCount, (i) =>
      diagLen(i - (cols - 1))
    );
    const antiNums = parseLine("副对角投影(r+c)", antiSums, lineCount, (i) => antiLen(i));
    return { errs, rowNums, colNums, diagNums, antiNums };
  }

  async function reconstruct() {
    const { errs, rowNums, colNums, diagNums, antiNums } = collectInput();
    if (errs.length > 0) {
      // 非法输入：清除旧网格，逐条给出可定位反馈。
      setErrors(errs);
      setResult(null);
      setView("edit");
      return;
    }
    setLoading(true);
    setErrors([]);
    try {
      const resp = await fetch("/api/reconstruct", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          rows,
          cols,
          row_sums: rowNums,
          col_sums: colNums,
          diag_sums: diagNums,
          antidiag_sums: antiNums,
          known_cells: Object.entries(known).map(([key, v]) => {
            const [r, c] = key.split(",").map(Number);
            return { row: r, col: c, value: v };
          }),
          connected,
        }),
      });
      const body = await resp.json();
      if (resp.ok) {
        setResult(body);
        setWitnessIdx(0);
        setView(body.solutions.length > 0 ? "result" : "edit");
      } else {
        // 后端 422/503：统一转成可定位错误列表，并清除旧网格。
        const detail = Array.isArray(body.detail)
          ? body.detail.map((e) => ({ loc: translateLoc(e.loc), msg: e.msg }))
          : [{ loc: "服务器", msg: String(body.detail ?? "未知错误") }];
        setErrors(detail);
        setResult(null);
        setView("edit");
      }
    } catch (e) {
      setErrors([{ loc: "网络", msg: `请求失败：${e.message}` }]);
      setResult(null);
      setView("edit");
    } finally {
      setLoading(false);
    }
  }

  // ---------- 派生数据 ----------
  const witness =
    result && result.solutions.length > 0
      ? result.solutions[Math.min(witnessIdx, result.solutions.length - 1)]
      : null;
  const showWitness = view === "result" && witness !== null;

  const diffSet = useMemo(() => {
    const s = new Set();
    if (result && result.solutions.length === 2) {
      const [a, b] = result.solutions;
      for (let r = 0; r < rows; r += 1)
        for (let c = 0; c < cols; c += 1)
          if (a.grid[r][c] !== b.grid[r][c]) s.add(keyOf(r, c));
    }
    return s;
  }, [result, rows, cols]);

  const actual = useMemo(
    () => (showWitness ? computeLineSums(witness.grid, rows, cols) : null),
    [showWitness, witness, rows, cols]
  );

  const statsList = useMemo(() => {
    if (!result) return [];
    return result.solutions.map((s) => witnessStats(s.grid, rows, cols));
  }, [result, rows, cols]);

  const stats = witnessIdx < statsList.length ? statsList[witnessIdx] : null;

  const totals = useMemo(() => {
    const sum = (arr) =>
      arr.reduce((acc, s) => {
        const p = parseIntStrict(s);
        return acc + (p.ok ? p.value : 0);
      }, 0);
    return {
      row: sum(rowSums),
      col: sum(colSums),
      diag: sum(diagSums),
      anti: sum(antiSums),
    };
  }, [rowSums, colSums, diagSums, antiSums]);

  const totalsConsistent =
    totals.row === totals.col &&
    totals.col === totals.diag &&
    totals.diag === totals.anti;

  // ---------- 渲染 ----------
  function renderCell(r, c) {
    const key = keyOf(r, c);
    if (showWitness) {
      const v = witness.grid[r][c];
      const cls = ["cell", v === 1 ? "one" : "zero"];
      if (diffSet.has(key)) cls.push("diff");
      if (key in known) cls.push("known");
      return (
        <div key={key} className={cls.join(" ")} title={`(${r}, ${c})`}>
          {v}
          {key in known && <span className="pin" title="已知单元" />}
        </div>
      );
    }
    const v = known[key];
    const cls = ["cell", "editable"];
    if (v === 1) cls.push("known-one");
    else if (v === 0) cls.push("known-zero");
    return (
      <div
        key={key}
        className={cls.join(" ")}
        title={`(${r}, ${c}) 点击循环：未知 → 1 → 0 → 清除`}
        onClick={() => cycleKnown(r, c)}
      >
        {v === undefined ? "" : v}
      </div>
    );
  }

  function sumInput(value, onChange, placeholder) {
    return (
      <input
        className="sumInput"
        type="text"
        inputMode="numeric"
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
      />
    );
  }

  const statusBanner = () => {
    if (errors.length > 0)
      return <div className="banner error">输入非法：共 {errors.length} 处问题，见下方列表</div>;
    if (!result) return null;
    const connNote = result.connected ? "且全部夹杂物四邻连通" : "";
    if (result.status === "no_solution")
      return (
        <div className="banner nosol">
          无解：当前四向投影、已知单元{result.connected ? "与连续夹杂体连通约束" : ""}
          不存在可行网格（已清除旧网格）
        </div>
      );
    if (result.status === "unique")
      return (
        <div className="banner unique">
          唯一解：投影与已知单元{connNote}唯一确定该网格（{result.elapsed_ms} ms）
        </div>
      );
    return (
      <div className="banner multiple">
        多解：存在歧义，以下为行优先位串（0&lt;1）最小的两份见证，差异单元已高亮
        {result.connected ? "，均满足连续夹杂体四邻连通" : ""}（{result.elapsed_ms} ms）
      </div>
    );
  };

  return (
    <div className="page">
      <header>
        <h1>复合材料射线检测 · 四向投影重建复核台</h1>
        <p className="subtitle">
          编辑四向投影与已知单元 → 调用真实 API 完备重建 → 切换核对歧义见证。
          行列从 0 编号；对角投影按（行−列）递增，副对角投影按（行+列）递增。
        </p>
      </header>

      <div className="toolbar">
        <label>
          行数
          <input
            type="number"
            min={MIN_SIDE}
            max={MAX_SIDE}
            value={rows}
            onChange={(e) => {
              const v = Number(e.target.value);
              if (Number.isInteger(v) && v >= MIN_SIDE && v <= MAX_SIDE && v !== rows)
                applySize(v, cols);
            }}
          />
        </label>
        <label>
          列数
          <input
            type="number"
            min={MIN_SIDE}
            max={MAX_SIDE}
            value={cols}
            onChange={(e) => {
              const v = Number(e.target.value);
              if (Number.isInteger(v) && v >= MIN_SIDE && v <= MAX_SIDE && v !== cols)
                applySize(rows, v);
            }}
          />
        </label>
        <button className="primary" onClick={reconstruct} disabled={loading}>
          {loading ? "重建中…" : "重建"}
        </button>
        <label className="connToggle" title="要求任一有夹杂的返回网格中全部 1 单元仅经上下左右相邻互达（对角接触不连接）；无夹杂网格仍可成立">
          <input
            type="checkbox"
            checked={connected}
            onChange={(e) => setConnected(e.target.checked)}
          />
          连续夹杂体约束（四邻连通，对角不算）
        </label>
        <button onClick={() => loadExample(EXAMPLE_MULTIPLE)}>载入多解示例</button>
        <button onClick={() => loadExample(EXAMPLE_UNIQUE)}>载入唯一解示例</button>
        <button onClick={randomInstance}>随机实例</button>
        <button onClick={() => setKnown({})}>
          清空已知单元
        </button>
        <button
          onClick={() => {
            setRowSums(Array(rows).fill(""));
            setColSums(Array(cols).fill(""));
            setDiagSums(Array(lineCount).fill(""));
            setAntiSums(Array(lineCount).fill(""));
            clearOutcome();
            setView("edit");
          }}
        >
          清空投影
        </button>
        <span className={`totals ${totalsConsistent ? "ok" : "bad"}`}>
          总和：行 {totals.row} ｜ 列 {totals.col} ｜ 对角 {totals.diag} ｜ 副对角 {totals.anti}
          {totalsConsistent ? "（一致）" : "（不一致，必无解）"}
        </span>
      </div>

      {statusBanner()}

      {errors.length > 0 && (
        <div className="errorPanel">
          <h3>输入问题（{errors.length}）</h3>
          <ul>
            {errors.map((e, i) => (
              <li key={i}>
                <code>{e.loc}</code>：{e.msg}
              </li>
            ))}
          </ul>
        </div>
      )}

      {result && result.status === "no_solution" && (
        <div className="errorPanel">
          <h3>无解排查线索</h3>
          <ul>
            {!totalsConsistent && (
              <li>
                四组投影总和不一致：行 {totals.row}、列 {totals.col}、对角 {totals.diag}、副对角{" "}
                {totals.anti}（可行实例必须四者相等）
              </li>
            )}
            {totalsConsistent && (
              <li>四组投影总和一致（均为 {totals.row}），矛盾来自投影结构或已知单元约束</li>
            )}
            {result.connected && (
              <li>
                已启用连续夹杂体约束：全部 1 单元必须仅经上下左右相邻互达，对角接触不算连接
                ——可能存在满足四向投影但夹杂物被基材分隔成多个团簇的伪见证，可关闭该约束对照
              </li>
            )}
            <li>已知单元共 {Object.keys(known).length} 个，可尝试逐个清除定位冲突来源</li>
          </ul>
        </div>
      )}

      <div className="main">
        <section className="gridPanel">
          <div className="viewSwitch">
            <button
              className={view === "edit" ? "active" : ""}
              onClick={() => setView("edit")}
            >
              编辑已知单元
            </button>
            <button
              className={view === "result" ? "active" : ""}
              disabled={!result || result.solutions.length === 0}
              onClick={() => setView("result")}
            >
              查看重建见证
            </button>
            {result && result.solutions.length === 2 && view === "result" && (
              <span className="witnessTabs">
                {[0, 1].map((i) => (
                  <button
                    key={i}
                    className={witnessIdx === i ? "active" : ""}
                    onClick={() => setWitnessIdx(i)}
                  >
                    见证 {i + 1}
                  </button>
                ))}
              </span>
            )}
          </div>

          {showWitness && stats && (
            <div className="witnessStats">
              {result.solutions.map((_, i) => {
                const s = statsList[i];
                const nonPrimary = s.firstNonPrimary
                  ? `(${s.firstNonPrimary[0]}, ${s.firstNonPrimary[1]})`
                  : "—";
                return (
                  <span
                    key={i}
                    className={`wstat ${i === witnessIdx ? "active" : ""}`}
                    title="主团簇 = 含行优先最早夹杂物的团簇"
                  >
                    见证 {i + 1}：夹杂数量 <b>{s.count}</b> ｜ 连通团簇数{" "}
                    <b className={s.clusters > 1 ? "bad" : "ok"}>{s.clusters}</b> ｜
                    首个非主团簇坐标 <b>{nonPrimary}</b>
                  </span>
                );
              })}
              {result.connected && stats.clusters > 1 && (
                <span className="wstat warn">
                  异常：已启用连续夹杂体约束，返回网格却存在 {stats.clusters} 个团簇
                </span>
              )}
            </div>
          )}

          <div
            className="grid"
            style={{ gridTemplateColumns: `44px repeat(${cols}, 46px) 160px` }}
          >
            <div className="corner">行＼列</div>
            {Array.from({ length: cols }, (_, c) => (
              <div key={c} className="hdr">
                {c}
              </div>
            ))}
            <div className="corner">行投影（实算/目标）</div>

            {Array.from({ length: rows }, (_, r) => (
              <Fragment key={r}>
                <div className="hdr">{r}</div>
                {Array.from({ length: cols }, (_, c) => renderCell(r, c))}
                <div className="sumCell">
                  {sumInput(rowSums[r], (v) =>
                    setRowSums((a) => a.map((x, i) => (i === r ? v : x)))
                  )}
                  {showWitness && actual && (
                    <Badge actual={actual.row[r]} targetText={rowSums[r]} />
                  )}
                </div>
              </Fragment>
            ))}

            <div className="corner">列投影</div>
            {Array.from({ length: cols }, (_, c) => (
              <div key={c} className="sumCell col">
                {sumInput(colSums[c], (v) =>
                  setColSums((a) => a.map((x, i) => (i === c ? v : x)))
                )}
                {showWitness && actual && (
                  <Badge actual={actual.col[c]} targetText={colSums[c]} compact />
                )}
              </div>
            ))}
            <div className="corner" />
          </div>

          <div className="legend">
            <span><i className="sw one" /> 1（夹杂物）</span>
            <span><i className="sw zero" /> 0（基材）</span>
            <span><i className="sw diff" /> 两见证差异单元</span>
            <span><i className="sw knownsw" /> 已知单元（角标）</span>
            {!showWitness && <span>当前为编辑模式：点击网格循环设置已知单元</span>}
          </div>

          {showWitness && (
            <div className="bits">
              行优先位串（0&lt;1）：<code>{witness.bits}</code>
              {result.solutions.length === 2 && (
                <span className="diffNote">
                  两见证在 {diffSet.size} 个单元上不同：
                  {[...diffSet].map((k) => `(${k})`).join(" ")}
                </span>
              )}
            </div>
          )}
        </section>

        <aside className="sidePanels">
          <section className="panel">
            <h3>对角投影（按 行−列 递增）</h3>
            {diagSums.map((s, k) => {
              const d = k - (cols - 1);
              return (
                <div className="line" key={k}>
                  <span className="lbl">
                    d={d}（{diagLen(d)} 格）
                  </span>
                  {sumInput(s, (v) =>
                    setDiagSums((a) => a.map((x, i) => (i === k ? v : x)))
                  )}
                  {showWitness && actual && (
                    <Badge actual={actual.diag[k]} targetText={s} />
                  )}
                </div>
              );
            })}
          </section>
          <section className="panel">
            <h3>副对角投影（按 行+列 递增）</h3>
            {antiSums.map((s, k) => (
              <div className="line" key={k}>
                <span className="lbl">
                  s={k}（{antiLen(k)} 格）
                </span>
                {sumInput(s, (v) =>
                  setAntiSums((a) => a.map((x, i) => (i === k ? v : x)))
                )}
                {showWitness && actual && (
                  <Badge actual={actual.anti[k]} targetText={s} />
                )}
              </div>
            ))}
          </section>
        </aside>
      </div>
    </div>
  );
}
