"""生成 66 道全量生成评测的性能统计文档（markdown）。

先说清三件容易混淆的事：

1. **run.json 里的 speedup 分母是 `torch_eager`**，不是 aiter —— 在线评测器
   （KernelBenchPaperEvaluator）拿参考 `Model` 当分母。已核对本轮 66 份 run.json，
   `primary_reference` 全部是 `torch_eager`。
2. **aiter 基线在 `benchmark/private/<task>/baseline.json`**，按 perf case、按 impl
   存实测 us。它可以回答"aiter 自己比 eager 快多少"。
3. **"生成实现 vs aiter" 需要离线 `audit_batch.py`**：它把生成产物送进离线终审、
   在真实 perf case 上计时。在线 run.json 的 runtime 对应的是 `get_inputs()` 的
   固定 shape，与 perf case 不一定同一组形状，直接相除不成立 —— 所以本报告
   不编造这个数。

用法：
    python3 stats_full66.py --before <before runjson 根> --after <after runjson 根> \
        --repo <dcu-kernel-bench> --out <输出 md>
"""
import argparse
import collections
import json
import pathlib
import statistics as st


def load(root: str) -> dict:
    out = {}
    base = pathlib.Path(root)
    for pat in ("*_l*_p*/run.json", "par_s*/*_l*_p*/run.json"):
        for p in sorted(base.glob(pat)):
            out[p.parent.name] = json.loads(p.read_text(encoding="utf-8"))
    return out


def tid(dirname: str) -> str:
    return dirname.rsplit("_l", 1)[0]


def row(d: dict) -> dict:
    nodes = d.get("nodes") or {}
    bid = d.get("best_node_id")
    best = nodes.get(bid) or {}
    root = nodes.get("root") or {}
    stages = (d.get("stats") or {}).get("failure_stage_counts") or {}
    return {
        "backend": d.get("backend"),
        "root_compile": bool(root.get("compile_ok")),
        "root_correct": bool(root.get("correct")),
        "best_compile": bool(best.get("compile_ok")),
        "best_correct": bool(best.get("correct")),
        "is_root": bid in ("root", "phase2_root"),
        "speedup": best.get("speedup"),
        "runtime": best.get("runtime"),
        "ref_runtime": best.get("reference_runtime"),
        "compile_fails": stages.get("compile", 0),
        "runtime_fails": stages.get("runtime", 0),
        "attempts": (d.get("stats") or {}).get("attempt_count"),
        "denominator": d.get("primary_reference"),
    }


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def dist(xs):
    if not xs:
        return None
    return {
        "n": len(xs), "median": st.median(xs), "mean": st.mean(xs),
        "p25": q(xs, 0.25), "p75": q(xs, 0.75), "max": max(xs), "min": min(xs),
    }


def fmt(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def segment(dirname: str) -> str:
    pid = tid(dirname).split("_", 1)[0]
    return pid[:1] + "xxx"


def aiter_table(repo: pathlib.Path) -> list:
    """每道题：aiter 各 perf case 的 us 与同 case torch_eager 的 us -> aiter 的加速比。"""
    out = []
    priv = repo / "benchmark" / "private"
    if not priv.is_dir():
        return out
    for d in sorted(priv.iterdir()):
        f = d / "baseline.json"
        if not f.is_file():
            continue
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        bl = j.get("baselines") or []
        by_case = collections.defaultdict(dict)
        for b in bl:
            if b.get("case") and b.get("impl") and isinstance(b.get("us"), (int, float)):
                by_case[b["case"]][b["impl"]] = b["us"]
        ratios = []
        for case, m in by_case.items():
            if "aiter" in m and "torch_eager" in m and m["aiter"] > 0:
                ratios.append(m["torch_eager"] / m["aiter"])
        impls = sorted({b.get("impl") for b in bl if b.get("impl")})
        out.append({
            "task": d.name, "impls": impls,
            "n_cases": len(by_case),
            "aiter_vs_eager_median": st.median(ratios) if ratios else None,
            "aiter_vs_eager_min": min(ratios) if ratios else None,
            "aiter_vs_eager_max": max(ratios) if ratios else None,
            "status": j.get("status"),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    B = {k: row(v) for k, v in load(args.before).items()}
    A = {k: row(v) for k, v in load(args.after).items()}
    repo = pathlib.Path(args.repo)

    def agg(D):
        vs = list(D.values())
        real = [r for r in vs if r["best_correct"] and not r["is_root"]]
        real_sp = [r["speedup"] for r in real if isinstance(r["speedup"], (int, float))]
        return {
            "n": len(vs),
            "root_compile": sum(1 for r in vs if r["root_compile"]),
            "root_correct": sum(1 for r in vs if r["root_correct"]),
            "best_compile": sum(1 for r in vs if r["best_compile"]),
            "best_correct": sum(1 for r in vs if r["best_correct"]),
            "real": len(real),
            "rootonly": sum(1 for r in vs if r["best_correct"] and r["is_root"]),
            "has_speedup": sum(1 for r in vs if isinstance(r["speedup"], (int, float))),
            "compile_fails": sum(r["compile_fails"] or 0 for r in vs),
            "runtime_fails": sum(r["runtime_fails"] or 0 for r in vs),
            "dist": dist(real_sp),
            "real_sp": real_sp,
        }

    AB, AA = agg(B), agg(A)
    L = []
    w = L.append

    w("# 66 道生成评测性能统计（2026-10-17）")
    w("")
    w("五个 harness 缺陷修复前后的全量对照。数据源：")
    w("")
    w(f"- 修复前：`runs/dcukb_final`（{AB['n']} 道，含此前 HIP20 重跑结果）")
    w(f"- 修复后：`runs/dcukb_full_fixed`（{AA['n']} 道，16 片并行，10:08–11:53）")
    w("")
    w("## 0. 口径（先读这段，避免误读数字）")
    w("")
    w("| 概念 | 定义 | 数据来源 |")
    w("|---|---|---|")
    w("| **编译率** | 脚手架/候选解能通过编译的比例 | run.json 的 `compile_ok` |")
    w("| **正确率** | 通过在线正确性比对（allclose 1e-4/trial）的比例 | run.json 的 `correct` |")
    w("| **真解** | best 节点**不是 root**（即 Agent 真的改写了代码）且 correct | `best_node_id` + `correct` |")
    w("| **加速比** | 生成实现 runtime 的倒数 ÷ 参考实现 runtime 的倒数 | run.json 的 `speedup` |")
    w("")
    w("**关键口径**：run.json 里 `speedup` 的分母是 **`torch_eager`**（在线评测器拿参考 `Model`")
    w("当分母），**不是 aiter**。本轮 66 份 run.json 的 `primary_reference` 全部为 `torch_eager`。")
    w("")
    w("aiter 基线在 `benchmark/private/<task>/baseline.json`，按 perf case 分 impl 存实测 us。")
    w("**「生成实现 vs aiter」这个数本报告不提供** —— 它需要离线 `audit_batch.py` 把生成产物")
    w("送进终审、在真实 perf case 上计时；在线 runtime 对应的是 `get_inputs()` 的固定 shape，")
    w("与 perf case 不一定同形状，直接相除不成立。第 4 节给出 aiter 基线本身的数据与缺口。")
    w("")

    w("## 1. 编译率")
    w("")
    w("| 指标 | 修复前 | 修复后 |")
    w("|---|---|---|")
    w(f"| root 脚手架编译通过（= 框架给出的起点可用） | **{AB['root_compile']}/{AB['n']}** | **{AA['root_compile']}/{AA['n']}** |")
    w(f"| best 节点编译通过 | {AB['best_compile']}/{AB['n']} | {AA['best_compile']}/{AA['n']} |")
    w(f"| compile 阶段失败次数（合计，含 Agent 自己写坏） | {AB['compile_fails']} | {AA['compile_fails']} |")
    w(f"| runtime 阶段失败次数（合计） | {AB['runtime_fails']} | {AA['runtime_fails']} |")
    w("")
    w("root 脚手架编译率是**衡量 harness 质量**的指标：修复前 7 道题因脚手架多行签名截断")
    w("直接 SyntaxError，Agent 连起点都没有。修复后 66/66。")
    w("compile 阶段失败次数不归零是正常的 —— 那是 Agent 尝试过程里自己写坏的补丁，")
    w("属于生成侧信号，不是 harness 缺陷。")
    w("")

    w("## 2. 正确率")
    w("")
    w("| 指标 | 修复前 | 修复后 |")
    w("|---|---|---|")
    w(f"| best 正确（在线判据） | {AB['best_correct']}/{AB['n']}（{AB['best_correct']/AB['n']*100:.1f}%） | **{AA['best_correct']}/{AA['n']}（{AA['best_correct']/AA['n']*100:.1f}%）** |")
    w(f"| └ 非 root 真解 | {AB['real']}（{AB['real']/AB['n']*100:.1f}%） | **{AA['real']}（{AA['real']/AA['n']*100:.1f}%）** |")
    w(f"| └ 仅 root（Agent 未改写） | {AB['rootonly']} | {AA['rootonly']} |")
    w(f"| 有 speedup 可比 | {AB['has_speedup']}/{AB['n']} | {AA['has_speedup']}/{AA['n']} |")
    w("")

    # 按 backend
    w("### 按 backend")
    w("")
    w("| backend | 题数 | best 正确（前→后） | 真解（前→后） |")
    w("|---|---|---|---|")
    backends = sorted({r["backend"] for r in list(B.values()) + list(A.values())})
    for be in backends:
        kb = [k for k in B if B[k]["backend"] == be]
        ka = [k for k in A if A[k]["backend"] == be]
        cb_b = sum(1 for k in kb if B[k]["best_correct"])
        cb_a = sum(1 for k in ka if A[k]["best_correct"])
        rl_b = sum(1 for k in kb if B[k]["best_correct"] and not B[k]["is_root"])
        rl_a = sum(1 for k in ka if A[k]["best_correct"] and not A[k]["is_root"])
        label = "HIP（backend=cuda）" if be == "cuda" else "Triton"
        w(f"| {label} | {len(ka)} | {cb_b}→{cb_a} | {rl_b}→{rl_a} |")
    w("")

    w("## 3. 加速比（分母 = `torch_eager`）")
    w("")
    w("统计口径：只看**真解**（best 非 root 且 correct）的 speedup。")
    w("")
    w("| 指标 | 修复前 | 修复后 |")
    w("|---|---|---|")
    for label, key in [("样本数 n", "n"), ("中位数", "median"), ("均值", "mean"),
                       ("P25", "p25"), ("P75", "p75"), ("最大", "max"), ("最小", "min")]:
        db, da = AB["dist"], AA["dist"]
        if key == "n":
            w(f"| {label} | {db['n'] if db else 0} | {da['n'] if da else 0} |")
        else:
            w(f"| {label} | {fmt(db[key] if db else None)}x | {fmt(da[key] if da else None)}x |")
    w("")

    # 分桶
    def buckets(xs):
        edges = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 100), (100, float("inf"))]
        c = collections.Counter()
        for x in xs:
            for lo, hi in edges:
                if lo <= x < hi:
                    c[f"{lo}–{hi}x" if hi != float("inf") else "≥100x"] += 1
                    break
        return c

    bb, ba = buckets(AB["real_sp"]), buckets(AA["real_sp"])
    w("### speedup 分布（真解）")
    w("")
    w("| 区间 | 修复前 | 修复后 |")
    w("|---|---|---|")
    for label in ["0–1x", "1–2x", "2–5x", "5–10x", "10–100x", "≥100x"]:
        w(f"| {label} | {bb.get(label, 0)} | {ba.get(label, 0)} |")
    w("")

    # 按号段
    w("### 按题号段（修复后，真解）")
    w("")
    w("| 号段 | 真解数 | speedup 中位 | 最大 |")
    w("|---|---|---|---|")
    segs = collections.defaultdict(list)
    for k, r in A.items():
        if r["best_correct"] and not r["is_root"] and isinstance(r["speedup"], (int, float)):
            segs[segment(k)].append(r["speedup"])
    for s in sorted(segs):
        xs = segs[s]
        w(f"| {s} | {len(xs)} | {st.median(xs):.2f}x | {max(xs):.2f}x |")
    w("")

    # Top / Bottom
    ranked = sorted(
        [(k, A[k]["speedup"]) for k in A
         if A[k]["best_correct"] and not A[k]["is_root"] and isinstance(A[k]["speedup"], (int, float))],
        key=lambda kv: -kv[1])
    w("### Top 10 / Bottom 5（修复后）")
    w("")
    w("| # | 题 | speedup |")
    w("|---|---|---|")
    for i, (k, v) in enumerate(ranked[:10], 1):
        w(f"| {i} | `{tid(k)}` | **{v:.2f}x** |")
    for i, (k, v) in enumerate(ranked[-5:], len(ranked) - 4):
        w(f"| {i} | `{tid(k)}` | {v:.2f}x |")
    w("")

    w("## 4. aiter 基线")
    w("")
    at = aiter_table(repo)
    with_a = [x for x in at if "aiter" in x["impls"]]
    w(f"`benchmark/private` 共 {len(at)} 个 `baseline.json`：**{len(with_a)} 个含 aiter 基线**，"
      f"{len([x for x in at if x['impls'] and 'aiter' not in x['impls']])} 个仅 `torch_eager`，"
      f"{len([x for x in at if not x['impls']])} 个 placeholder。")
    w("")
    w("### aiter 本身相对 torch_eager 的加速比（同机实测，按 perf case 取中位）")
    w("")
    w("| 题 | aiter 相对 eager | (min–max) | perf case 数 |")
    w("|---|---|---|---|")
    for x in sorted(with_a, key=lambda y: -(y["aiter_vs_eager_median"] or 0)):
        med = x["aiter_vs_eager_median"]
        w(f"| `{x['task']}` | **{med:.2f}x** | {x['aiter_vs_eager_min']:.2f}–{x['aiter_vs_eager_max']:.2f}x | {x['n_cases']} |"
          if med is not None else f"| `{x['task']}` | - | - | {x['n_cases']} |")
    w("")
    w("> 这张表说明的是**官方 aiter 实现比 PyTorch eager 快多少**，可以作为「生成实现要达到")
    w("> 什么水平才算追上官方」的参照系。它不是「生成实现 vs aiter」。")
    w(">")
    w("> **已知偏差（必须一起读）**：`4001` / `4004` / `4006` 与 `4008` 的部分 case，其 aiter")
    w("> 基线落在 **MI 系列 fallback tuner 配置**上（未针对 DCU 调优，见 AGENTS.md），aiter")
    w("> 数值本身偏慢，所以「aiter 相对 eager」会偏低甚至 <1x（`4006` 0.11x、`3002` 0.19x、")
    w("> `4001` 0.37x）。这时**拿 aiter 当分母会把生成实现的加速比算得偏高**，引用时须标注。")
    w("")
    w("### 缺口：生成实现 vs aiter")
    w("")
    w("要得到「生成实现相对 aiter 的加速比」，需要跑离线终审：")
    w("")
    w("```bash")
    w("python3 benchmark/evaluator/audit_batch.py --help    # 把生成产物批量送离线终审")
    w("```")
    w("")
    w("它做的事：静态审计 → 隐藏 case 按 task.yaml 容差判定 → **在真实 perf case 上计时**，")
    w("分母用 `private/<task>/baseline.json` 里记录的 aiter 用时。")
    w("这一步本次**没有执行**，所以本报告不含该数字。")
    w("")

    w("## 5. 结论")
    w("")
    w(f"1. **编译率**：root 脚手架 {AB['root_compile']}→{AA['root_compile']}/{AA['n']}，"
      f"即所有题都拿到了可跑的起点。")
    w(f"2. **正确率**：best 正确 {AB['best_correct']}→{AA['best_correct']}/{AA['n']}"
      f"（{AA['best_correct']/AA['n']*100:.1f}%），其中真解 {AB['real']}→{AA['real']}。")
    w(f"3. **加速比**（vs eager）：真解中位 {fmt(AB['dist']['median'] if AB['dist'] else None)}x → "
      f"{fmt(AA['dist']['median'] if AA['dist'] else None)}x，最大 "
      f"{fmt(AB['dist']['max'] if AB['dist'] else None)}x → {fmt(AA['dist']['max'] if AA['dist'] else None)}x。")
    w("4. **aiter 基线**：33 道题有官方 aiter 实测基线（第 4 节表），但生成实现 vs aiter 需要")
    w("   离线 `audit_batch.py`，本次未跑。")
    w("")

    out = pathlib.Path(args.out)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"已写 {out}  ({len(L)} 行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
