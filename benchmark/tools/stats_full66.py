"""生成 66 道全量生成评测的**双基线**性能统计文档（markdown）。

三套数字，口径不同，**不可混比**：

| 套 | 来源 | 正确性判据 | 加速比分母 |
|---|---|---|---|
| 在线 | run.json（PyramidKernel / KernelBench 口径） | 输入 cast fp32 + allclose(atol=rtol=1e-4) | `torch_eager`（框架没有"外部基线"模式） |
| 离线-双基线 | `offline_audit.json`（`audit_batch.py`） | 隐藏 case + `task.yaml` 分档容差 | `aiter` 官方实现 **与** `torch_eager`，各算一遍 |

所以「对 aiter 的加速比」只可能来自离线那一套；在线 run.json 的 `speedup` 分母
恒为 `torch_eager`（本轮 66 份已核对，`primary_reference` 全是 torch_eager）。

用法：
    python3 stats_full66.py --before <before runjson 根> --after <after runjson 根> \
        --offline-audit <offline_audit.json> --repo <dcu-kernel-bench> --out <输出 md>
"""
import argparse
import collections
import json
import pathlib
import statistics as st

# 20 道 HIP track 题（manifest backend=cuda）的基线口径分三档，
# 依据 benchmark/tools/hipdnn_baseline_probe.md：
#   - conv 族：probe 实测 torch_eager 就走 MIOpen（vendor 库），故 eager 分母不弱
#   - probe 明确列为"无 MIOpen 路径、纯 ATen"的算子
#   - probe 未覆盖的（deform attention / norm 族），不作断言
CONV_FAMILY_MIOPEN = {"5001_conv2d_fwd", "5002_conv2d_bwd",
                      "5008_conv_bias_act", "5009_conv_depthtospace"}
PURE_ATEN = {
    "5003_deform_conv2d_fwd", "5004_deform_conv2d_bwd",
    "5005_spconv_fprop", "5006_spconv_bwd", "5007_spconv_indice_pairs",
    "6001_fused_adamw", "8001_cumsum", "8002_gather", "8003_scatter_nd",
}

# 有 aiter 基线、但离线终审跑不出结果的题（2026-10-17 实测，定向重试复现）。
AUDIT_FAILURES = {
    "1016_lean_atten_paged": "exit 1，未产出 JSON 报告",
    "2004_fused_qk_concat": "exit -11（SIGSEGV），审计过程崩溃",
}

# 离线终审里「静态审计判负」的题各自命中了哪些禁用算子。
# offline_audit.json 只记 static_audit_passed 布尔值，具体符号是用同一份
# benchmark/evaluator/static_audit.py 对 best_code.py 复跑得到的（2026-10-17）。
STATIC_AUDIT_HITS = {
    "1020_mla_decode_rope": ["torch.matmul", "torch.softmax"],
    "1022_pa_prefill": ["torch.einsum", "torch.softmax"],
    "1026_sage_attention_qk_int8_per_block": ["torch.matmul", "torch.softmax"],
    "3009_routing": ["torch.matmul", "torch.sigmoid"],
    "4004_batched_gemm_bf16": ["torch.matmul"],
    "4008_gemm_a16w4": ["torch.matmul"],
}


def load(root: str) -> dict:
    out, base = {}, pathlib.Path(root)
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
        "compile_fails": stages.get("compile", 0),
        "runtime_fails": stages.get("runtime", 0),
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
    return {"n": len(xs), "median": st.median(xs), "mean": st.mean(xs),
            "p25": q(xs, 0.25), "p75": q(xs, 0.75), "max": max(xs), "min": min(xs)}


def fmt(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def audit_case_cache(cache_dir: pathlib.Path, task: str) -> dict:
    """读 audit_batch.py 的单题缓存（含 correctness_failures 明细）。"""
    p = cache_dir / f"{task}.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def aiter_baseline_table(repo: pathlib.Path) -> dict:
    """private/*/baseline.json：aiter 相对 torch_eager 的加速比（同 perf case）。"""
    out = {}
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
        by_case = collections.defaultdict(dict)
        for b in j.get("baselines") or []:
            if b.get("case") and b.get("impl") and isinstance(b.get("us"), (int, float)):
                by_case[b["case"]][b["impl"]] = b["us"]
        ratios = [m["torch_eager"] / m["aiter"] for m in by_case.values()
                  if "aiter" in m and "torch_eager" in m and m["aiter"] > 0]
        impls = sorted({b.get("impl") for b in j.get("baselines") or [] if b.get("impl")})
        out[d.name] = {"impls": impls, "n_cases": len(by_case),
                       "median": st.median(ratios) if ratios else None,
                       "min": min(ratios) if ratios else None,
                       "max": max(ratios) if ratios else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--offline-audit", default="", help="audit_batch.py 产出的 offline_audit.json")
    ap.add_argument("--model", default="gpt-6-astra",
                    help="生成模型名。注意 run.json **不记录**模型名，只能从 llm.env 的 "
                         "OPENAI_MODEL 取，故此处由调用方传入。")
    ap.add_argument("--audit-cache", default=".dcu_runs/offline_audit/offline_audit",
                    help="audit_batch.py 的单题缓存目录（读 correctness_failures 明细）")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", required=True)
    # 报告措辞：默认沿用"修复前 / 修复后"（harness 缺陷修复那次对比）；换实验时用
    # 这组参数改写标题、两列表头与数据源说明，避免数字对但标签错。
    ap.add_argument("--title", default="66 道生成评测性能统计（双基线，2026-10-17）")
    ap.add_argument("--context", default="五个 harness 缺陷修复后的全量 66 道结果。")
    ap.add_argument("--before-label", default="修复前")
    ap.add_argument("--after-label", default="修复后")
    ap.add_argument("--before-path", default="runs/dcukb_final")
    ap.add_argument("--after-path", default="runs/dcukb_full_fixed")
    args = ap.parse_args()
    BL, AL = args.before_label, args.after_label

    B = {k: row(v) for k, v in load(args.before).items()}
    A = {k: row(v) for k, v in load(args.after).items()}
    repo = pathlib.Path(args.repo)
    cache_dir = pathlib.Path(args.audit_cache)
    audit = None
    if args.offline_audit and pathlib.Path(args.offline_audit).is_file():
        audit = json.loads(pathlib.Path(args.offline_audit).read_text(encoding="utf-8"))
    base_tab = aiter_baseline_table(repo)

    def agg(D):
        vs = list(D.values())
        real = [r for r in vs if r["best_correct"] and not r["is_root"]]
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
            "dist": dist([r["speedup"] for r in real if isinstance(r["speedup"], (int, float))]),
        }

    AB, AA = agg(B), agg(A)
    hip_ids = {tid(k) for k, r in A.items() if r["backend"] == "cuda"}
    audited_ids = {r["task"] for r in audit["rows"]} if audit else set()
    # "无 aiter 基线"以 audit_batch 的 skipped 为准（它按 baseline.json 的 impls 判）；
    # 有 aiter 但本次没审出结果的，单独列出来，不能混进"无 aiter"。
    skipped_ids = ({s["task"] for s in audit.get("skipped", [])} if audit else set())
    all_ids = {tid(k) for k in A}
    no_aiter = sorted(skipped_ids) if audit else sorted(all_ids)
    unaudited = sorted(all_ids - skipped_ids - audited_ids) if audit else []

    L = []
    w = L.append

    w(f"# {args.title}")
    w("")
    w(f"{args.context}数据源：")
    w("")
    w(f"- 生成模型：`{args.model}`（取自 `llm.env` 的 `OPENAI_MODEL`；**run.json 未记录模型名**）")
    w(f"- {BL}在线：`{args.before_path}`（{AB['n']} 道）")
    w(f"- {AL}在线：`{args.after_path}`（{AA['n']} 道，16 片并行）")
    if audit:
        w(f"- {AL}离线终审：`offline_audit.json`（审 {audit['aggregates']['audited']} 道有 aiter 基线的题）")
    w("")
    w("## 0. 口径：三套数字不可混比")
    w("")
    w("| 套 | 来源 | 正确性判据 | 加速比分母 |")
    w("|---|---|---|---|")
    w("| **在线** | `run.json`（KernelBench/PyramidKernel 口径） | 输入 cast fp32 + `allclose(atol=rtol=1e-4)` | **`torch_eager`**（框架没有\"外部基线\"模式） |")
    w("| **离线·aiter** | `offline_audit.json`（`audit_batch.py`） | 隐藏 case + `task.yaml` **分档容差** | `aiter` 官方实现 |")
    w("| **离线·eager** | 同上 | 同上 | `torch_eager` |")
    w("")
    w("**因此**：① 在线 speedup **不是**对 aiter 的加速比（本轮 66 份 `primary_reference` 全为")
    w("`torch_eager`）；② 离线比在线**严得多**（隐藏 case + 分档容差，而在线是单一 shape 的")
    w("1e-4 allclose），所以同一题的离线正确性与在线正确性不是一回事。")
    w("")

    w("## 1. 编译率（在线）")
    w("")
    w(f"| 指标 | {BL} | {AL} |")
    w("|---|---|---|")
    w(f"| root 脚手架编译通过（= 框架给出的起点可用） | **{AB['root_compile']}/{AB['n']}** | **{AA['root_compile']}/{AA['n']}** |")
    w(f"| best 节点编译通过 | {AB['best_compile']}/{AB['n']} | {AA['best_compile']}/{AA['n']} |")
    w(f"| compile 阶段失败次数（合计，含 Agent 自己写坏） | {AB['compile_fails']} | {AA['compile_fails']} |")
    w(f"| runtime 阶段失败次数（合计） | {AB['runtime_fails']} | {AA['runtime_fails']} |")
    w("")
    w("root 脚手架编译率衡量 **harness 质量**：脚手架坏掉的题 Agent 连起点都没有。")
    w("compile 阶段失败次数不归零是")
    w("正常的 —— 那是 Agent 尝试里自己写坏的补丁，属生成侧信号。")
    w("")

    w("## 2. 正确率")
    w("")
    w("### 2.1 在线（KernelBench 口径）")
    w("")
    w(f"| 指标 | {BL} | {AL} |")
    w("|---|---|---|")
    w(f"| best 正确 | {AB['best_correct']}/{AB['n']}（{AB['best_correct']/AB['n']*100:.1f}%） | **{AA['best_correct']}/{AA['n']}（{AA['best_correct']/AA['n']*100:.1f}%）** |")
    w(f"| └ 非 root 真解 | {AB['real']}（{AB['real']/AB['n']*100:.1f}%） | **{AA['real']}（{AA['real']/AA['n']*100:.1f}%）** |")
    w(f"| └ 仅 root（Agent 未改写） | {AB['rootonly']} | {AA['rootonly']} |")
    w(f"| 有 speedup 可比 | {AB['has_speedup']}/{AB['n']} | {AA['has_speedup']}/{AA['n']} |")
    w("")
    w(f"按 backend（{AL}）：")
    w("")
    w("| backend | 题数 | best 正确（前→后） | 真解（前→后） |")
    w("|---|---|---|---|")
    for be, label in (("cuda", "HIP（backend=cuda）"), ("triton", "Triton")):
        # 两列必须用**同一批**题：按 after 的 backend 归属取集合，再回查 before
        # 里同名目录的结果。否则某个题的 backend 在两轮间变了，就会打印出
        # 「20→19，总数 19」这种自相矛盾的行。
        ka = [k for k in A if A[k]["backend"] == be]
        kb = [k for k in ka if k in B]
        w(f"| {label} | {len(ka)} | "
          f"{sum(1 for k in kb if B[k]['best_correct'])}→{sum(1 for k in ka if A[k]['best_correct'])} | "
          f"{sum(1 for k in kb if B[k]['best_correct'] and not B[k]['is_root'])}→"
          f"{sum(1 for k in ka if A[k]['best_correct'] and not A[k]['is_root'])} |")
    w("")

    if audit:
        ag = audit["aggregates"]
        rows_a = audit["rows"]

        def cache_of(task):
            return audit_case_cache(cache_dir, task)

        # 注意：offline_audit.json 的 rows 里**没有** static_audit_passed，
        # 它只出现在单题缓存里。所以分类必须读缓存，不能从 rows 判。
        static_bad = [r for r in rows_a
                      if cache_of(r["task"]).get("static_audit_passed") is False]
        # 提交实现在隐藏 case 上崩溃/编译失败（终审拿不到 JSON 报告）：这是**失败**，
        # 不是「没审到」。判据是 rows 里带 failure_reason（audit_batch 现在会写）。
        crashed = [r for r in rows_a if r.get("failure_reason")]
        corr_bad = [r for r in rows_a
                    if cache_of(r["task"]).get("static_audit_passed") is not False
                    and not r["correctness_passed"]
                    and not r.get("failure_reason")
                    and cache_of(r["task"]).get("correctness_case_count")]
        passed = [r for r in rows_a if r["correctness_passed"]]
        ids_bad = ({id(r) for r in static_bad} | {id(r) for r in corr_bad}
                   | {id(r) for r in crashed} | {id(r) for r in passed})
        other = [r for r in rows_a if id(r) not in ids_bad]
        w("### 2.2 离线终审（静态审计 + 隐藏 case + `task.yaml` 分档容差）")
        w("")
        # 离线正确性不依赖基线：没有 aiter 分母的题也必须过隐藏 case，所以这里统计
        # 的是**全量**审计结果；能不能给双基线是另一件事（看 measurement_impls）。
        headline = audit.get("headline_impl") or "aiter"
        has_impl = [r for r in rows_a if headline in (r.get("measurement_impls") or [])]
        w(f"**{ag['audited']} 道全审**：通过 {len(passed)} 道"
          f"（{len(passed)/ag['audited']*100:.0f}%）。其中 {len(has_impl)} 道有 "
          f"`{headline}` 分母、可给双基线；其余 {ag['audited'] - len(has_impl)} 道只能给 "
          f"`torch_eager` 口径。")
        w("")
        w("未通过的题**不是同一类原因**，性质完全不同：")
        w("")
        w("| 未通过原因 | 题数 | 说明 |")
        w("|---|---|---|")
        w(f"| **静态审计判负**（未进入正确性环节） | **{len(static_bad)}** | "
          f"实现里直接调用被禁的 ATen 算子完成核心计算，违反「禁止绕过」铁律 |")
        w(f"| **正确性不达标**（隐藏 case 对不上） | **{len(corr_bad)}** | "
          f"静态审计通过，但边界 shape 处理不到位 |")
        if crashed:
            w(f"| **提交实现在隐藏 case 上跑不起来** | **{len(crashed)}** | "
              f"编译失败 / 共享内存超限 / 后端 pass crash，见下方 C |")
        if other:
            w(f"| 审计未产出结果（无明细） | **{len(other)}** | 见下方 C |")
        w("")
        if static_bad:
            w("#### A. 静态审计判负")
            w("")
            w("| 题 | 命中的禁用算子 |")
            w("|---|---|")
            for r in static_bad:
                hits = STATIC_AUDIT_HITS.get(r["task"])
                label = "、".join(f"`{h}`" for h in hits) if hits else "（未复跑取证）"
                w(f"| `{r['task']}` | {label} |")
            w("")
            w("这些题在**在线**被判为正确 —— 在线评测不执行禁用列表（题面 forbidden 在线上只是")
            w("软约束，硬审计在离线）。所以它们不是「性能不好」，而是**实现方式不合规**。")
            w("")
        if corr_bad:
            w("#### B. 正确性不达标")
            w("")
            w("| 题 | 失败 case |")
            w("|---|---|")
            for r in corr_bad:
                cache = cache_of(r["task"])
                cases = cache.get("correctness_failures") or []
                n = cache.get("correctness_case_count")
                label = "、".join(f"`{c}`" for c in cases) if cases else "（未取到明细）"
                w(f"| `{r['task']}` | {label} （{len(cases)}/{n if n else '?'}）|")
            w("")
            w("失败 case 全部是 `hidden_` 前缀的隐藏 case，形态为**非 2 次幂、超长/超宽、")
            w("极大规模、fp8 动态量化、零行**等边界；在线只跑 `get_inputs()` 的单一固定")
            w("shape，这些边界不会出现，所以在线全对、离线翻车。**这 "
              f"{len(corr_bad)} 道才是真正能归因于「离线判据更严」的。**")
            w("")
        if crashed:
            w("#### C. 提交实现在隐藏 case 上跑不起来")
            w("")
            w("| 题 | 崩溃原因 |")
            w("|---|---|")
            for r in crashed:
                tail = " ".join((r.get("failure_tail") or "").split())
                tail = tail.replace("|", "\\|")
                w(f"| `{r['task']}` | `{tail[-220:]}` |" if tail else
                  f"| `{r['task']}` | {r.get('failure_reason')} |")
            w("")
            w("这类**不算「离线判据更严」**，而是实现根本没在隐藏 shape 上跑通：在线只跑")
            w("`get_inputs()` 的单一固定 shape，隐藏 case 的 `head_size=256`、`head_size=96`")
            w("（非 2 次幂）、`k=131/n=53/m=1` 这类边界直接把它打崩。")
            w("")
            w("归因提示：`RuntimeError: PassManager::run failed` 发生在 DCU triton 后端的")
            w("`make_ttgir` pass 里（不是提交代码里），同一签名也出现在 1019 的 aiter 基线采集，")
            w("因此这一条**既可能是提交实现生成了该后端处理不了的 IR，也可能是后端限制**；")
            w("另外两条（共享内存超限、`tl.arange` 非 2 次幂）责任明确在实现侧。")
            w("")
        if other:
            w("#### D. 审计未产出结果（无明细）")
            w("")
            for r in other:
                c = cache_of(r["task"])
                w(f"- `{r['task']}`：`static_audit_passed={c.get('static_audit_passed')}`、"
                  f"`exit_code={c.get('exit_code')}`")
            w("")
        w("> 两套判据不互相否定：在线回答「能不能跑通并对上参考」，离线回答「换一批")
        w("> shape/边界还对不对、以及实现是否合规」。引用时必须说明用的是哪一套。")
        w("")

    w("## 3. 在线加速比（分母 = `torch_eager`）")
    w("")
    w("统计口径：只看**真解**（best 非 root 且 correct）。")
    w("")
    w(f"| 指标 | {BL} | {AL} |")
    w("|---|---|---|")
    for label, key in [("样本数 n", "n"), ("中位数", "median"), ("均值", "mean"),
                       ("P25", "p25"), ("P75", "p75"), ("最大", "max"), ("最小", "min")]:
        db, da = AB["dist"], AA["dist"]
        if key == "n":
            w(f"| {label} | {db['n'] if db else 0} | {da['n'] if da else 0} |")
        else:
            w(f"| {label} | {fmt(db[key] if db else None)}x | {fmt(da[key] if da else None)}x |")
    w("")

    def buckets(xs):
        edges = [(0, 1, "0–1x"), (1, 2, "1–2x"), (2, 5, "2–5x"),
                 (5, 10, "5–10x"), (10, 100, "10–100x"), (100, float("inf"), "≥100x")]
        c = collections.Counter()
        for x in xs:
            for lo, hi, name in edges:
                if lo <= x < hi:
                    c[name] += 1
                    break
        return c

    bb = buckets([r["speedup"] for r in B.values()
                  if r["best_correct"] and not r["is_root"] and isinstance(r["speedup"], (int, float))])
    ba = buckets([r["speedup"] for r in A.values()
                  if r["best_correct"] and not r["is_root"] and isinstance(r["speedup"], (int, float))])
    w(f"| 区间 | {BL} | {AL} |")
    w("|---|---|---|")
    for label in ["0–1x", "1–2x", "2–5x", "5–10x", "10–100x", "≥100x"]:
        w(f"| {label} | {bb.get(label, 0)} | {ba.get(label, 0)} |")
    w("")

    ranked = sorted(
        [(k, A[k]["speedup"]) for k in A
         if A[k]["best_correct"] and not A[k]["is_root"] and isinstance(A[k]["speedup"], (int, float))],
        key=lambda kv: -kv[1])
    w(f"Top 10（{AL}，在线口径）：")
    w("")
    w("| # | 题 | 在线 speedup |")
    w("|---|---|---|")
    for i, (k, v) in enumerate(ranked[:10], 1):
        w(f"| {i} | `{tid(k)}` | **{v:.2f}x** |")
    w("")

    if audit:
        ag = audit["aggregates"]
        w("## 4. 双基线表（离线终审；生成实现 vs aiter **与** vs eager）")
        w("")
        w("分母来自 `benchmark/private/<task>/baseline.json` 的实测值，perf 形状与计时参数取")
        w("`private/<task>/perf_cases.json`。**同一行的两列口径一致**，可以横向比。")
        w("")
        w("| 题 | 离线正确性 | vs **aiter** | vs eager | 备注 |")
        w("|---|---|---|---|---|")
        for r in sorted(audit["rows"], key=lambda r: -(r.get("aiter_mean_speedup") or -1)):
            a = r.get("aiter_mean_speedup")
            e = r.get("torch_eager_mean_speedup")
            note = []
            if r.get("aiter_is_mi_fallback"):
                note.append("MI 兜底†")
            if not r["correctness_passed"]:
                note.append("离线正确性未过")
            if tid(r["task"]) in hip_ids:
                note.append("HIP")
            w(f"| `{r['task']}` | {'✓' if r['correctness_passed'] else '✗'} | "
              f"{('**' + fmt(a) + 'x**') if a is not None else '-'} | "
              f"{(fmt(e) + 'x') if e is not None else '-'} | {'；'.join(note)} |")
        w("")
        w(f"汇总（aiter 有测值的 {ag['aiter_measured']} 道）：**aiter 中位 "
          f"{ag['aiter_median_speedup']:.2f}x、均值 {ag['aiter_mean_speedup']:.2f}x、"
          f"最大 {ag['aiter_max_speedup']:.2f}x**。")
        w("")
        w(f"† **MI 兜底基线 {ag['mi_fallback_count']} 题**：该题的 aiter 基线落在 **MI 系列 fallback")
        w("tuner** 配置上（未针对 DCU 调优，见 AGENTS.md）—— 也就是说这个 aiter 数值**不等于")
        w("aiter 在 DCU 上能达到的最好成绩**，因此由它导出的加速比不能直接当「与官方实现的距离」。")
        w("方向因题而异，必须结合下一小节一起看（例：`1014` 的 aiter 仍比 eager 快 28x，而生成的")
        w("解比 aiter 慢约 13x；`4006` 的 aiter 反而比 eager 慢 8x，于是「vs aiter」被抬到 10.49x）。")
        w("这 "
          f"{ag['mi_fallback_count']} 道是：")
        w("、".join(f"`{r['task']}`" for r in audit["rows"] if r.get("aiter_is_mi_fallback")))
        w("")
        with_a = {k: v for k, v in base_tab.items() if "aiter" in v["impls"] and v["median"] is not None}
        if with_a:
            w("### 分母质量对照：aiter 本身相对 eager（按 perf case 取中位）")
            w("")
            w("| 题 | aiter 相对 eager | (min–max) |")
            w("|---|---|---|")
            for k, v in sorted(with_a.items(), key=lambda kv: -kv[1]["median"])[:6]:
                w(f"| `{k}` | **{v['median']:.2f}x** | {v['min']:.2f}–{v['max']:.2f}x |")
            w("| … | | |")
            for k, v in sorted(with_a.items(), key=lambda kv: kv[1]["median"])[:4]:
                w(f"| `{k}` | **{v['median']:.2f}x** | {v['min']:.2f}–{v['max']:.2f}x |")
            w("")
            w("这张表回答的是「该题的 aiter 分母本身强不强」：>1x 说明 aiter 比 eager 快，拿它当")
            w("分母更难；<1x（`4006` 0.11x、`3002` 0.19x、`4001` 0.37x）说明该 aiter 配置比 PyTorch")
            w("还慢，拿它当分母会把生成实现算得偏高。")
            w("")
            w("> **不要用上表的两个加速比相除来互推**：双基线表里的 vs-aiter / vs-eager 是逐 case")
            w("> 求比再聚合（均值或中位），而这张表是逐 case 的 `eager_us/aiter_us` 再取中位；")
            w("> 各 case 强弱差别大时两者不满足简单乘除关系（`1014` 就是一例：逐 case 比值 28.2x 与")
            w("> 56.0x，均值之比却是 45.6）。要互推请回到单题缓存 `offline_audit/<task>.json` 的")
            w("> `by_impl.*.cases` 逐 case 核对。")
            w("")

    w(f"## 5. 只有 `torch_eager` 基线的题（含 HIP 那 {len(hip_ids)} 道）")
    w("")
    w(f"66 道里有 **{len(no_aiter)} 道没有 aiter 基线**（`baseline.json` 的 impls 只有")
    w(f"`torch_eager`），因此只能给 `torch_eager` 口径。其中 **{len(hip_ids)} 道是 HIP track")
    w("（`backend=cuda`）**，全部落在这一类。")
    if audit:
        # 这些题以前被 audit_batch 的「无 aiter 基线就跳过」直接丢掉了，等于离线零覆盖；
        # 现在全审，所以这里给出它们自己的隐藏 case 结果与 eager 口径加速比。
        by_task = {r["task"]: r for r in audit.get("rows") or []}
        be_of = {}
        for k, v in A.items():
            be_of.setdefault(tid(k), v["backend"])
        no_a = [t for t in sorted(no_aiter) if t in by_task]
        ok_no_a = [t for t in no_a if by_task[t]["correctness_passed"]]
        sp_no_a = [(t, by_task[t]["torch_eager_mean_speedup"]) for t in no_a
                   if isinstance(by_task[t].get("torch_eager_mean_speedup"), (int, float))]
        w("")
        w(f"**这 {len(no_a)} 道现在也过离线终审**（正确性不依赖基线）：隐藏 case 通过 "
          f"**{len(ok_no_a)}/{len(no_a)}**，其中 {len(sp_no_a)} 道能量出「生成实现 vs "
          f"`torch_eager`」的离线加速比。")
        w("")
        w("| 题 | backend | 离线正确性 | vs eager（离线） | 备注 |")
        w("|---|---|---|---|---|")
        for t, sp_e in sorted(sp_no_a, key=lambda kv: -kv[1]):
            w(f"| `{t}` | {be_of.get(t, '')} | ✓ | **{sp_e:.2f}x** |  |")
        for t in no_a:
            r = by_task[t]
            if r["correctness_passed"]:
                continue
            note = "离线正确性未过"
            if r.get("failure_reason"):
                tail = " ".join((r.get("failure_tail") or "").split())[-120:].replace("|", "\\|")
                note = f"跑不起来：`{tail}`" if tail else r["failure_reason"]
            spx = r.get("torch_eager_mean_speedup")
            w(f"| `{t}` | {be_of.get(t, '')} | ✗ | "
              f"{f'{spx:.2f}x' if isinstance(spx, (int, float)) else '-'} | {note} |")
        w("")
    if unaudited:
        w("")
        w(f"另有 **{len(unaudited)} 道有 aiter 基线但离线终审跑不出结果**（不是没有基线，也不要")
        w("据此推断它们性能差）：")
        w("")
        w("| 题 | 失败原因 |")
        w("|---|---|")
        for t in unaudited:
            w(f"| `{t}` | {AUDIT_FAILURES.get(t, '未记录')} |")
        w("")
        w("这几道属离线终审工具侧的待修项（不是生成质量差），定向重试同样失败。")
        w("它们的在线结果（分母")
        w("eager）是：")
        w("")
        for t in unaudited:
            hit = [k for k in A if tid(k) == t]
            if not hit:
                continue
            r = A[hit[0]]
            sp = r["speedup"]
            w(f"- `{t}`：在线 best 正确 {r['best_correct']}，speedup "
              f"{fmt(sp) + 'x' if isinstance(sp, (int, float)) else '-'}"
              f"{'（仅 root）' if r['is_root'] else ''}")
    w("")
    w("### 5.1 为什么 HIP 题没有 aiter / hipDNN 基线")
    w("")
    w(f"这 {len(hip_ids)} 道题源自 `opendtk/hipdnn_samples`（准入记录见 `benchmark/sources/`）。按 AGENTS.md，")
    w("**闭源 hipDNN 库只作性能基线、不作题源**，本应给它们补 hipDNN 分母。但")
    w("`hipdnn_baseline_probe.md` 的实测结论是：")
    w("")
    w("> **在当前机器（DTK 26.04）上 hipDNN 做不了基线。** 阻塞点在 vendor 自己的引擎插件")
    w("> `libmiopen_legacy_plugin.so`（Convolution plan builder 内部 segfault），不在我们的工具链；")
    w("> 官方样例本身要求 DTK ≥ 26.10。hipDNN 的 Python 绑定也未随 26.04 发布，全盘无")
    w("> `hipdnn*.whl`。")
    w("")
    w("aiter 侧同样没有对等算子可作分母（hipDNN 样例覆盖的 deform conv / spconv /")
    w("depth_to_space / adamw / ctc_loss / cumsum / gather / scatter_nd 在 aiter 里都没有")
    w(f"可直接对位的实现）。**所以这 {len(hip_ids)} 道的基线状况与 hipDNN 探针的结论完全一致：只有")
    w("`torch_eager` 一个分母。**")
    w("")
    w("### 5.2 但这不代表分母一定弱 —— 分三档看")
    w("")
    w("探针第 5 节实测：`libtorch_hip.so` 链接 `libMIOpen.so.1`、")
    w("`torch.backends.cudnn.is_available()` 为 True（DCU 上即 MIOpen 后端）、")
    w("`F.conv2d` 实际落到 MIOpen 的 `winograd_fwd_2d_fp32_3x3_s1_16x16x4` 内核。因此：")
    w("")
    w("| 档 | 题 | `torch_eager` 分母的含义 |")
    w("|---|---|---|")
    conv = sorted(t for t in hip_ids if t in CONV_FAMILY_MIOPEN)
    aten = sorted(t for t in hip_ids if t in PURE_ATEN)
    rest = sorted(hip_ids - set(conv) - set(aten))
    w(f"| **conv 族：eager 即 vendor 库分母** | {len(conv)} 道：" +
      "、".join(f"`{t}`" for t in conv) +
      " | torch 走 MIOpen → eager 分母**就是 vendor 实现**，不弱 |")
    w(f"| **纯 ATen** | {len(aten)} 道：" + "、".join(f"`{t}`" for t in aten) +
      " | 探针明确列为无 MIOpen 路径 → eager 是纯 ATen，偏弱 |")
    w(f"| **探针未覆盖** | {len(rest)} 道：" + "、".join(f"`{t}`" for t in rest) +
      " | 未验证是否走 MIOpen，**不作断言** |")
    w("")
    w(f"引用这 {len(hip_ids)} 道的 speedup 时必须注明分母含义：conv 族（5001/5002/5008/5009）的 eager")
    w("分母已经是 vendor 库，其加速比含金量高于其余各道。")
    w("")

    w("## 6. 结论")
    w("")
    w(f"1. **编译率**：root 脚手架 {AB['root_compile']}→{AA['root_compile']}/{AA['n']}，"
      f"所有题都拿到了可跑的起点。")
    w(f"2. **正确率（在线）**：best 正确 {AB['best_correct']}→{AA['best_correct']}/{AA['n']}"
      f"（{AA['best_correct']/AA['n']*100:.1f}%），真解 {AB['real']}→{AA['real']}。")
    if audit:
        w(f"3. **正确率（离线终审）**：审 {audit['aggregates']['audited']} 道有 aiter 基线的题，"
          f"通过 {audit['aggregates']['correctness_passed']} 道 —— 比在线严，两套都对但量的不是同一件事。")
        w(f"4. **对 aiter 的加速比**（离线）：中位 "
          f"{audit['aggregates']['aiter_median_speedup']:.2f}x、均值 "
          f"{audit['aggregates']['aiter_mean_speedup']:.2f}x、最大 "
          f"{audit['aggregates']['aiter_max_speedup']:.2f}x；其中 "
          f"{audit['aggregates']['mi_fallback_count']} 道是 MI 兜底基线、引用须标注。")
    _n = 5 if audit else 3
    w(f"{_n}. **在线加速比**（vs eager）：真解中位 "
      f"{fmt(AB['dist']['median'] if AB['dist'] else None)}x → "
      f"{fmt(AA['dist']['median'] if AA['dist'] else None)}x。")
    w(f"{_n + 1}. **基线覆盖**：{len(no_aiter)} 道无 aiter 基线（含 HIP 全部 {len(hip_ids)} 道），"
      f"只能给 eager 口径；HIP 的 conv 族 eager 分母即 MIOpen。")
    w("")

    out = pathlib.Path(args.out)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"已写 {out}  ({len(L)} 行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
