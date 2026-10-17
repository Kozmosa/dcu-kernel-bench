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
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    B = {k: row(v) for k, v in load(args.before).items()}
    A = {k: row(v) for k, v in load(args.after).items()}
    repo = pathlib.Path(args.repo)
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

    w("# 66 道生成评测性能统计（双基线，2026-10-17）")
    w("")
    w("五个 harness 缺陷修复后的全量 66 道结果。数据源：")
    w("")
    w(f"- 修复前在线：`runs/dcukb_final`（{AB['n']} 道）")
    w(f"- 修复后在线：`runs/dcukb_full_fixed`（{AA['n']} 道，16 片并行，10:08–11:53）")
    if audit:
        w(f"- 修复后离线终审：`offline_audit.json`（审 {audit['aggregates']['audited']} 道有 aiter 基线的题）")
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
    w("| 指标 | 修复前 | 修复后 |")
    w("|---|---|---|")
    w(f"| root 脚手架编译通过（= 框架给出的起点可用） | **{AB['root_compile']}/{AB['n']}** | **{AA['root_compile']}/{AA['n']}** |")
    w(f"| best 节点编译通过 | {AB['best_compile']}/{AB['n']} | {AA['best_compile']}/{AA['n']} |")
    w(f"| compile 阶段失败次数（合计，含 Agent 自己写坏） | {AB['compile_fails']} | {AA['compile_fails']} |")
    w(f"| runtime 阶段失败次数（合计） | {AB['runtime_fails']} | {AA['runtime_fails']} |")
    w("")
    w("root 脚手架编译率衡量 **harness 质量**：修复前 7 道因脚手架多行签名截断直接")
    w("SyntaxError，Agent 连起点都没有；修复后 66/66。compile 阶段失败次数不归零是")
    w("正常的 —— 那是 Agent 尝试里自己写坏的补丁，属生成侧信号。")
    w("")

    w("## 2. 正确率")
    w("")
    w("### 2.1 在线（KernelBench 口径）")
    w("")
    w("| 指标 | 修复前 | 修复后 |")
    w("|---|---|---|")
    w(f"| best 正确 | {AB['best_correct']}/{AB['n']}（{AB['best_correct']/AB['n']*100:.1f}%） | **{AA['best_correct']}/{AA['n']}（{AA['best_correct']/AA['n']*100:.1f}%）** |")
    w(f"| └ 非 root 真解 | {AB['real']}（{AB['real']/AB['n']*100:.1f}%） | **{AA['real']}（{AA['real']/AA['n']*100:.1f}%）** |")
    w(f"| └ 仅 root（Agent 未改写） | {AB['rootonly']} | {AA['rootonly']} |")
    w(f"| 有 speedup 可比 | {AB['has_speedup']}/{AB['n']} | {AA['has_speedup']}/{AA['n']} |")
    w("")
    w("按 backend（修复后）：")
    w("")
    w("| backend | 题数 | best 正确（前→后） | 真解（前→后） |")
    w("|---|---|---|---|")
    for be, label in (("cuda", "HIP（backend=cuda）"), ("triton", "Triton")):
        kb = [k for k in B if B[k]["backend"] == be]
        ka = [k for k in A if A[k]["backend"] == be]
        w(f"| {label} | {len(ka)} | "
          f"{sum(1 for k in kb if B[k]['best_correct'])}→{sum(1 for k in ka if A[k]['best_correct'])} | "
          f"{sum(1 for k in kb if B[k]['best_correct'] and not B[k]['is_root'])}→"
          f"{sum(1 for k in ka if A[k]['best_correct'] and not A[k]['is_root'])} |")
    w("")

    if audit:
        ag = audit["aggregates"]
        w("### 2.2 离线终审（隐藏 case + `task.yaml` 分档容差，严于在线）")
        w("")
        w(f"只审有 aiter 基线的题：**审了 {ag['audited']} 道，正确性通过 {ag['correctness_passed']} 道"
          f"（{ag['correctness_passed']/ag['audited']*100:.0f}%）**。")
        w("")
        bad = [r["task"] for r in audit["rows"] if not r["correctness_passed"]]
        w(f"未通过（{len(bad)} 道）：" + "、".join(f"`{t}`" for t in bad))
        w("")
        w("> 这些题在**在线**是判对的（单一 shape、1e-4 allclose），但在**隐藏 case + 分档容差**")
        w("> 下不成立。两个数字都对，只是量的不是同一件事：在线反映「能不能跑通并对上参考」，")
        w("> 离线反映「换一批 shape/边界还对不对」。引用时必须说明用的是哪一套。")
        w("")

    w("## 3. 在线加速比（分母 = `torch_eager`）")
    w("")
    w("统计口径：只看**真解**（best 非 root 且 correct）。")
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
    w("| 区间 | 修复前 | 修复后 |")
    w("|---|---|---|")
    for label in ["0–1x", "1–2x", "2–5x", "5–10x", "10–100x", "≥100x"]:
        w(f"| {label} | {bb.get(label, 0)} | {ba.get(label, 0)} |")
    w("")

    ranked = sorted(
        [(k, A[k]["speedup"]) for k in A
         if A[k]["best_correct"] and not A[k]["is_root"] and isinstance(A[k]["speedup"], (int, float))],
        key=lambda kv: -kv[1])
    w("Top 10（修复后，在线口径）：")
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

    w("## 5. 只有 `torch_eager` 基线的题（含 HIP 那 20 道）")
    w("")
    w(f"66 道里有 **{len(no_aiter)} 道没有 aiter 基线**（`baseline.json` 的 impls 只有")
    w(f"`torch_eager`），因此只能给 `torch_eager` 口径。其中 **{len(hip_ids)} 道是 HIP track")
    w("（`backend=cuda`）**，全部落在这一类。")
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
        w("这两道已用 `--tasks ... --force` 定向重试，**可复现地失败**（`2004_fused_qk_concat`")
        w("是 SIGSEGV）。属离线终审工具侧的待修项，与生成质量无关。它们的在线结果（分母")
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
    w("这 20 道题源自 `opendtk/hipdnn_samples`（准入记录见 `benchmark/sources/`）。按 AGENTS.md，")
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
    w("可直接对位的实现）。**所以这 20 道的基线状况与 hipDNN 探针的结论完全一致：只有")
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
    w("引用这 20 道的 speedup 时必须注明分母含义：conv 族（5001/5002/5008/5009）的 eager")
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
    w(f"5. **在线加速比**（vs eager）：真解中位 "
      f"{fmt(AB['dist']['median'] if AB['dist'] else None)}x → "
      f"{fmt(AA['dist']['median'] if AA['dist'] else None)}x。")
    w(f"6. **基线覆盖**：{len(no_aiter)} 道无 aiter 基线（含 HIP 全部 {len(hip_ids)} 道），"
      f"只能给 eager 口径；HIP 的 conv 族 eager 分母即 MIOpen。")
    w("")

    out = pathlib.Path(args.out)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"已写 {out}  ({len(L)} 行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
