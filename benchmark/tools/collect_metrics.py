"""按题目级指标汇总一次全量生成跑的结果。

这些指标与 `stats_full66.py` 的口径不同：那边比的是「before/after 两轮」，
这边是**单轮内部**的题目级覆盖率与收敛速度，用于回答「66 道题里有多少道真的
被解出来了、第几次尝试解出来的、后面还精修了多少」。

指标定义（accepted_tasks = manifest 里的题数，本题 66）：

  Compile Rate     产出 ≥1 个 compile_ok 候选的题数 / accepted_tasks
  Correct Rate     产出 ≥1 个 correct 候选的题数 / accepted_tasks
  Success@K        每题 K 次尝试内出现过 correct 候选的题数 / accepted_tasks
  Fast1@K          每题 K 次尝试内出现过 correct 且 speedup > 1 的候选的题数 / accepted_tasks
  Geomean/Median/Mean/Best Speedup   每题 best correct 候选 speedup 的几何平均/中位/均值/最大
  First Correct Median               每个 solved 题首次 correct 的尝试序号中位数
  Refine Gain Median                 每题 (最终 best speedup / 首次 correct speedup) 的中位数

口径约定（写死在代码里，避免下次换脚本走样）：

  1. `root` 是框架给的直通脚手架，**不算** Agent 产出，因此 Compile/Correct Rate、
     Success@K、Fast1@K 默认排除 root；`--include-root` 可切换成含 root 的口径。
  2. 「参考 baseline」= run.json 里的 `primary_reference`（本题全为 `torch_eager`），
     所以「快于 baseline」就是 speedup > 1。
  3. 尝试序号：`root` 记 0，`n1..nN`（phase1）按编号递增，`phase2_n1..nN` 接在后面。
     框架不给显式 attempt index，这是唯一稳定的创建顺序（node id 就是按创建分配的）。
  4. 题目目录名形如 `<alias>_l<level>_p<problem_id>`，alias 按 `_l` 切分还原。

用法：
    python benchmark/tools/collect_metrics.py --run-root .dcu_runs/en66 --out benchmark/tools/metrics_en.md
    python benchmark/tools/collect_metrics.py --run-root <root> --csv metrics.csv --include-root
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
import statistics

ROOT_NODE_IDS = ("root", "phase2_root")

# 最高加速比那种"离谱"数字的成因备注（按题名索引；没有条目就走通用说明）。
# 写在这里而不是手改 md，是为了重跑脚本时这段话不会丢。
TOP_TASK_NOTES = {
    "7001_ctc_loss": [
        "分母是题面给的**朴素参考实现**：`_ctc_one` 对每个 batch 元素各跑一遍 Python 循环"
        "（前向 `for t in range(1, tb)` + 反向再来一遍），每一步临时造 6~7 个张量、发 6~7 次 "
        "kernel。在线 shape（T=500、N=4、tb≤200）下约 **1.1 万次 launch**，272 ms 几乎全是主机侧开销。",
        "",
        "生成的实现是一个 Triton kernel：一个 program 处理一个 batch 元素，状态维 S 放寄存器"
        "（`tl.arange(0,B)`），**时间维递推写在 kernel 内部**，对数域三分支 logsumexp 手写"
        "（`_ctc_lse3`），前向存 alpha、反向在同一 kernel 里算 gamma 并 `atomic_add` 写梯度 —— "
        "整个 500 步递推只有 **1 次 launch**，0.30 ms。",
        "",
        "所以这 900x 的绝大部分来自**消除上万次 launch + Python 开销**，算法复杂度两边一样"
        "（都是 O(T·S) 对数域 CTC），没有任何渐进优势。",
        "",
        "**引用时必须注明分母**：不能写成“比生产级 CTC 快 900 倍”。真正对位的生产实现是 "
        "`torch.nn.functional.ctc_loss`，而它正是本题的禁用算子（所以不会出现在分母里）。"
        "离线那 4319x 更是同一原因在更大 perf case（T=1024、N=16、C=32）上的放大：参考实现"
        "成本 ∝ T·N，生成实现把 N 摊到多个 program、T 留在 kernel 内循环。",
    ],
}


def attempt_order(node_id: str) -> tuple:
    """创建顺序：root=0，phase1 的 nK=K，phase2 的 phase2_nK 接在后面。"""
    if node_id in ROOT_NODE_IDS:
        return (0, 0)
    m = re.fullmatch(r"phase2_n(\d+)", node_id)
    if m:
        return (2, int(m.group(1)))
    m = re.fullmatch(r"n(\d+)", node_id)
    if m:
        return (1, int(m.group(1)))
    return (3, 0)  # 未知命名，排在最后


def load_tasks(run_root: pathlib.Path) -> dict:
    """alias -> run.json dict。兼容扁平布局与 par_sNN/ 分片布局。"""
    out = {}
    for p in sorted(run_root.rglob("run.json")):
        alias = p.parent.name.rsplit("_l", 1)[0]
        out[alias] = json.loads(p.read_text(encoding="utf-8"))
    return out


def node_list(run: dict) -> list:
    nodes = run.get("nodes") or {}
    if isinstance(nodes, dict):
        return list(nodes.values())
    return nodes


def task_facts(run: dict, include_root: bool) -> dict:
    nodes = node_list(run)
    ranked = []
    for n in nodes:
        nid = n.get("node_id")
        ranked.append((attempt_order(nid), nid, n))
    ranked.sort(key=lambda x: (x[0], x[1] or ""))
    # 逐个编号：root 的序号为 0，其余按顺序 1..N
    seq = {}
    counter = 0
    for order, nid, _ in ranked:
        if nid in ROOT_NODE_IDS:
            seq[nid] = 0
        else:
            counter += 1
            seq[nid] = counter

    considered = [n for _, nid, n in ranked if include_root or nid not in ROOT_NODE_IDS]
    produced = [n for n in considered]
    compile_ok = [n for n in produced if n.get("compile_ok")]
    correct = [n for n in produced if n.get("correct")]
    correct_with_idx = sorted(((seq[n["node_id"]], n) for n in correct), key=lambda x: x[0])

    def speed(n):
        v = n.get("speedup")
        return v if isinstance(v, (int, float)) else None

    correct_sp = [(i, n, speed(n)) for i, n in correct_with_idx]
    best = None
    for _, n, sp in correct_sp:
        if sp is not None and (best is None or sp > best):
            best = sp
    champ_id = run.get("best_node_id")
    champ_sp = next((speed(n) for n in nodes if n.get("node_id") == champ_id), None)
    first_sp = next((sp for _, _, sp in correct_sp if sp is not None), None)
    first_idx = correct_sp[0][0] if correct_sp else None
    fast_idx = next((i for i, _, sp in correct_sp if sp is not None and sp > 1.0), None)

    return {
        "alias": None,
        "nodes": len(nodes),
        "attempts": counter,
        "compile_ok": bool(compile_ok),
        "correct": bool(correct),
        "first_correct_idx": first_idx,
        "first_fast_idx": fast_idx,
        "first_correct_speedup": first_sp,
        "best_speedup": best,
        "best_node": max(correct, key=lambda n: speed(n) or -1)["node_id"] if correct else None,
        "correct_count": len(correct),
        "champion_node": champ_id,
        "champion_speedup": champ_sp,
        "reference": run.get("primary_reference"),
        "backend": run.get("backend"),
    }


def pct(k: int, n: int) -> str:
    return f"{k}/{n}（{k / n * 100:.1f}%）" if n else "-"


def geomean(vals: list) -> float | None:
    vals = [v for v in vals if isinstance(v, (int, float)) and v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True, type=pathlib.Path)
    ap.add_argument("--out", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--include-root", action="store_true",
                    help="把 root 直通脚手架也算作候选（默认不算，因为不是 Agent 产出）")
    ap.add_argument("--k", default="1,5,10", help="Success@K / Fast1@K 的 K 列表")
    ap.add_argument("--note", default="", help="一句话标注这批 run.json 的来源（写进报告开头）")
    ap.add_argument("--offline-audit", default="",
                    help="audit_batch.py 产出的 offline_audit.json；给了就追加一节离线终审")
    ap.add_argument("--audit-cache", default="",
                    help="audit_batch.py 的单题缓存目录（读 static_audit_passed 判静态判负）")
    args = ap.parse_args()

    tasks = load_tasks(args.run_root)
    if not tasks:
        print(f"[FAIL] {args.run_root} 下没有 run.json")
        return 1
    rows = []
    for alias, run in sorted(tasks.items()):
        f = task_facts(run, args.include_root)
        f["alias"] = alias
        rows.append(f)

    n = len(rows)
    ks = [int(x) for x in args.k.split(",") if x.strip()]
    solved = [r for r in rows if r["correct"]]
    sp = [r["best_speedup"] for r in solved if isinstance(r["best_speedup"], (int, float))]
    first_idx = [r["first_correct_idx"] for r in solved if r["first_correct_idx"]]
    gains = [r["best_speedup"] / r["first_correct_speedup"] for r in solved
             if isinstance(r["best_speedup"], (int, float))
             and isinstance(r["first_correct_speedup"], (int, float))
             and r["first_correct_speedup"] > 0]

    L = []
    w = L.append
    refs = {r["reference"] for r in rows}
    w(f"# 题目级指标（{args.run_root}）")
    w("")
    w(f"- accepted_tasks = **{n}**（solved = {sum(1 for r in rows if r['correct'])}）")
    w(f"- 候选口径：{'包含 root' if args.include_root else '**排除 root**（root 是框架直通脚手架，不算 Agent 产出）'}")
    w(f"- 参考 baseline（speedup 分母）：{', '.join(f'`{x}`' for x in sorted(refs))}")
    if args.note:
        w(f"- 数据来源：{args.note}")
    w("")
    w("## 指标定义")
    w("")
    w("| 指标 | 定义 |")
    w("|---|---|")
    w("| Compile Rate | 题目级：至少产出 1 个 `compile_ok` 候选的题目数 / accepted_tasks |")
    w("| Correct Rate | 题目级：至少产出 1 个 `correct` 候选的题目数 / accepted_tasks |")
    w("| Success@K | 每题 K 次尝试内至少出现 1 个 correct 候选的题目比例 |")
    w("| Fast1@K | 每题 K 次尝试内至少出现 1 个 correct 且快于参考 baseline（speedup > 1）的候选的题目比例 |")
    w("| Geomean Speedup | 每题 best correct 候选 speedup 的几何平均（只在 solved tasks 上算） |")
    w("| Median Speedup | 每题 best correct 候选 speedup 的中位数 |")
    w("| Best Speedup | 所有题里最高的 best correct speedup |")
    w("| Mean Speedup | 每题 best correct 候选 speedup 的算术平均 |")
    w("| First Correct Median | 每个 solved task 第一次出现 correct 的尝试序号的中位数 |")
    w("| Refine Gain Median | 每题（最终 best correct speedup / 第一次 correct 的 speedup）的中位数 |")
    w("")
    w("两点口径说明：**① root 不在候选池内**（否则 Correct Rate 恒为 100%）；"
      "**② best = 所有 correct 候选里的最大 speedup**，与框架按 score 挑的 `best_node_id` 可能不同，"
      "逐题明细最后一列给出 champion 供对照。")
    w("")
    w("| 指标 | 值 |")
    w("|---|---|")
    w(f"| Compile Rate | **{pct(sum(1 for r in rows if r['compile_ok']), n)}** |")
    w(f"| Correct Rate | **{pct(sum(1 for r in rows if r['correct']), n)}** |")
    for k in ks:
        hit = sum(1 for r in rows if r["first_correct_idx"] is not None and r["first_correct_idx"] <= k)
        w(f"| Success@{k} | {pct(hit, n)} |")
    for k in ks:
        hit = sum(1 for r in rows if r["first_fast_idx"] is not None and r["first_fast_idx"] <= k)
        w(f"| Fast1@{k} | {pct(hit, n)} |")
    hit_all = sum(1 for r in rows if r["first_fast_idx"] is not None)
    w(f"| Fast1@all（整个预算内） | {pct(hit_all, n)} |")
    w("")
    w("K 曲线（K = 每题允许的尝试次数上限）：")
    w("")
    w("| K | " + " | ".join(str(k) for k in range(1, 11)) + " |")
    w("|---" * 11 + "|")
    w("| Success@K | " + " | ".join(
        str(sum(1 for r in rows if r["first_correct_idx"] is not None and r["first_correct_idx"] <= k))
        for k in range(1, 11)) + " |")
    w("| Fast1@K | " + " | ".join(
        str(sum(1 for r in rows if r["first_fast_idx"] is not None and r["first_fast_idx"] <= k))
        for k in range(1, 11)) + " |")
    w("")
    w(f"在 {len(solved)} 道 solved 题上统计 speedup。**best 的定义是「所有 correct 候选里的最大"
      f" speedup」**，与框架自己选的 `best_node_id`（按 score 挑）可能不同；下表最后一列给出"
      f" 框架 champion 的 speedup 供对照：")
    w("")
    w("")
    w("| 指标 | 值 |")
    w("|---|---|")
    g = geomean(sp)
    w(f"| Geomean Speedup | **{g:.2f}x** |" if g else "| Geomean Speedup | - |")
    w(f"| Median Speedup | **{statistics.median(sp):.2f}x** |" if sp else "| Median Speedup | - |")
    w(f"| Mean Speedup | {statistics.mean(sp):.2f}x |" if sp else "| Mean Speedup | - |")
    w(f"| Best Speedup | **{max(sp):.2f}x** |" if sp else "| Best Speedup | - |")
    w(f"| First Correct Median（尝试序号） | **{statistics.median(first_idx):.1f}** |"
      if first_idx else "| First Correct Median | - |")
    w(f"| Refine Gain Median | **{statistics.median(gains):.2f}x** |" if gains else "| Refine Gain Median | - |")
    w("")
    if first_idx:
        dist = {}
        for i in first_idx:
            dist[i] = dist.get(i, 0) + 1
        w("首次 correct 的尝试序号分布：" + "、".join(f"{k}→{v} 道" for k, v in sorted(dist.items())))
        w("")
    w("## 逐题明细")
    w("")
    w("| 题 | backend | 尝试数 | 首次 correct | 首次快于 baseline | best speedup（max correct） | best 节点 | 框架 champion |")
    w("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: -(x["best_speedup"] or 0)):
        bs = r["best_speedup"]
        bs_txt = f"{bs:.2f}x" if isinstance(bs, (int, float)) else "-"
        fc = r["first_correct_idx"] if r["first_correct_idx"] is not None else "-"
        ff = r["first_fast_idx"] if r["first_fast_idx"] is not None else "-"
        ch = r["champion_speedup"]
        ch_txt = f"{ch:.2f}x（{r['champion_node']}）" if isinstance(ch, (int, float)) else "-"
        w(f"| `{r['alias']}` | {r['backend']} | {r['attempts']} | {fc} | {ff} | "
          f"{bs_txt} | {r['best_node'] or '-'} | {ch_txt} |")
    w("")

    # 简要描述（放在「指标定义」之前，自动生成以免和上面的数字脱节）
    miss = [r["alias"] for r in rows if not r["correct"]]
    slow = [r["alias"] for r in rows if r["correct"] and r["first_fast_idx"] is None]
    desc = [f"- **能编译**：{sum(1 for r in rows if r['compile_ok'])}/{n} 道题至少产出一个可编译候选，"
            f"脚手架不是瓶颈。"]
    desc.append(f"- **能写对**：Correct Rate {sum(1 for r in rows if r['correct'])}/{n}"
                f"（{sum(1 for r in rows if r['correct'])/n*100:.1f}%）"
                f"{('；' + str(len(miss)) + ' 道始终没写出正确实现：' + '、'.join(f'`{a}`' for a in miss)) if miss else ''}。")
    if first_idx:
        desc.append(f"- **收敛速度**：首次 correct 的中位尝试序号 {statistics.median(first_idx):.1f}，"
                    f"Success@1 {sum(1 for r in rows if r['first_correct_idx'] == 1)}/{n}、"
                    f"Success@5 {sum(1 for r in rows if r['first_correct_idx'] is not None and r['first_correct_idx'] <= 5)}/{n}。")
    if sp:
        desc.append(f"- **跑得比 baseline 快**：Fast1@all {hit_all}/{n}"
                    f"（{hit_all/n*100:.1f}%）；solved 题 best speedup 几何平均 {g:.2f}x、"
                    f"中位 {statistics.median(sp):.2f}x、最好 {max(sp):.2f}x"
                    f"{('；另有 ' + str(len(slow)) + ' 道虽写对但整个预算内没跑赢 eager') if slow else ''}。")
    if gains:
        desc.append(f"- **精修收益有限**：Refine Gain 中位 {statistics.median(gains):.2f}x —— 第一次写对之后，"
                    f"剩下的尝试只能再提升这么多，难点在「第一次对不对」。")
    # ---- 离线终审小节（可选）----
    if args.offline_audit and pathlib.Path(args.offline_audit).is_file():
        audit = json.loads(pathlib.Path(args.offline_audit).read_text(encoding="utf-8"))
        arows = {r["task"]: r for r in audit.get("rows") or []}
        cache_dir = pathlib.Path(args.audit_cache) if args.audit_cache else None

        def _cache(t):
            if cache_dir is None:
                return {}
            f = cache_dir / f"{t}.json"
            return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}

        ag = audit.get("aggregates") or {}
        head = audit.get("headline_impl") or "aiter"
        static_bad = [t for t in arows if _cache(t).get("static_audit_passed") is False]
        crashed = [t for t in arows if arows[t].get("failure_reason")]
        corr_bad = [t for t in arows
                    if not arows[t]["correctness_passed"]
                    and t not in static_bad and t not in crashed]
        passed = [t for t in arows if arows[t]["correctness_passed"]]
        with_impl = [(t, arows[t][f"{head}_mean_speedup"]) for t in arows
                     if isinstance(arows[t].get(f"{head}_mean_speedup"), (int, float))]
        L.append(f"## 离线终审（隐藏 case + 静态审计，分母 = `{head}`）")
        L.append("")
        L.append(f"- **{ag.get('audited', len(arows))} 道全审，通过 {len(passed)} 道**")
        L.append(f"- 未通过 {len(arows) - len(passed)}：静естатиc".replace("静естатиc", "静") )
        L[-1] = (f"- 未通过 {len(arows) - len(passed)}：静态判负 {len(static_bad)}、"
                 f"隐藏 case 跑不起来 {len(crashed)}、数值不达标 {len(corr_bad)}")
        if with_impl:
            vals = sorted(v for _, v in with_impl)
            L.append(f"- `{head}` 有测值 {len(vals)} 道：中位 **{statistics.median(vals):.2f}x**、"
                     f"均值 {statistics.mean(vals):.2f}x、最大 {max(vals):.2f}x；"
                     f"MI 兜底基线 {ag.get('mi_fallback_count', 0)} 道（分母偏大，加速比偏高，引用须标注）")
        L.append("")
        L.append("**为什么不能只看在线**：在线只跑 `get_inputs()` 的单一 shape、不执行禁用列表，")
        L.append(f"所以在线的 {sum(1 for r in rows if r['correct'])} 道 solved 实现里，有 "
                 f"{len(static_bad)} 道把核心计算交给了被禁的算子、"
                 f"{len(crashed)} 道换一批隐藏 shape 就崩、"
                 f"{len(corr_bad)} 道数值对不上。离线才是判据的硬边。")
        L.append("")
        if static_bad:
            L.append("| 静态判负的题 |")
            L.append("|---|")
            for t in sorted(static_bad):
                L.append(f"| `{t}` |")
            L.append("")
        if crashed:
            L.append("| 隐藏 case 跑不起来的题 | 崩溃摘要 |")
            L.append("|---|---|")
            for t in sorted(crashed):
                tail = " ".join((arows[t].get("failure_tail") or "").split())[-120:].replace("|", "\\|")
                L.append(f"| `{t}` | `{tail}` |" if tail else f"| `{t}` | {arows[t].get('failure_reason')} |")
            L.append("")
        # ---- 合规口径：在线 solved 再叠一道离线门槛 ----
        has_cand = [r["alias"] for r in rows if r["best_speedup"]]
        gated = [t for t in has_cand if t in passed]
        dropped = [t for t in has_cand if t not in passed]
        sp_on = [r["best_speedup"] for r in rows if r["alias"] in has_cand and r["best_speedup"]]
        sp_gate = [r["best_speedup"] for r in rows if r["alias"] in gated and r["best_speedup"]]
        L.append("### 加离线门槛后的口径（对外引用建议用这一栏）")
        L.append("")
        L.append("| 指标 | 只算在线 | **在线 ∩ 离线通过** |")
        L.append("|---|---|---|")
        L.append(f"| 算 solved 的题数 | {len(has_cand)}/{n} | **{len(gated)}/{n}** |")
        if sp_on and sp_gate:
            L.append(f"| 加速比中位 | {statistics.median(sp_on):.2f}x | "
                     f"**{statistics.median(sp_gate):.2f}x** |")
            L.append(f"| 加速比均值 | {statistics.mean(sp_on):.2f}x | "
                     f"**{statistics.mean(sp_gate):.2f}x** |")
            L.append(f"| 加速比几何平均 | {geomean(sp_on):.2f}x | **{geomean(sp_gate):.2f}x** |")
            L.append(f"| 加速比最大 | {max(sp_on):.2f}x | {max(sp_gate):.2f}x |")
        L.append("")
        L.append(f"被门槛剔掉的 {len(dropped)} 道："
                 + "、".join(f"`{t}`" for t in sorted(dropped)))
        L.append("")
        L.append("注意：`Success@K` / `First Correct Median` / `Refine Gain Median` **不能**这样加门槛——")
        L.append("离线终审只审最终 `best_code.py`，不审每一次尝试；要按尝试门槛化就得把每个候选都拿去审计")
        L.append("（约 66×12≈790 次，单题 1–2 分钟），代价过高，所以这三项保留在线口径并在结论里注明。")
        L.append("")
        if with_impl:
            L.append(f"| 题 | vs {head} | vs eager | 备注 |")
            L.append("|---|---|---|---|")
            for t, v in sorted(with_impl, key=lambda kv: -kv[1]):
                note = "MI 兜底†" if arows[t].get("aiter_is_mi_fallback") else ""
                e = arows[t].get("torch_eager_mean_speedup")
                L.append(f"| `{t}` | **{v:.2f}x** | {e:.2f}x | {note} |" if isinstance(e, (int, float))
                         else f"| `{t}` | **{v:.2f}x** | - | {note} |")
            L.append("")

    # 观察（数据驱动）+ 最高加速比成因（人工备注，key 为题名）
    obs = []
    if slow:
        obs.append(f"- **Correct Rate {sum(1 for r in rows if r['correct'])}/{n} 但 Fast1@all 只有 "
                   f"{hit_all}/{n}**：差出来的 {len(slow)} 道是「能写对、但整个预算内没跑赢 "
                   f"`torch_eager`」（"
                   + "、".join(f"`{a}`" for a in slow[:6]) +
                   ("…" if len(slow) > 6 else "") +
                   "）——这 10 道横跨 quant/GEMM、MoE、decode-attention 与 conv，"
                   "共同点是 eager 分母本身已经是 ATen/MIOpen 的高度优化实现，生成实现很难超过。")
    if gains:
        obs.append(f"- **Refine Gain 中位只有 {statistics.median(gains):.2f}x**：一旦第一次写对，"
                   f"后续尝试平均只再提升这么多。也就是说瓶颈在「第一次能不能写对」"
                   f"（Success@1 {sum(1 for r in rows if r['first_correct_idx'] == 1)}/{n}），"
                   f"而不是精修阶段——把预算从 10 次加到 20 次，预期收益有限。")
    if obs:
        L.append("## 两点观察")
        L.append("")
        L.extend(obs)
        L.append("")

    if sp:
        top = max(solved, key=lambda r: r["best_speedup"] or 0)
        L.append(f"## 最高加速比为什么这么高（`{top['alias']}` {top['best_speedup']:.2f}x）")
        L.append("")
        note = TOP_TASK_NOTES.get(top["alias"])
        if note:
            L.extend(note)
        else:
            L.append(f"`{top['alias']}` 的 best correct 候选比参考 baseline 快 "
                     f"{top['best_speedup']:.2f}x；逐题明细里有它的节点与尝试序号，"
                     f"可回到对应的 `best_code.py` 看具体实现。")
        L.append("")

    idx = L.index("## 指标定义")
    L[idx:idx] = ["## 简要描述"] + desc + [""]

    text = "\n".join(L) + "\n"
    if args.out:
        pathlib.Path(args.out).write_text(text, encoding="utf-8")
        print(f"已写 {args.out}（{len(L)} 行）")
    else:
        print(text)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print(f"已写 {args.csv}")

    # 控制台摘要
    print()
    print(f"accepted={n}  compile_ok={sum(1 for r in rows if r['compile_ok'])}  "
          f"correct={sum(1 for r in rows if r['correct'])}  "
          f"fast1@all={hit_all}")
    if sp:
        print(f"speedup: geomean {g:.2f}x  median {statistics.median(sp):.2f}x  "
              f"mean {statistics.mean(sp):.2f}x  best {max(sp):.2f}x")
    if first_idx:
        print(f"first_correct_median {statistics.median(first_idx):.1f}  "
              f"refine_gain_median {statistics.median(gains):.2f}x" if gains else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
