#!/usr/bin/env python3
"""批量离线终审：把在线生成跑的产物 best_code.py 逐题喂给 audit_model_class，
汇总出「与原算子基线（默认 aiter）的加速比」。

为什么需要它：在线（PyramidKernel）与离线（本评测集终审）是两条链，口径不同，
不能混用：

  在线  KernelBench 口径：输入统一 cast fp32、allclose(atol=rtol=1e-4)；
        speedup 的分母只能是 torch_eager / torch_compile_*（框架没有"外部基线"模式）
  离线  本文件这一侧：隐藏 case 按 task.yaml 的分档容差判定；perf 用
        private/<id>/perf_cases.json 的形状与计时参数；speedup 的分母来自
        private/<id>/baseline.json 里采的基线（aiter 官方实现 / torch_eager）

所以「对 aiter 的加速比」只能在这里算：audit_model_class.py 会把 baseline.json 里
**每一条**基线都算一遍，用 impl 字段区分（见其 stage_perf）。本脚本负责遍历、缓存、
汇总，并明确标注两种口径不可混比。

用法（在仓库根目录，或任意 cwd——audit_model_class 的路径相对它自己的文件解析）：

    python3 benchmark/evaluator/audit_batch.py \
        --run-root /root/dcudeploy/runs/dcukb_all \
        [--manifest benchmark/kernelbench_compat/manifest_dcukb.yaml] \
        [--impl aiter] [--tasks 1002_paged_attention,1017_mha] \
        [--include-no-aiter] [--limit N] [--force] [--dry-run]

断点续跑：每题结果缓存在 <run-root>/offline_audit/<task>.json，重跑自动跳过已完成的；
加 --force 重算。汇总写到 <run-root>/offline_audit.json。

退出码：0 正常；1 有题失败（但仍写出已完成的汇总）。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PRIV = REPO_ROOT / "benchmark" / "private"
DEFAULT_MANIFEST = REPO_ROOT / "benchmark" / "kernelbench_compat" / "manifest_dcukb.yaml"
AUDIT = Path(__file__).parent / "audit_model_class.py"

# 这些题的 aiter 基线是在**缺 BW200 tuner 配置**时回退到 MI 系列配置跑出来的：
# 未针对 DCU 调优 → 分母偏大 → 对 aiter 的 speedup 会**偏高**。引用时必须标注。
# （来源：逐题核对 309 个本地 tuner 配置，见 .dcu_runs/blockers-and-fixes.md）
AITER_MI_FALLBACK = {
    "1014_hstu_attention", "1017_mha", "4001_batched_gemm_a8w8",
    "4004_batched_gemm_bf16", "4006_gemm_a16w16",     # 全部 case
    "4008_gemm_a16w4",                                 # 部分 case
}

TASK_DIR_RE = re.compile(r"^(?P<alias>.+?)_l(?P<level>\d+)_p(?P<pid>\d+)$")


def parse_last_json(text: str) -> dict | None:
    """audit_model_class 每个 stage 结束都 print 一次 JSON，最后一次最完整。
    这里反复 raw_decode，取最后一个成功解析的对象。"""
    dec = json.JSONDecoder()
    idx, last = 0, None
    while idx < len(text):
        start = text.find("{", idx)
        if start < 0:
            break
        try:
            obj, end = dec.raw_decode(text, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(obj, dict):
            last = obj
        idx = end
    return last


def baseline_impls(task: str) -> set[str]:
    f = PRIV / task / "baseline.json"
    if not f.exists():
        return set()
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return set()
    return {b.get("impl") for b in (d.get("baselines") or []) if b.get("impl")}


def discover_tasks(run_root: Path, manifest: Path | None) -> list[tuple[str, str]]:
    """返回 [(task_id, 产物目录名)]。优先用 manifest；manifest 缺失、解析失败或没有
    tasks 时**回退**到按产物目录名反推（曾因空 manifest 直接返回空列表而静默漏题）。"""
    if manifest and manifest.exists():
        try:
            import yaml  # 只在需要 manifest 时依赖
            payload = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
            tasks = payload.get("tasks") or []
            if tasks:
                pairs = []
                for t in tasks:
                    alias = str(t["alias"])
                    pairs.append((alias, f"{alias}_l{int(t['level'])}_p{int(t['problem_id'])}"))
                return pairs
            print(f"[warn] manifest 里没有 tasks，改用目录名反推: {manifest}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 读 manifest 失败（改用目录名反推）: {exc}", file=sys.stderr)
    pairs = []
    for p in sorted(run_root.iterdir()):
        if not p.is_dir():
            continue
        m = TASK_DIR_RE.match(p.name)
        if m:
            pairs.append((m.group("alias"), p.name))
    return pairs


def run_one(task: str, submission: Path, timeout: int, skip_perf: bool) -> dict:
    cmd = [sys.executable, str(AUDIT), "--task", task, "--submission", str(submission)]
    if skip_perf:
        cmd.append("--skip-perf")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"task": task, "ok": False, "reason": f"超时（>{timeout}s）"}
    report = parse_last_json(r.stdout or "")
    if not report:
        tail = ((r.stderr or "") + (r.stdout or ""))[-600:]
        return {"task": task, "ok": False, "reason": f"未拿到 JSON 报告（exit {r.returncode}）",
                "tail": tail}
    stages = report.get("stages") or {}
    corr = stages.get("correctness") or {}
    perf = stages.get("performance") or {}

    # 按 impl 归拢 speedup
    by_impl: dict[str, list[dict]] = {}
    for s in perf.get("speedups") or []:
        by_impl.setdefault(str(s.get("impl")), []).append(s)
    per_impl = {}
    for impl, rows in by_impl.items():
        vals = [float(x["speedup"]) for x in rows if isinstance(x.get("speedup"), (int, float))]
        if not vals:
            continue
        per_impl[impl] = {
            "case_count": len(vals),
            "mean_speedup": sum(vals) / len(vals),
            "median_speedup": statistics.median(vals),
            "min_speedup": min(vals),
            "max_speedup": max(vals),
            "cases": [{"case": x["case"], "median_us": x.get("median_us"),
                       "baseline_us": x.get("baseline_us"), "speedup": x.get("speedup")}
                      for x in rows],
        }

    return {
        "task": task,
        "ok": True,
        "submission": str(submission),
        "static_audit_passed": (stages.get("static_audit") or {}).get("passed"),
        "correctness_passed": corr.get("passed"),
        "correctness_case_count": corr.get("case_count"),
        "correctness_failures": [f.get("case") for f in (corr.get("failures") or [])],
        "baseline_status": perf.get("baseline_status"),
        "timing": perf.get("timing"),
        "by_impl": per_impl,
        "aiter_baseline_is_mi_fallback": task in AITER_MI_FALLBACK,
        "exit_code": r.returncode,
    }


def fmt(v, nd=2):
    return "-" if v is None else f"{float(v):.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", required=True, type=Path,
                    help="在线跑的产物根目录（含 <task>_l<L>_p<P>/best_code.py）")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                    help="任务清单（缺省用 kernelbench_compat/manifest_dcukb.yaml）")
    ap.add_argument("--impl", default="aiter", help="作为主指标汇报的基线实现（默认 aiter）")
    ap.add_argument("--tasks", default="", help="只审这些题（逗号分隔）")
    ap.add_argument("--include-no-aiter", action="store_true",
                    help="连没有 aiter 基线的题也审（只能得到 torch_eager 口径）")
    ap.add_argument("--limit", type=int, default=0, help="最多审几题（0=不限）")
    ap.add_argument("--timeout", type=int, default=1800, help="单题超时秒数")
    ap.add_argument("--skip-perf", action="store_true", help="只做正确性，不计时")
    ap.add_argument("--force", action="store_true", help="忽略缓存重算")
    ap.add_argument("--dry-run", action="store_true", help="只列将要审的题")
    args = ap.parse_args()

    run_root: Path = args.run_root
    if not run_root.is_dir():
        print(f"[FAIL] run-root 不存在: {run_root}", file=sys.stderr)
        return 2
    cache_dir = run_root / "offline_audit"

    only = {s.strip() for s in args.tasks.split(",") if s.strip()}
    pairs = discover_tasks(run_root, args.manifest)
    if only:
        pairs = [p for p in pairs if p[0] in only]

    plan, skipped = [], []
    for task, dirname in pairs:
        sub = run_root / dirname / "best_code.py"
        if not sub.exists():
            skipped.append((task, "无 best_code.py（该题没跑或未产出）"))
            continue
        impls = baseline_impls(task)
        if args.impl not in impls and not args.include_no_aiter:
            skipped.append((task, f"无 {args.impl} 基线（impls={sorted(impls) or '空'}）"))
            continue
        plan.append((task, dirname, sub))

    if args.limit:
        plan = plan[: args.limit]

    print(f"run-root : {run_root}")
    print(f"待审     : {len(plan)} 题     跳过: {len(skipped)} 题")
    if skipped:
        reasons: dict[str, int] = {}
        for _, why in skipped:
            key = why.split("（")[0]
            reasons[key] = reasons.get(key, 0) + 1
        for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"    - {k}: {v} 题")
    if args.dry_run:
        for task, _, sub in plan:
            print(f"    {task:46} <- {sub}")
        return 0
    if not plan:
        print("没有需要审的题。")
        return 0

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[FAIL] 无法创建缓存目录 {cache_dir}: {exc}\n"
              f"       （--dry-run 不需要写权限；真正审题时 run-root 必须可写）", file=sys.stderr)
        return 2

    results, failures = [], 0
    for i, (task, dirname, sub) in enumerate(plan, 1):
        cache = cache_dir / f"{task}.json"
        if cache.exists() and not args.force:
            try:
                results.append(json.loads(cache.read_text(encoding="utf-8")))
                print(f"[{i}/{len(plan)}] {task:46} 用缓存")
                continue
            except Exception:
                pass
        print(f"[{i}/{len(plan)}] {task:46} 审...", flush=True)
        res = run_one(task, sub, args.timeout, args.skip_perf)
        if not res.get("ok"):
            failures += 1
            print(f"          [FAIL] {res.get('reason')}")
        else:
            got = res["by_impl"].get(args.impl)
            head = f"{args.impl} {fmt(got['mean_speedup'])}x（{got['case_count']} case）" if got else f"无 {args.impl} 数据"
            print(f"          correct={res['correctness_passed']}  {head}")
        cache.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(res)

    # ---- 汇总 ----
    rows = [r for r in results if r.get("ok")]
    summary = {
        "run_root": str(run_root),
        "headline_impl": args.impl,
        "audited": len(results),
        "skipped": [{"task": t, "reason": w} for t, w in skipped],
        "rows": [],
        "aggregates": {},
    }
    for r in sorted(rows, key=lambda x: x["task"]):
        got = r["by_impl"].get(args.impl) or {}
        eager = r["by_impl"].get("torch_eager") or {}
        summary["rows"].append({
            "task": r["task"],
            "correctness_passed": r["correctness_passed"],
            "aiter_mean_speedup": got.get("mean_speedup"),
            "aiter_case_count": got.get("case_count"),
            "aiter_is_mi_fallback": r.get("aiter_baseline_is_mi_fallback"),
            "torch_eager_mean_speedup": eager.get("mean_speedup"),
            "measurement_impls": sorted(r["by_impl"].keys()),
        })

    sp = [x["aiter_mean_speedup"] for x in summary["rows"]
          if isinstance(x.get("aiter_mean_speedup"), (int, float))]
    summary["aggregates"] = {
        "audited": len(rows),
        "correctness_passed": sum(1 for r in rows if r["correctness_passed"]),
        f"{args.impl}_measured": len(sp),
        f"{args.impl}_mean_speedup": (sum(sp) / len(sp)) if sp else None,
        f"{args.impl}_median_speedup": statistics.median(sp) if sp else None,
        f"{args.impl}_max_speedup": max(sp) if sp else None,
        "mi_fallback_count": sum(1 for x in summary["rows"] if x.get("aiter_is_mi_fallback")),
        "_note": ("口径：离线终审，隐藏 case 用 task.yaml 分档容差；perf 用 private/<id>/perf_cases.json "
                  "的形状与计时参数，分母为 baseline.json 里该 impl 的实测耗时。"
                  "与在线（KernelBench 1e-4 + torch_eager 分母）不可混比。"
                  "aiter_is_mi_fallback=True 的题分母偏大 → 加速比偏高，引用须标注。"),
    }
    out = run_root / "offline_audit.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 108)
    print(f"离线终审汇总（分母 = {args.impl}）")
    print("=" * 108)
    print(f"{'题':46} {'correct':>8} {args.impl + ' speedup':>16} {'MI兜底':>8} {'torch_eager':>12}")
    for x in summary["rows"]:
        print(f"{x['task']:46} {str(x['correctness_passed']):>8} "
              f"{fmt(x['aiter_mean_speedup']):>16} "
              f"{('是' if x['aiter_is_mi_fallback'] else ''):>8} "
              f"{fmt(x['torch_eager_mean_speedup']):>12}")
    a = summary["aggregates"]
    print("-" * 108)
    print(f"审了 {a['audited']} 题；正确性通过 {a['correctness_passed']}；"
          f"拿到 {args.impl} speedup 的 {a[f'{args.impl}_measured']} 题")
    if a[f"{args.impl}_mean_speedup"] is not None:
        print(f"{args.impl} speedup: 均值 {a[f'{args.impl}_mean_speedup']:.2f}x   "
              f"中位 {a[f'{args.impl}_median_speedup']:.2f}x   最大 {a[f'{args.impl}_max_speedup']:.2f}x")
    print(f"其中 MI 兜底基线 {a['mi_fallback_count']} 题（分母偏大，加速比偏高，须标注）")
    print(f"\n-> {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
