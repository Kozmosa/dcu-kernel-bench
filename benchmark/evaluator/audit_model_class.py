# audit_model_class.py — model_class 候选（ModelNew 源文件）的离线终审
#
# 与在线（搜索循环内）KernelBench 语义互补，终审执行本评测集自己的标准：
#   阶段 1  静态审计：task.yaml forbidden + 全局闭源库黑名单（static_audit.py）
#   阶段 2  隐藏 Case 正确性：private/<id>/hidden_cases.json（支持 {"inherit": <task_id>}），
#           输入由继承任务的 make_inputs 生成，按 task.yaml 容差比较
#   阶段 3  性能测量：private/<id>/perf_cases.json，cuda event 计时取中位数，
#           baseline.json 有正式基线时输出 speedup
#
# 用法（DCU 真机，需 DTK 环境）：
#   python audit_model_class.py --task 1002_paged_attention --submission best_code.py
# 退出码：0 = 全部通过；1 = 任一阶段失败；2 = 用法/配置错误。

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS = REPO_ROOT / "benchmark" / "tasks"
PRIV = REPO_ROOT / "benchmark" / "private"
AUDIT = Path(__file__).parent / "static_audit.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_case_file(task_id: str, kind: str) -> tuple[Path, Path]:
    """返回 (case 文件路径, 输入生成器所在任务目录)。支持 inherit 引用。"""
    path = PRIV / task_id / f"{kind}_cases.json"
    if not path.exists():
        print(f"[audit] 缺少 {path}", file=sys.stderr)
        sys.exit(2)
    data = json.loads(path.read_text(encoding="utf-8"))
    source_task = data.get("inherit")
    if source_task:
        return PRIV / source_task / f"{kind}_cases.json", TASKS / source_task
    return path, TASKS / task_id


def load_tolerances(task_dir: Path) -> dict:
    spec = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    return spec["tolerance"]


def load_init_names(task_dir: Path) -> list:
    """io.init_inputs 的字段名列表（Model 构造参数名）；未声明时为空。"""
    spec = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    io = spec.get("io") or {}
    return [e["name"] for e in (io.get("init_inputs") or []) if isinstance(e, dict) and "name" in e]


def resolve_init_args(init_names: list, case: dict) -> dict:
    """按 task.yaml init_inputs 字段名从 case 取 Model/ModelNew 构造参数；
    无交集时回退 head_size（1002 系历史契约）。"""
    kwargs = {n: case[n] for n in init_names if n in case}
    if not kwargs and "head_size" in case:
        kwargs = {"head_size": case["head_size"]}
    if not kwargs:
        raise KeyError(
            f"case {case.get('name')} 与 io.init_inputs({init_names}) 无公共字段，无法构造 Model"
        )
    return kwargs


def resolve_tolerance_limit(tol: dict, case_dtype: str, out_dtype) -> dict:
    """tolerance 按输出 dtype 组织：优先输出 dtype 键（整数量化输出即 atol 0 块），
    其次 case 的输入 dtype 键，最后任一含 atol 的块。"""
    out_key = str(out_dtype).replace("torch.", "")
    for key in (out_key, case_dtype):
        limit = tol.get(key)
        if isinstance(limit, dict) and "atol" in limit:
            return limit
    for limit in tol.values():
        if isinstance(limit, dict) and "atol" in limit:
            return limit
    raise KeyError(f"task.yaml tolerance 缺少数值条目（尝试过 {out_key}/{case_dtype}）")


def stage_static(task_dir: Path, submission: Path) -> dict:
    r = subprocess.run(
        [sys.executable, str(AUDIT), str(task_dir), str(submission)],
        capture_output=True, text=True,
    )
    return {"passed": r.returncode == 0, "detail": json.loads(r.stdout) if r.stdout else {"stderr": r.stderr}}


def stage_correctness(task_dir: Path, gen_task_dir: Path, submission: Path, cases: list[dict], tol: dict) -> dict:
    ref_mod = load_module(task_dir / "reference.py", "audit_ref_model")
    cand_mod = load_module(submission, "audit_candidate")
    gen_mod = load_module(gen_task_dir / "reference.py", "audit_case_generator")

    device = torch.device("cuda")
    init_names = load_init_names(task_dir)
    failures = []
    for case in cases:
        fields = {k: v for k, v in case.items() if k != "name"}
        inputs = gen_mod.make_inputs(**fields)
        inputs = [t.to(device) for t in inputs]

        init_kwargs = resolve_init_args(init_names, case)
        expected = ref_mod.Model(**init_kwargs).to(device)(*inputs)
        actual = cand_mod.ModelNew(**init_kwargs).to(device)(*inputs)
        torch.cuda.synchronize()

        limit = resolve_tolerance_limit(tol, str(case.get("dtype", "")), expected.dtype)
        ok = (
            actual.shape == expected.shape
            and actual.dtype == expected.dtype
            and torch.allclose(actual.float(), expected.float(), atol=limit["atol"], rtol=limit["rtol"])
        )
        if not ok:
            diff = (actual.float() - expected.float()).abs().max().item() if actual.shape == expected.shape else None
            failures.append({"case": case["name"], "max_diff": diff})
    return {"passed": not failures, "case_count": len(cases), "failures": failures}


def stage_perf(task_dir: Path, gen_task_dir: Path, submission: Path, cases: list[dict], baseline: dict) -> dict:
    cand_mod = load_module(submission, "audit_perf_candidate")
    gen_mod = load_module(gen_task_dir / "reference.py", "audit_perf_generator")
    device = torch.device("cuda")
    init_names = load_init_names(task_dir)

    results = []
    for case in cases:
        fields = {k: v for k, v in case.items() if k != "name"}
        inputs = [t.to(device) for t in gen_mod.make_inputs(**fields)]
        model = cand_mod.ModelNew(**resolve_init_args(init_names, case)).to(device)

        for _ in range(5):
            model(*inputs)
        torch.cuda.synchronize()
        times = []
        for _ in range(20):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            model(*inputs)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000.0)  # ms -> us
        results.append({"case": case["name"], "median_us": statistics.median(times)})

    baselines = baseline.get("baselines") or []
    out = {"passed": True, "cases": results, "baseline_status": baseline.get("status")}
    if baselines:
        out["speedups"] = [
            {**r, "speedup": b["us"] / r["median_us"]}
            for r in results
            for b in baselines
            if b.get("case") == r["case"]
        ]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, help="任务 id，如 1002_paged_attention")
    ap.add_argument("--submission", required=True, type=Path, help="ModelNew 候选源文件")
    ap.add_argument("--skip-perf", action="store_true")
    args = ap.parse_args()

    task_dir = TASKS / args.task
    if not task_dir.exists() or not args.submission.exists():
        print(f"[audit] 任务目录或提交文件不存在: {task_dir} / {args.submission}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("[audit] 需要 CUDA（DCU）设备", file=sys.stderr)
        return 2

    report = {"task": args.task, "submission": str(args.submission), "stages": {}}

    static = stage_static(task_dir, args.submission)
    report["stages"]["static_audit"] = static
    if not static["passed"]:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    hidden_path, gen_dir = resolve_case_file(args.task, "hidden")
    hidden = json.loads(hidden_path.read_text(encoding="utf-8"))
    tol = load_tolerances(task_dir)
    correctness = stage_correctness(task_dir, gen_dir, args.submission, hidden["cases"], tol)
    report["stages"]["correctness"] = correctness
    if not correctness["passed"]:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    if not args.skip_perf:
        perf_path, gen_dir = resolve_case_file(args.task, "perf")
        perf = json.loads(perf_path.read_text(encoding="utf-8"))
        baseline = json.loads((PRIV / args.task / "baseline.json").read_text(encoding="utf-8"))
        report["stages"]["performance"] = stage_perf(task_dir, gen_dir, args.submission, perf["cases"], baseline)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
