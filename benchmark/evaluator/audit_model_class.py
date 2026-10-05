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


def load_init_names(task_dir: Path) -> list | None:
    """task.yaml `io.init_inputs` 声明的构造参数名，按签名顺序。

    返回 None 表示**未声明**（老任务）；返回 [] 表示**显式声明无超参**
    （上游 2009_softmax 等就是这么写的，应无参构造 Model）。
    对齐上游 main 的 audit_model_class.py 契约。
    """
    spec = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    io = spec.get("io") or {}
    declared = io.get("init_inputs")
    if declared is None:
        return None
    return [e["name"] for e in declared if isinstance(e, dict) and "name" in e]


def resolve_tolerance_limit(tol: dict, case_dtype: str, out_dtype) -> dict:
    """按输出 dtype 优先挑容差块，其次 case 的输入 dtype，最后任一含 atol 的块。

    为什么要看输出 dtype：量化类任务（4xxx 号段）输出是 int8/int32，需要
    atol=0 之类的独立容差块，不能沿用输入 dtype 的半精度容差。
    """
    out_key = str(out_dtype).replace("torch.", "")
    for key in (out_key, case_dtype):
        limit = tol.get(key)
        if isinstance(limit, dict) and "atol" in limit:
            return limit
    for limit in tol.values():
        if isinstance(limit, dict) and "atol" in limit:
            return limit
    raise KeyError(f"task.yaml tolerance 缺少数值条目（尝试过 {out_key}/{case_dtype}）")


def move_inputs_to_device(inputs, device) -> list:
    """把输入搬到设备；**非张量元素原样透传**。

    有的题面有可选输入：例如 `1022_pa_prefill` 的 `alibi_slopes` 在 use_alibi=False 时
    由 make_inputs 返回 `None`。KernelBench 上游的 `_process_input_tensor` 同样是
    "非张量原样返回"；这里保持一致——否则 `[t.to(device) for t in inputs]` 会在
    `None` 上抛 AttributeError，让整题的基线采集与终审直接失败（不是数值错，是崩）。
    """
    return [t.to(device) if isinstance(t, torch.Tensor) else t for t in inputs]


def stage_static(task_dir: Path, submission: Path) -> dict:
    r = subprocess.run(
        [sys.executable, str(AUDIT), str(task_dir), str(submission)],
        capture_output=True, text=True,
    )
    return {"passed": r.returncode == 0, "detail": json.loads(r.stdout) if r.stdout else {"stderr": r.stderr}}


def case_inputs(gen_mod, case: dict) -> list:
    """按 case 字段调用任务的输入生成器（字段名即生成器参数名）。

    生成器在（继承链上）任务的 reference.py::make_inputs。带 seq_lens 这类算子
    特有参数的生成器也直接透传字段即可，不做算子特定的特殊处理。
    """
    fields = {k: v for k, v in case.items() if k not in ("name", "init_inputs")}
    return gen_mod.make_inputs(**fields)


def _signature_param_names(ref_mod) -> list:
    """Model.__init__ 的参数名（排除 self）；拿不到签名时返回空表。"""
    import inspect

    model = getattr(ref_mod, "Model", None)
    if model is None:
        return []
    try:
        return [p for p in inspect.signature(model.__init__).parameters if p != "self"]
    except (TypeError, ValueError):
        return []


def case_init_kwargs(task_dir: Path, ref_mod, case: dict) -> dict:
    """Model/ModelNew 的**关键字**构造参数——按名绑定，避免位置错位。

    为什么必须是 kwargs：构造参数往往是稀疏给出的（case 只写其中几个），
    位置式列表一旦 append 就会占用后面参数的槽位。实测 1020_mla_decode_rope：
    `get_init_inputs()=[512,64,64]`（共 5 个参数，sm_scale/is_neox_style 走默认），
    case 只给了 `is_neox_style` → 位置式会把它放到 `sm_scale` 的位置上，
    **静默算错**。按名绑定不存在这个问题（上游 main 的 `resolve_init_args` 同理）。

    取值来源：
      - `io.init_inputs` 已声明 → 只认这些名字（上游契约，声明即权威）
      - 未声明 → 退回 `Model.__init__` 签名里出现在 case 中的名字
    只传 case 里出现的名字，其余交给 Python 默认值。声明为 `io.init_inputs: []`
    的任务返回 {}，即无参构造。

    若 case 与声明/签名**没有任何公共字段**，回退到 `get_init_inputs()` 的默认值
    按签名位置补齐。这修的是上游的一处真实缺口：4006_gemm_a16w16 / 4007 /
    4008 的 case 只给 m/n/k（4008 还给了 group_size），而构造参数
    `in_features`/`out_features` 只在 `get_init_inputs()=[1024, 8192]` 里；
    上游 `resolve_init_args` 只取 case 公共字段，结果是 4006/4007 抛 KeyError、
    4008 拿到只有 group_size 的 kwargs 后 TypeError——三道 GEMM 题在真机上
    根本构造不出 Model。
    """
    declared = load_init_names(task_dir)
    if declared == []:
        return {}
    names = declared if declared is not None else _signature_param_names(ref_mod)
    kwargs = {name: case[name] for name in names if name in case}

    # 逐参数补齐：case 没给、但签名里有默认值的，用 get_init_inputs() 的值显式
    # 补上（值等价于 Python 默认值，只是让"缺必需参数"这类问题在这里暴露，
    # 而不是在 Model(...) 里变成 TypeError）。
    if hasattr(ref_mod, "get_init_inputs"):
        order = _signature_param_names(ref_mod)
        defaults = list(ref_mod.get_init_inputs())
        if order and defaults:
            for name, value in zip(order, defaults):
                kwargs.setdefault(name, value)
    return kwargs


def case_init_args(task_dir: Path, ref_mod, case: dict) -> list:
    """`case_init_kwargs` 的签名序列表形式（仅供按位置取参的 aiter 适配器使用）。

    ⚠️ 只在适配器确实按 `init_args[0]` 这类位置取值时用；位置式对稀疏参数
    天然不安全（见 case_init_kwargs 的说明）。优先让适配器接收 kwargs。
    """
    kwargs = case_init_kwargs(task_dir, ref_mod, case)
    order = load_init_names(task_dir) or _signature_param_names(ref_mod)
    return [kwargs[name] for name in order if name in kwargs]


def stage_correctness(task_dir: Path, gen_task_dir: Path, submission: Path, cases: list[dict], tol: dict) -> dict:
    ref_mod = load_module(task_dir / "reference.py", "audit_ref_model")
    cand_mod = load_module(submission, "audit_candidate")
    gen_mod = load_module(gen_task_dir / "reference.py", "audit_case_generator")

    device = torch.device("cuda")
    failures = []
    for case in cases:
        inputs = move_inputs_to_device(case_inputs(gen_mod, case), device)
        init_kwargs = case_init_kwargs(task_dir, ref_mod, case)

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


def stage_perf(task_dir: Path, gen_task_dir: Path, submission: Path, perf: dict, baseline: dict) -> dict:
    ref_mod = load_module(task_dir / "reference.py", "audit_perf_ref")
    cand_mod = load_module(submission, "audit_perf_candidate")
    gen_mod = load_module(gen_task_dir / "reference.py", "audit_perf_generator")
    device = torch.device("cuda")

    # 计时参数取自 perf_cases.json，必须与 record_baseline.py 一致；
    # 否则 speedup 的分子分母不是同一口径的测量。
    timing = perf.get("timing") or {}
    warmup = int(timing.get("warmup_iters", 5))
    repeat = int(timing.get("repeat_iters", 20))
    reduction = timing.get("reduction", "median")

    results = []
    for case in perf["cases"]:
        inputs = move_inputs_to_device(case_inputs(gen_mod, case), device)
        init_kwargs = case_init_kwargs(task_dir, ref_mod, case)
        model = cand_mod.ModelNew(**init_kwargs).to(device)

        for _ in range(warmup):
            model(*inputs)
        torch.cuda.synchronize()
        times = []
        for _ in range(repeat):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            model(*inputs)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000.0)  # ms -> us
        results.append({
            "case": case["name"],
            "median_us": statistics.median(times),
            "mean_us": statistics.fmean(times),
        })

    baselines = baseline.get("baselines") or []
    out = {
        "passed": True,
        "timing": {
            "warmup_iters": warmup,
            "repeat_iters": repeat,
            "reduction": reduction,
            "method": "cuda_event",
        },
        "cases": results,
        "baseline_status": baseline.get("status"),
    }
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
        report["stages"]["performance"] = stage_perf(task_dir, gen_dir, args.submission, perf, baseline)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
