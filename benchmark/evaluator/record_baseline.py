#!/usr/bin/env python3
"""record_baseline.py — 采集 aiter 官方实现的性能基线（离线终审用）。

对 `private/<id>/perf_cases.json` 的每个 case：
  1. 用（继承链上）任务的 reference.py::make_inputs 生成输入
  2. 用 task.yaml 的容差校验该 case 的 reference 输出，作为 golden
  3. 调用 private/<id>/aiter_impl.py 声明的 aiter 官方实现
  4. **先校验 aiter 输出与 golden 一致**（不一致则拒绝记录，避免记下错误基线）
  5. cuda event 计时，按 perf_cases 的 timing 配置预热/重复，取中位数
  6. 写入 private/<id>/baseline.json 的 baselines[]

baselines 的消费方是 audit_model_class.py，它按 `case` 名匹配并算 speedup。

用法（DCU 真机，需 DTK 环境 + aiter 源码已在 PYTHONPATH）：
    python record_baseline.py --task 1002_paged_attention \
        --aiter-root /root/dcudeploy/aiter_shim

退出码：0 = 采集成功；1 = 校验失败；2 = 用法/配置错误。
"""

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch
import yaml

# 与离线终审共用同一套 case 解析（case_inputs / case_init_kwargs /
# resolve_tolerance_limit），避免「baseline 的输入口径」与「终审的输入口径」
# 各写一份而漂移。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_model_class import (  # noqa: E402
    case_init_kwargs,
    case_inputs,
    check_close,
    move_inputs_to_device,
    resolve_tolerance_limit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS = REPO_ROOT / "benchmark" / "tasks"
PRIV = REPO_ROOT / "benchmark" / "private"

# 每个 case 重复计时前额外预热轮数（perf_cases.timing.warmup_iters 之外的保险）
EXTRA_WARMUP = 3


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case_file(task_id: str, kind: str) -> tuple[Path, str]:
    """返回 (case 文件路径, 提供输入生成器的 task_id)。支持 {"inherit": <task_id>}。"""
    path = PRIV / task_id / f"{kind}_cases.json"
    if not path.exists():
        print(f"[baseline] 缺少 {path}", file=sys.stderr)
        sys.exit(2)
    data = json.loads(path.read_text(encoding="utf-8"))
    source = data.get("inherit")
    if source:
        target = PRIV / source / f"{kind}_cases.json"
        if not target.exists():
            print(f"[baseline] {path} 继承的 {target} 不存在", file=sys.stderr)
            sys.exit(2)
        return target, source
    return path, task_id


def aiter_impl_file(task_id: str, inherit_source: str | None) -> Path:
    """aiter 实现适配器：优先本题，其次继承源。"""
    for tid in filter(None, [task_id, inherit_source]):
        path = PRIV / tid / "aiter_impl.py"
        if path.exists():
            return path
    print(
        f"[baseline] 找不到 private/{task_id}/aiter_impl.py"
        + (f"（或继承源 private/{inherit_source}/）" if inherit_source else "")
        + "：需为该算子提供 aiter 调用适配器",
        file=sys.stderr,
    )
    sys.exit(2)


def reference_output(ref_mod, inputs: list, init_kwargs: dict):
    """兼容两种 reference 形态：函数式 reference(...) 或 model_class 的 Model(...)。

    model_class 用**关键字**构造（与终审同为 case_init_kwargs 的产物），
    避免稀疏构造参数在位置式下错位。
    """
    if hasattr(ref_mod, "reference"):
        return ref_mod.reference(*inputs)
    if hasattr(ref_mod, "Model"):
        return ref_mod.Model(**init_kwargs)(*inputs)
    raise RuntimeError("reference.py 既没有 reference() 也没有 Model()")


def tolerances(task_dir: Path) -> dict:
    spec = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    return spec["tolerance"]


def time_callable(fn, warmup: int, repeat: int) -> tuple[float, float]:
    """cuda event 计时，返回 (中位数 us, 均值 us)。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)  # ms -> us
    return statistics.median(times), statistics.fmean(times)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, help="任务 id，如 1002_paged_attention")
    ap.add_argument("--impl", default="aiter", help="基线实现标识，写入 baselines[].impl")
    ap.add_argument("--dry-run", action="store_true", help="只校验不写回 baseline.json")
    args = ap.parse_args()

    task_id = args.task
    task_dir = TASKS / task_id
    if not task_dir.exists():
        print(f"[baseline] 任务目录不存在: {task_dir}", file=sys.stderr)
        return 2
    perf_path, gen_task_id = case_file(task_id, "perf")
    perf = json.loads(perf_path.read_text(encoding="utf-8"))
    impl_path = aiter_impl_file(task_id, gen_task_id if gen_task_id != task_id else None)

    if not torch.cuda.is_available():
        print("[baseline] 需要 CUDA（DCU）设备", file=sys.stderr)
        return 2

    # 语义依据：正式任务目录的 reference.py。model_class 形态有 Model；
    # reference_output() 也兼容函数式 reference()——但那种形态要靠
    # io.init_inputs 声明或 Model 签名才能解析构造参数，否则会无参构造。
    ref_mod = load_module(task_dir / "reference.py", f"baseline_ref_{task_id}")
    gen_mod = load_module(TASKS / gen_task_id / "reference.py", f"baseline_gen_{gen_task_id}")
    impl_mod = load_module(impl_path, f"baseline_impl_{task_id}")

    tol = tolerances(task_dir)
    timing = perf.get("timing") or {}
    warmup = int(timing.get("warmup_iters", 10))
    repeat = int(timing.get("repeat_iters", 30))
    reduction = timing.get("reduction", "median")

    print(f"[baseline] task       = {task_id}")
    print(f"[baseline] perf cases = {perf_path}")
    print(f"[baseline] 输入生成器 = tasks/{gen_task_id}/reference.py::make_inputs")
    print(f"[baseline] aiter 适配 = {impl_path.relative_to(REPO_ROOT)}")
    print(f"[baseline] 计时       = warmup {warmup} + repeat {repeat} ({reduction})")
    print()

    device = torch.device("cuda")
    baselines, failures = [], []

    for case in perf["cases"]:
        name = case["name"]
        inputs = move_inputs_to_device(case_inputs(gen_mod, case), device)
        init_kwargs = case_init_kwargs(task_dir, ref_mod, case)

        # golden（按 task.yaml 容差校验用）。容差块选取与终审共用同一口径
        # （输出 dtype 优先），避免基线与终审各挑一套容差。
        expected = reference_output(ref_mod, inputs, init_kwargs)

        # aiter 官方实现。适配器契约：run(inputs, init_kwargs: dict, device)
        # —— 按名取参（原 dev 分支的适配器按位置取 init_args[0]，需改成
        # init_kwargs["head_size"] 之类；稀疏构造参数下位置式不安全）。
        out, ctx = impl_mod.run(inputs, init_kwargs, device)

        # 数值判定与终审共用 check_close（分段感知）；未声明 tolerance_segments
        # 的题行为与原来的整体 allclose 完全一致。
        if out is not None and out.shape == expected.shape and out.dtype == expected.dtype:
            ok, detail = check_close(out, expected, tol, str(case.get("dtype", "")),
                                     ref_mod=ref_mod, init_kwargs=init_kwargs,
                                     task_dir=task_dir, inputs=inputs)
        else:
            ok, detail = False, {"mode": "shape-or-dtype", "max_diff": None}
        if not ok:
            failures.append({"case": name, "max_diff": detail.get("max_diff"),
                             "detail": detail})
            print(f"  [FAIL] {name}: 与 reference 不一致 "
                  f"(max_diff={detail.get('max_diff')}, mode={detail['mode']})")
            continue

        median_us, mean_us = time_callable(lambda: impl_mod.run(inputs, init_kwargs, device),
                                           warmup + EXTRA_WARMUP, repeat)
        baselines.append({
            "case": name,
            "us": round(median_us, 3),
            "mean_us": round(mean_us, 3),
            "impl": args.impl,
            "timing": {"warmup_iters": warmup, "repeat_iters": repeat, "reduction": "median",
                       "method": "cuda_event"},
            "max_diff_vs_reference": round(detail.get("max_diff") or 0.0, 6),
            "compare_mode": detail["mode"],
        })
        print(f"  [OK]   {name}: median {median_us:.1f} us  (mean {mean_us:.1f} us)")

    print()
    if failures:
        print(f"[baseline] {len(failures)} 个 case 校验失败，拒绝写入基线：{failures}", file=sys.stderr)
        return 1
    if not baselines:
        print("[baseline] 没有采集到任何基线", file=sys.stderr)
        return 1

    baseline_path = PRIV / task_id / "baseline.json"
    doc = json.loads(baseline_path.read_text(encoding="utf-8"))
    if args.dry_run:
        print(f"[baseline] dry-run：将写入 {baseline_path}")
        print(json.dumps(baselines, ensure_ascii=False, indent=2))
        return 0

    doc["status"] = "measured"
    doc["baselines"] = baselines
    doc["environment"] = {
        "device": torch.cuda.get_device_name(0),
        "gcn_arch": torch.cuda.get_device_properties(0).gcnArchName,
        "torch": torch.__version__,
    }
    import triton
    doc["environment"]["triton"] = triton.__version__
    baseline_path.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[baseline] 已写入 {baseline_path}（{len(baselines)} 条）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
