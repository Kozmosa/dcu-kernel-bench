#!/usr/bin/env python3
"""run_eval.py — 单任务评测管线骨架。

阶段：静态审计 → 编译 → 公开 Case → 隐藏 Case → 边界 Case → 性能计时。
任一隐藏正确性 Case 失败则不进入性能评分。

用法（DCU 真机上）：
    python run_eval.py --task benchmark/tasks/0001_vecadd \
                       --submission <agent 提交的 kernel 文件或目录>

当前状态：骨架。static audit 可直接运行；
compile/correctness/performance 依赖 DTK 环境（hipcc、DTK 版 PyTorch），
在真机上落实 environment.yaml 后补全标记为 TODO(dcu) 的部分。
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def stage_static_audit(task_dir: Path, submission: Path) -> bool:
    files = [submission] if submission.is_file() else sorted(submission.rglob("*.*"))
    cmd = [sys.executable, str(Path(__file__).parent / "static_audit.py"),
           str(task_dir), *[str(f) for f in files]]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    return r.returncode == 0


def stage_compile(task_dir: Path, submission: Path) -> bool:
    # TODO(dcu): hip track —— 用 environment.yaml 的 flags 调 hipcc 编译
    #   submission kernel.hip + 通用 harness（Host 分配、数据搬运、结果回读）
    # triton track —— import kernel.py 触发 JIT 预热即可
    raise NotImplementedError("需在 DCU 真机环境补全（hipcc / triton JIT）")


def stage_correctness(task_dir: Path, case_sets: list[str]) -> bool:
    # TODO(dcu): 按 tasks/<id>/public_cases.json 与 private/<id>/hidden_cases.json
    #   生成输入 → 跑 submission 与 reference.py → 按 task.yaml 容限比较
    raise NotImplementedError("需在 DCU 真机环境补全")


def stage_performance(task_dir: Path) -> dict:
    # TODO(dcu): private/<id>/perf_cases.json 输入，预热 + 重复取中位数，
    #   与 private/<id>/baseline.json 对比得出 speedup
    raise NotImplementedError("需在 DCU 真机环境补全")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, type=Path)
    ap.add_argument("--submission", required=True, type=Path)
    ap.add_argument("--skip-perf", action="store_true")
    args = ap.parse_args()

    result = {"task": str(args.task), "stages": {}}

    ok = stage_static_audit(args.task, args.submission)
    result["stages"]["static_audit"] = ok
    if not ok:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    for stage in (stage_compile, stage_correctness):
        try:
            ok = stage(args.task, args.submission)  # type: ignore[arg-type]
        except NotImplementedError as e:
            print(f"[run_eval] {stage.__name__}: {e}")
            ok = None
        result["stages"][stage.__name__] = ok
        if ok is False:
            break

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
