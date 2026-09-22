#!/usr/bin/env python3
"""static_audit.py — 提交代码静态审计（CPU 即可运行）。

检查 Agent 提交文件中是否出现 task.yaml `forbidden` 列表里的符号
（默认含 MIOpen / rocBLAS / hipBLASLt / hipDNN 等闭源库），
发现即判违规，不进入编译阶段。

用法：
    python static_audit.py benchmark/tasks/0001_vecadd <提交文件> [<提交文件>...]
退出码：0 = 通过，1 = 违规，2 = 用法/配置错误。
"""

import json
import re
import sys
from pathlib import Path

# 全局禁用（对所有任务生效），与 task.yaml 的 forbidden 取并集
GLOBAL_FORBIDDEN = [
    "rocblas", "hipblas", "hipblaslt", "miopen", "hipdnn",
    "cublas", "cudnn", "cutlass",
]


def load_task_forbidden(task_dir: Path) -> list[str]:
    """从 task.yaml 读 forbidden 列表（避免引入 yaml 依赖，做最小解析）。"""
    task_yaml = task_dir / "task.yaml"
    if not task_yaml.exists():
        print(f"[audit] 缺少 {task_yaml}", file=sys.stderr)
        sys.exit(2)
    forbidden, in_block = [], False
    for line in task_yaml.read_text(encoding="utf-8").splitlines():
        if re.match(r"^forbidden:\s*$", line):
            in_block = True
            continue
        if in_block:
            m = re.match(r"^\s+-\s+(\S+)\s*$", line)
            if m:
                forbidden.append(m.group(1))
            elif line.strip() and not line.startswith(" "):
                break
    return forbidden


def audit_file(path: Path, patterns: list[str]) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    # 去掉注释，减少注释里提到库名造成的误报
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"#[^\n]*", "", text) if path.suffix == ".py" else text
    hits = []
    for pat in patterns:
        for m in re.finditer(re.escape(pat), text, flags=re.I):
            line_no = text.count("\n", 0, m.start()) + 1
            hits.append({"file": str(path), "line": line_no, "symbol": pat})
    return hits


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    task_dir = Path(sys.argv[1])
    files = [Path(p) for p in sys.argv[2:]]
    patterns = sorted(set(GLOBAL_FORBIDDEN) | set(load_task_forbidden(task_dir)))

    all_hits = []
    for f in files:
        if not f.exists():
            print(f"[audit] 文件不存在: {f}", file=sys.stderr)
            return 2
        all_hits.extend(audit_file(f, patterns))

    report = {"task_dir": str(task_dir), "passed": not all_hits, "violations": all_hits}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not all_hits else 1


if __name__ == "__main__":
    sys.exit(main())
