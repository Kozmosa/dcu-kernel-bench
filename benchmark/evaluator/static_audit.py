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
    # 设备原语库（scan / sort / reduce / segmented 原语）。HIP track 上
    # rocprim / hipcub 与 cutlass 同类，都是"把核心计算整段交给库"的绕过路径；
    # 实测：一份用 hipcub.DeviceScan.InclusiveScan + rocprim.inclusive_scan 完成
    # 全部 scan 的提交，在补这条之前对 8001_cumsum 报 passed=true、零违规。
    # 只列 ROCm 侧：本评测集只在 DCU 上跑，CUDA 的 thrust / cub 无意义，
    # 而 `cub` 会以子串形式误伤 cuboid / incumbent 等词（3D 卷积题很常见）。
    "rocprim", "hipcub",
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


def _strip_python_docstrings(text: str) -> str:
    """用 ast 定位并整行清空 docstring（module/class/function 首个字符串常量）。

    model_class 任务的题面 docstring 会逐字列出 forbidden 禁用项，属于规则
    声明而非调用；不清空会导致所有框架生成候选被误杀。清空保留行号。
    """
    import ast

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    spans: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            spans.append((body[0].lineno, body[0].end_lineno))
    lines = text.splitlines(keepends=True)
    for lo, hi in spans:
        for i in range(lo - 1, min(hi, len(lines))):
            lines[i] = "\n" if lines[i].endswith("\n") else ""
    return "".join(lines)


def _symbol_matches(name: str, pattern: str) -> bool:
    """Match identifiers/namespaces, never substrings of user kernel names."""
    name, pattern = name.lower(), pattern.lower()
    if pattern.startswith("."):
        return name.rsplit(".", 1)[-1] == pattern.lstrip(".").rstrip("(")
    return name == pattern or name.startswith(pattern + ".") or (
        "." not in pattern and name.rsplit(".", 1)[-1] == pattern
    )


def _python_hits(path: Path, text: str, patterns: list[str]) -> list[dict]:
    import ast

    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [{"file": str(path), "line": exc.lineno, "symbol": "syntax_error"}]

    aliases = {}
    definitions = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split(".")[0]] = item.name if item.asname else item.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            for item in node.names:
                aliases[item.asname or item.name] = (node.module or "") + "." + item.name
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions[node.name] = node

    def dotted(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            base = dotted(node.value)
            return base + "." + node.attr if base else node.attr
        if isinstance(node, ast.Subscript):  # Triton kernel[grid](...)
            return dotted(node.value)
        return ""

    # Module-level code is executable. Function bodies are inspected only when
    # referenced from it or from ModelNew.__init__/forward (including aliases,
    # inherited classes and helpers passed as values, not just direct calls).
    pending = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    if "ModelNew" in definitions:
        pending.append(definitions["ModelNew"])
    else:
        # Callable submissions without a ModelNew keep conservative whole-file
        # checking: their entry point is task-specific.
        pending.extend(definitions.values())
    visited = set()
    hits = set()
    while pending:
        root = pending.pop()
        if id(root) in visited:
            continue
        visited.add(id(root))
        if isinstance(root, ast.ClassDef):
            methods = {n.name: n for n in root.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            for name, method in methods.items():
                definitions[root.name + "." + name] = method
                # self/cls method references are resolved conservatively across
                # classes, so inherited helpers cannot silently escape the scan.
                definitions.setdefault("self." + name, method)
                definitions.setdefault("cls." + name, method)
            pending.extend(n for n in root.body if n not in methods.values())
            pending.extend(methods[n] for n in ("__init__", "forward", "__call__") if n in methods)
            pending.extend(root.bases + root.decorator_list)
            continue
        for node in ast.walk(root):
            if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
                name = dotted(node)
                raw = ast.unparse(node) if isinstance(node, (ast.Name, ast.Attribute)) else ""
                if raw in definitions:
                    pending.append(definitions[raw])
                if name in definitions:
                    pending.append(definitions[name])
                for pattern in patterns:
                    if _symbol_matches(name, pattern):
                        hits.add((node.lineno, pattern))
            # Embedded HIP/C++ source strings still need a native-symbol audit.
            # Docstrings and ordinary messages are not executable source.
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if re.search(r"#include|__global__|PYBIND11_MODULE|\b(?:hipcub|rocprim)::", node.value):
                    for hit in _native_hits(path, node.value, patterns):
                        hits.add((node.lineno + hit["line"] - 1, hit["symbol"]))
            if isinstance(node, ast.Call) and dotted(node.func) in {"getattr", "eval", "exec"}:
                # Dynamic dispatch cannot be proved dead; inspect all definitions
                # instead of exempting potentially reachable forbidden code.
                pending.extend(definitions.values())
    return [{"file": str(path), "line": line, "symbol": symbol} for line, symbol in sorted(hits)]


def _native_hits(path: Path, text: str, patterns: list[str]) -> list[dict]:
    # Keep line numbers when deleting multiline comments.
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group().count("\n"), text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    hits = []
    for pattern in patterns:
        # Library APIs often use a prefix (rocblas_sgemm, hipblasGemmEx).
        suffix = r"[A-Za-z0-9_]*" if pattern.lower() in GLOBAL_FORBIDDEN else r"(?![A-Za-z0-9_])"
        regex = r"(?<![A-Za-z0-9_])" + re.escape(pattern) + suffix
        for match in re.finditer(regex, text, flags=re.I):
            hits.append({"file": str(path), "line": text.count("\n", 0, match.start()) + 1, "symbol": pattern})
    return hits


def audit_file(path: Path, patterns: list[str]) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".py":
        return _python_hits(path, text, patterns)
    return _native_hits(path, text, patterns)


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
