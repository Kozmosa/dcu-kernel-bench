#!/usr/bin/env python3
"""static_audit.py — 提交代码静态审计（CPU 即可运行）。

两类检查，都判 `passed=false`，但 `kind` 不同、修法不同：

  forbidden  把核心计算交给闭源/原语库（task.yaml `forbidden` ∪ `GLOBAL_FORBIDDEN`）。
             这是"合规"问题：代码能跑，但不算自己写算子。
  platform   用了这台机器没有的符号（CUDA runtime API、CUDA 头、CUDART_* 常量、
             PTX 内联汇编、Triton 的 NVIDIA 命名空间）。这是"能不能编译"的问题：
             在 DCU（hipcc + AMD Triton）上必然编不过。
             注意 `at::cuda::` / `torch.cuda.` / `<ATen/cuda/...>` 在 ROCm 版 torch 里
             是**正确写法**，不在检查范围。

用法：
    python static_audit.py benchmark/tasks/0001_vecadd <提交文件> [<提交文件>...]
    python static_audit.py --no-platform <task_dir> <提交文件>    # 只查 forbidden
退出码：0 = 通过，1 = 违规（任一类），2 = 用法/配置错误。
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

# 平台符号：这台机器（DCU / gfx936，hipcc + AMD Triton）上没有的东西。
# 与 GLOBAL_FORBIDDEN 的区别：forbidden 是"能把活干完但不是你干的"，
# platform 是"根本编译不过 / 跑不起来"。实测依据（r3 数据）：
#   · `CUDART_INF_F` 未定义 → 1020_mla_decode_rope 直接编译失败；
#   · 模板自带的 `#include <cuda_runtime.h>` 若要真编译必须换成 <hip/hip_runtime.h>；
#   · 而 `__launch_bounds__`(48 次)、`warpSize`(108 次) 在 hipcc 下都能编过 → 不查。
# 每项：(kind 标签, 正则, 修法提示)。正则只在**原生源码**（.cu/.cpp/hip 字符串）里跑，
# Triton 那两条在 Python AST 里按解析后的全名匹配。
PLATFORM_PATTERNS = [
    ("cuda_include", r"#include\s*[<\"]cuda(?:_runtime|_runtime_api|_fp16|_bf16)?\.h[>\"]",
     "CUDA runtime 头在 hipcc 下不存在，改 #include <hip/hip_runtime.h>"),
    ("cuda_include", r"#include\s*[<\"]device_launch_parameters\.h[>\"]",
     "device_launch_parameters.h 是 CUDA 专有头，hipcc 下不需要"),
    ("cuda_runtime_api", r"\bcuda(?:Malloc|MallocManaged|Free|FreeHost|Memcpy|MemcpyAsync|Memset|"
                         r"StreamCreate|StreamCreateWithFlags|StreamDestroy|StreamSynchronize|"
                         r"StreamWaitEvent|EventCreate|EventCreateWithFlags|EventDestroy|"
                         r"EventSynchronize|EventRecord|EventElapsedTime|DeviceSynchronize|"
                         r"GetLastError|PeekAtLastError|GetErrorString|GetErrorName|LaunchKernel|"
                         r"SetDevice|GetDevice|GetDeviceCount|GetDeviceProperties|"
                         r"HostAlloc|HostGetDevicePointer|FuncSetAttribute|FuncGetAttributes)\w*",
     "CUDA runtime API 在 hipcc 下叫 hip*（cudaMalloc → hipMalloc 等）"),
    ("cuda_runtime_type", r"\bcuda(?:Stream_t|Event_t|Error_t|DeviceProp|MallocAsync)\b",
     "CUDA runtime 类型/枚举在 hipcc 下叫 hip*"),
    ("cudart_constant", r"\bCUDART_[A-Z0-9_]+\b",
     "CUDART_* 数学常量不是 hipcc 提供的，用 INFINITY / NAN / acosf(-1.0f)"),
    ("ptx_asm", r"asm\s*(?:volatile\s*)?\(\s*\"[^\"]*(?:shfl\.sync|ld\.global|st\.global|"
                r"mma\.sync|bar\.sync|cp\.async|\.reg\b)",
     "CUDA PTX 内联汇编在 AMD 后端不可用，用 __shfl_* / 内建函数重写"),
]

# Triton 侧的 NVIDIA 专有入口（在 Python AST 里按解析后的全名匹配）
PLATFORM_PYTHON_PATTERNS = [
    (".extra.cuda", "triton_cuda_namespace",
     "tl.extra.cuda 是 NVIDIA libdevice；AMD 后端用 triton.language.extra.libdevice"),
    ("inline_asm_elementwise", "triton_inline_asm",
     "tl.inline_asm_elementwise 里的 PTX 在 AMD 后端不可用"),
]


def _platform_native_hits(path: Path, text: str) -> list[dict]:
    """在原生源码里找 CUDA 专有头 / runtime API / CUDART 常量 / PTX。"""
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group().count("\n"), text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    hits = []
    for label, regex, hint in PLATFORM_PATTERNS:
        for match in re.finditer(regex, text):
            hits.append({"file": str(path), "line": text.count("\n", 0, match.start()) + 1,
                         "symbol": match.group().strip(), "kind": "platform",
                         "rule": label, "hint": hint})
    return hits


def _platform_python_hits(path: Path, name: str, lineno: int) -> list[dict]:
    """依据解析后的全名（含 import 别名）判断 Triton 侧平台错配。"""
    lowered = name.lower()
    hits = []
    for needle, label, hint in PLATFORM_PYTHON_PATTERNS:
        if needle in lowered:
            hits.append({"file": str(path), "line": lineno, "symbol": name,
                         "kind": "platform", "rule": label, "hint": hint})
    return hits


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
            m = re.match(r"^\s*-\s+(\S+?)(?:\s+#.*)?\s*$", line)
            if m:
                forbidden.append(m.group(1))
            elif line.strip() and not line.startswith(" "):
                break
    return forbidden


def _symbol_matches(name: str, pattern: str) -> bool:
    """Match identifiers/namespaces, never substrings of user kernel names."""
    name, pattern = name.lower(), pattern.lower().rstrip("()")
    if pattern.startswith("."):
        # These entries describe PyTorch tensor methods, not Triton language
        # primitives inside a handwritten kernel (e.g. tl.sort / tl.cat).
        return not name.startswith("triton.") and name.rsplit(".", 1)[-1] == pattern.lstrip(".")
    aliases = {"f": "torch.nn.functional", "functional": "torch.nn.functional",
               "nn": "torch.nn", "aten": "torch.ops.aten", "autograd": "torch.autograd"}
    first, sep, rest = pattern.partition(".")
    if sep and first in aliases:
        pattern = aliases[first] + "." + rest
    return name == pattern or name.startswith(pattern + ".") or (
        "." not in pattern and name.rsplit(".", 1)[-1] == pattern
    )


def _python_hits(path: Path, text: str, patterns: list[str], check_platform: bool = True) -> list[dict]:
    import ast

    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [{"file": str(path), "line": exc.lineno, "symbol": "syntax_error"}]

    docstrings = set()
    for owner in ast.walk(tree):
        if isinstance(owner, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = owner.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))
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
    # Defaults/decorators execute at definition time even when a helper body
    # is dead. Ignoring those would turn a false-positive fix into a bypass.
    def definition_setup(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.decorator_list + node.args.defaults + [n for n in node.args.kw_defaults if n is not None]
        if isinstance(node, ast.ClassDef):
            nodes = node.bases + node.decorator_list + [k.value for k in node.keywords]
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    nodes += definition_setup(child)
                else:
                    nodes.append(child)
            return nodes
        return []
    for definition in tree.body:
        pending.extend(definition_setup(definition))
    visited = set()
    hits = set()
    platform_hits: list[dict] = []

    def add_platform(entries):
        for entry in entries:
            if entry not in platform_hits:
                platform_hits.append(entry)

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
                if check_platform and name:
                    add_platform(_platform_python_hits(path, name, node.lineno))
            # Embedded HIP/C++ source strings still need a native-symbol audit.
            # Docstrings and ordinary messages are not executable source.
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                if re.search(r"#include|__global__|PYBIND11_MODULE|\b(?:hipcub|rocprim)::", node.value):
                    for hit in _native_hits(path, node.value, patterns):
                        hits.add((node.lineno + hit["line"] - 1, hit["symbol"]))
                    if check_platform:
                        for hit in _platform_native_hits(path, node.value):
                            hit = dict(hit, line=node.lineno + hit["line"] - 1)
                            if hit not in platform_hits:
                                platform_hits.append(hit)
            if isinstance(node, ast.Call) and dotted(node.func) in {"getattr", "eval", "exec"}:
                # Dynamic dispatch cannot be proved dead; inspect all definitions
                # instead of exempting potentially reachable forbidden code.
                pending.extend(definitions.values())
    violations = [{"file": str(path), "line": line, "symbol": symbol, "kind": "forbidden"}
                  for line, symbol in sorted(hits)]
    violations.extend(sorted(platform_hits, key=lambda h: (h["line"], h["symbol"])))
    return violations


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


def audit_file(path: Path, patterns: list[str], check_platform: bool = True) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".py":
        return _python_hits(path, text, patterns, check_platform=check_platform)
    hits = [dict(h, kind="forbidden") for h in _native_hits(path, text, patterns)]
    if check_platform:
        hits.extend(_platform_native_hits(path, text))
    return hits


def main() -> int:
    argv = [a for a in sys.argv[1:]]
    check_platform = True
    if "--no-platform" in argv:
        argv.remove("--no-platform")
        check_platform = False
    if len(argv) < 2:
        print(__doc__)
        return 2
    task_dir = Path(argv[0])
    files = [Path(p) for p in argv[1:]]
    patterns = sorted(set(GLOBAL_FORBIDDEN) | set(load_task_forbidden(task_dir)))

    all_hits = []
    for f in files:
        if not f.exists():
            print(f"[audit] 文件不存在: {f}", file=sys.stderr)
            return 2
        all_hits.extend(audit_file(f, patterns, check_platform=check_platform))

    kinds = sorted({h.get("kind", "forbidden") for h in all_hits})
    report = {"task_dir": str(task_dir), "passed": not all_hits,
              "kinds": kinds, "violations": all_hits}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not all_hits else 1



if __name__ == "__main__":
    sys.exit(main())
