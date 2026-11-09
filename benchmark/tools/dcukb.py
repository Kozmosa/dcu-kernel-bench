#!/usr/bin/env python3
"""dcukb.py — dcu-kernel-bench 算子收集流水线。

    scan   →  扫 aiter 源码，找出「有具体源码、核心计算可审查、无闭源库调用」的候选
    admit  →  逐算子准入：sha256 证据 + 依赖检测 + 独立编译性 → sources/<id>.yaml
    new    →  生成任务骨架：tasks/<id>/ + private/<id>/ + aiter 适配器骨架

设计约束（与评测集铁律一致）：
  - 只读 aiter 源码，绝不 vendor 进本仓库（third_party/ 是 submodule）
  - 生成的资产默认不落盘，必须显式 --write；已存在的文件一律不覆盖（除非 --force）
  - 不可自动化的部分（reference.py 的语义、task.yaml 的 io/shape/tolerance）留 TODO 骨架

用法：
    python benchmark/tools/dcukb.py scan  --aiter-root <aiter 检出> [--out candidates.json]
    python benchmark/tools/dcukb.py admit --id <task_id> --aiter-root <aiter 检出> \
                                          --file <相对路径>[:role] [--file ...] [--write]
    python benchmark/tools/dcukb.py new   --id <task_id> --name <op> --family <f> \
                                          --impl-lang <hip|triton> --entry <callable|model_class> \
                                          [--write]

退出码：0 = 成功；1 = 有候选/文件未通过判定；2 = 用法/配置错误。
"""

import argparse
import ast
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from string import Template

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS = REPO_ROOT / "benchmark" / "tasks"
PRIV = REPO_ROOT / "benchmark" / "private"
SOURCES = REPO_ROOT / "benchmark" / "sources"
CATALOG = REPO_ROOT / "operator_catalog.yaml"

AITER_REPO = "OpenDAS/aiter"
AITER_REMOTE = "https://developer.sourcefind.cn/codes/OpenDAS/aiter.git"

# kernelbench_compat 的目录层级映射 catalog 的 difficulty
DIFFICULTY_TO_LEVEL = {"basic": 1, "medium": 2, "hard": 3}
COMPAT_ROOT = REPO_ROOT / "benchmark" / "kernelbench_compat"

# 闭源库 / 不可审查框架：出现即不得准入（核心计算不在可审查源码中）
CLOSED_LIB_PATTERNS = [
    ("rocblas", "rocBLAS"),
    ("hipblaslt", "hipBLASLt"),
    ("hipblas", "hipBLAS"),
    ("miopen", "MIOpen"),
    ("hipdnn", "hipDNN"),
    ("cublas", "cuBLAS"),
    ("cudnn", "cuDNN"),
    ("cutlass", "CUTLASS"),
]

# 开源但属「模板框架包装」：核心计算不在算子自身源码里，同样不准入。
# 只匹配命名空间限定调用（opus:: / ck:: / ck_tile::）——不能匹配裸的 opus_，
# 否则 aiter_opus_plus.h 这类头文件名会造成假阳性。
FRAMEWORK_PATTERNS = [
    (r"\bck_tile::", "ck_tile"),
    (r"\bck::", "composable_kernel"),
    (r"\bopus::", "opus"),
]

FAMILY_KEYWORDS = [
    ("attention", ("attention", "mha", "mla", "fmha", "pa_decode", "pa_prefill", "flash")),
    ("moe", ("moe", "topk", "gating", "expert", "sorting", "align_block")),
    ("norm", ("rmsnorm", "layernorm", "groupnorm", "norm", "batchnorm")),
    ("rope", ("rope", "rotary", "pos_encoding", "position")),
    ("gemm", ("gemm", "bmm", "matmul", "mm_", "flatmm")),
    ("quant", ("quant", "fp8", "fp4", "int8", "int4", "mxfp", "smoothquant")),
    ("activation", ("activation", "silu", "gelu", "swiglu", "sigmoid", "tanh", "relu")),
    ("cache", ("cache", "kvcache", "reshape_and_cache", "paged")),
    ("reduce", ("reduce", "sum", "mean", "max_", "argmax", "cumsum")),
    ("sampling", ("sample", "top_p", "top_k", "beam", "repetition")),
]


# --------------------------------------------------------------------------
# 通用
# --------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def read_task_scalars(task_dir: Path, keys: tuple) -> dict:
    """从 task.yaml 读顶层标量字段（最小解析，不引入 pyyaml，同 static_audit.py）。

    只认顶格的 `key: value`，不处理嵌套——entry / difficulty 都是顶层字段。
    必须处理行尾注释：现有 task.yaml 常见 `entry: model_class   # 说明` 这种写法。
    """
    text = (task_dir / "task.yaml").read_text(encoding="utf-8")
    found = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z_]\w*):\s*(.*?)\s*$", line)
        if not m or m.group(1) not in keys:
            continue
        raw = m.group(2)
        if raw[:1] in ("'", '"'):                      # 带引号：取引号内内容
            quote = raw[0]
            end = raw.find(quote, 1)
            value = raw[1:end] if end > 0 else raw.strip(quote)
        else:                                          # 裸标量：先切掉行尾注释
            value = raw.split("#", 1)[0].strip()
        found[m.group(1)] = value
    return found


def compat_path(task_id: str, level: int) -> Path:
    return COMPAT_ROOT / f"level{level}" / f"{task_id}.py"


def model_class_tasks() -> list:
    """所有 entry: model_class 的任务 id（只有这些需要镜像）。"""
    ids = []
    for task_dir in sorted(TASKS.iterdir()) if TASKS.exists() else []:
        if not task_dir.is_dir() or not (task_dir / "task.yaml").exists():
            continue
        if read_task_scalars(task_dir, ("entry",)).get("entry") == "model_class":
            ids.append(task_dir.name)
    return ids


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"(?m)^\s*#.*$", "", text)
    return text


# 允许的第三方/标准库依赖（白名单之外的三方依赖 → needs_review）
ALLOWED_IMPORTS = {
    "triton", "torch", "numpy", "math", "typing", "functools", "dataclasses",
    "itertools", "os", "sys", "re", "warnings", "collections", "abc", "enum",
    "contextlib", "string", "copy", "json", "time", "logging", "types",
}


# 外部配置依赖：autotune 装饰器，或手工从 AITER_TRITON_CONFIGS_PATH 读 JSON。
# 后者不写 @triton.autotune，只看装饰器会漏检（moe_routing_sigmoid_top1_fused 即如此）。
CONFIG_PATH_MARKERS = ("AITER_TRITON_CONFIGS_PATH", "CONFIGS_PATH")
AUTOTUNE_MARKERS = ("@triton.autotune", "@triton.heuristics")


def needs_external_config(raw: str, code: str) -> bool:
    if any(m in raw for m in AUTOTUNE_MARKERS):
        return True
    if any(m in code for m in CONFIG_PATH_MARKERS):
        return True
    # 形如 _get_config(...) + json.load：手工读配置文件
    return bool(re.search(r"_get_config\s*\(", code)) and bool(re.search(r"json\.load", code))


# KernelBench 的 model_class 契约要求 forward 返回**单个**张量
# （run_and_check_correctness 会做 output.shape / allclose 比对）。
# 用 AST 精确判断**公开函数**的返回元数——正则标记整个文件会误伤大文件里的
# 内部辅助函数（norm.py / rope.py 各有十几个函数）。
def public_return_arity(raw: str) -> dict:
    """{公开函数名: 该函数所有 return 的元数集合}。解析失败返回 {}。"""
    try:
        tree = ast.parse(raw)
    except SyntaxError:
        return {}
    result = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        arities = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Return) and sub.value is not None:
                arities.add(len(sub.value.elts) if isinstance(sub.value, ast.Tuple) else 1)
        result[node.name] = sorted(arities)
    return result


def multi_output_public(raw: str) -> list:
    """公开函数中返回多个值的（如 fused_qk_cat 的 `return q_out, k_out`）。"""
    return [
        name for name, arities in public_return_arity(raw).items()
        if any(n > 1 for n in arities)
    ]


def void_public(raw: str) -> list:
    """公开函数中**没有任何 return 值**的 —— 通常是原地写预分配缓冲区。

    这类 API 同样不适配 model_class：KernelBench 要 forward 返回张量并与参考
    做全量比对，而原地写入既没有返回值、也常带"未定义填充区"。
    """
    return [name for name, arities in public_return_arity(raw).items() if not arities]


def third_party_imports(raw: str) -> list:
    """提取白名单外的顶层三方依赖（aiter/相对导入不算）。"""
    found = set()
    for line in raw.splitlines():
        m = re.match(r"\s*(?:from|import)\s+([A-Za-z_][\w.]*)", line)
        if not m:
            continue
        top = m.group(1).split(".")[0]
        if top in ALLOWED_IMPORTS or top == "aiter" or line.lstrip().startswith(("from .", "import .")):
            continue
        found.add(top)
    return sorted(found)


def guess_family(name: str) -> str:
    lowered = name.lower()
    for family, keys in FAMILY_KEYWORDS:
        if any(k in lowered for k in keys):
            return family
    return "elementwise"


def path_frameworks(source_path: str) -> list:
    """按路径段判定框架目录（ck_* / cktile_* / opus_* 等）。

    仅查文件内容是不够的：csrc/opus_xxx/include/xxx.cuh 里可能一个 opus:: 都不出现，
    但整个算子属于 opus 框架，按 sources/README.md 的准入标准同样不得准入。
    """
    found = []
    for seg in Path(source_path).parts:
        s = seg.lower()
        is_framework = (
            s.startswith(("ck_", "cktile_", "opus_"))
            or s.endswith(("_ck", "_cktile", "_opus"))
            or s in ("ck", "ck_tile", "opus")
        )
        if is_framework:
            label = "opus" if "opus" in s else "composable_kernel"
            if label not in found:
                found.append(label)
    return found


def internal_path(source_path: str) -> bool:
    """下划线开头的路径段表示内部实现，不是公开算子入口。

    例：aiter/ops/triton/_triton_kernels/attention/pa_mqa_logits.py 是某个多阶段
    fp8 算法的内部 stage kernel，没有可交付的公开算子契约，不应作为题目来源。
    """
    return any(seg.startswith("_") for seg in Path(source_path).parts)


def count_lines(path: Path) -> int:
    try:
        return sum(1 for _ in path.open("r", encoding="utf-8", errors="replace"))
    except OSError:
        return 0


_SAFE_SCALAR = re.compile(r"^[A-Za-z0-9_./+@-]+$")


def _scalar(value) -> str:
    """输出 YAML 标量；不确定安全的就加引号。"""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if _SAFE_SCALAR.match(text):
        return text
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _emit(key: str, value, indent: int) -> list:
    pad = "  " * indent
    lines = []
    if isinstance(value, dict):
        lines.append(f"{pad}{key}:")
        for k, v in value.items():
            lines.extend(_emit(k, v, indent + 1))
    elif isinstance(value, list):
        lines.append(f"{pad}{key}:")
        for item in value:
            if isinstance(item, dict):
                keys = list(item.items())
                for idx, (k, v) in enumerate(keys):
                    prefix = f"{pad}  - {k}:" if idx == 0 else f"{pad}    {k}:"
                    if isinstance(v, (dict, list)):
                        lines.append(prefix)
                        for kk, vv in (v.items() if isinstance(v, dict) else enumerate(v)):
                            lines.extend(_emit(str(kk), vv, indent + 3))
                    else:
                        lines.append(f"{prefix} {_scalar(v)}")
            else:
                lines.append(f"{pad}  - {_scalar(item)}")
    else:
        lines.append(f"{pad}{key}: {_scalar(value)}")
    return lines


def dump_yaml(path: Path, payload: dict, header: str = "") -> None:
    """最小 YAML 输出，刻意不依赖 pyyaml（与 static_audit.py 的做法一致）。

    dcukb 是维护者工具，应当只依赖标准库即可运行；采集/评测环节才需要 torch 等。
    """
    body = "\n".join(line for k, v in payload.items() for line in _emit(k, v, 0)) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text((header + body) if header else body, encoding="utf-8")


# --------------------------------------------------------------------------
# scan
# --------------------------------------------------------------------------

@dataclass
class Candidate:
    impl_lang: str
    source_path: str
    suggested_name: str
    family: str
    lines: int
    bytes: int
    device_kernels: list = field(default_factory=list)
    entry_points: list = field(default_factory=list)
    closed_libs: list = field(default_factory=list)
    frameworks: list = field(default_factory=list)
    internal_includes: list = field(default_factory=list)
    core_elsewhere: list = field(default_factory=list)
    uses_autotune: bool = False
    needs_external_config: bool = False
    multi_outputs: list = field(default_factory=list)
    void_publics: list = field(default_factory=list)
    is_internal_path: bool = False
    has_public_entry: bool = True
    unknown_imports: list = field(default_factory=list)
    opus_header_include: bool = False
    verdict: str = "needs_review"
    reasons: list = field(default_factory=list)
    advisories: list = field(default_factory=list)
    difficulty_hint: str = "medium"

    @property
    def key(self) -> str:
        return f"{self.impl_lang}:{self.source_path}"


def _scan_triton_file(path: Path, root: Path, family_filter: str | None) -> Candidate | None:
    raw = path.read_text(encoding="utf-8", errors="replace")
    code = strip_comments(raw)

    jit = sorted(set(re.findall(r"@triton\.jit\s*\n\s*def\s+([A-Za-z_]\w*)", raw)))
    public = sorted(set(re.findall(r"(?m)^def\s+([A-Za-z]\w*)\s*\(", code)))
    public = [n for n in public if not n.startswith("_")]

    name = path.stem
    family = guess_family(name)
    if family_filter and family != family_filter:
        return None
    if not jit:
        return None  # triton 轨道必须真的有 @triton.jit kernel

    closed = [label for pat, label in CLOSED_LIB_PATTERNS if re.search(pat, code, re.I)]
    frameworks = [label for pat, label in FRAMEWORK_PATTERNS if re.search(pat, code)]
    for label in path_frameworks(rel(root, path)):
        if label not in frameworks:
            frameworks.append(label)

    lines = count_lines(path)
    cand = Candidate(
        impl_lang="triton",
        source_path=rel(root, path),
        suggested_name=name,
        family=family,
        lines=lines,
        bytes=path.stat().st_size,
        device_kernels=jit,
        entry_points=public,
        closed_libs=closed,
        frameworks=frameworks,
        uses_autotune=any(m in raw for m in AUTOTUNE_MARKERS),
        needs_external_config=needs_external_config(raw, code),
        multi_outputs=multi_output_public(raw),
        void_publics=void_public(raw),
        is_internal_path=internal_path(rel(root, path)),
        has_public_entry=bool(public_return_arity(raw)),
        unknown_imports=third_party_imports(raw),
        difficulty_hint="basic" if lines < 200 else ("medium" if lines < 1500 else "hard"),
    )
    _judge(cand, has_core=bool(jit))
    return cand


def _scan_hip_file(path: Path, root: Path, family_filter: str | None) -> Candidate | None:
    raw = path.read_text(encoding="utf-8", errors="replace")
    code = strip_comments(raw)

    globals_ = sorted(set(re.findall(r"__global__\s+(?:void|[\w:]+)\s+([A-Za-z_]\w*)", code)))
    includes = sorted(set(re.findall(r'#include\s+"([^"]+)"', raw)))
    internal = [i for i in includes if not i.startswith(("<", "hip/"))]

    name = path.stem
    family = guess_family(name)
    if family_filter and family != family_filter:
        return None

    closed = [label for pat, label in CLOSED_LIB_PATTERNS if re.search(pat, code, re.I)]
    frameworks = [label for pat, label in FRAMEWORK_PATTERNS if re.search(pat, code)]
    for label in path_frameworks(rel(root, path)):
        if label not in frameworks:
            frameworks.append(label)

    # 没有 __global__ 但 include 了内部 .cuh → 核心计算在那个 .cuh 里
    core_elsewhere = []
    if not globals_:
        core_elsewhere = [i for i in internal if i.endswith((".cuh", ".hpp", ".h"))]

    lines = count_lines(path)
    cand = Candidate(
        impl_lang="hip",
        source_path=rel(root, path),
        suggested_name=name,
        family=family,
        lines=lines,
        bytes=path.stat().st_size,
        device_kernels=globals_,
        entry_points=sorted(set(re.findall(r"(?m)^(?:extern\s+\"C\"\s+)?\w[\w:<>,\s\*&]*\s+([A-Za-z_]\w*)\s*\(", code)))[:20],
        closed_libs=closed,
        frameworks=frameworks,
        internal_includes=internal,
        core_elsewhere=core_elsewhere,
        opus_header_include=any(
            i.endswith(("aiter_opus_plus.h", "opus/opus.hpp")) for i in internal
        ),
        difficulty_hint="basic" if lines < 200 else ("medium" if lines < 1500 else "hard"),
    )
    _judge(cand, has_core=bool(globals_))
    return cand


def _judge(cand: Candidate, has_core: bool) -> None:
    """判定 verdict 并写明理由。判据对齐 sources/README.md 的准入标准。

    注意：needs_review 不是「不合格」，而是「工具无法自动判定，需人工确认」——
    例如 autotune 类算子需要额外的 config JSON、或依赖了未知第三方库。
    """
    if cand.closed_libs:
        cand.verdict = "rejected"
        cand.reasons.append(f"调用闭源库：{', '.join(cand.closed_libs)}")
    if cand.frameworks:
        cand.verdict = "rejected"
        cand.reasons.append(f"核心计算走模板框架：{', '.join(cand.frameworks)}")
    if cand.verdict == "rejected":
        return

    if cand.is_internal_path:
        cand.verdict = "rejected"
        cand.reasons.append(
            "路径含下划线开头的段（_triton_kernels/ 等内部实现目录）：属多阶段算法的"
            "内部 stage kernel，没有可交付的公开算子契约，不作题目来源"
        )
        return

    if not cand.has_public_entry:
        cand.verdict = "needs_review"
        cand.reasons.append(
            "文件中没有非下划线的公开函数：缺少任务入口，需确认是否有对外 API"
        )
        return

    if not has_core:
        if cand.core_elsewhere:
            cand.verdict = "needs_review"
            cand.reasons.append(f"本文件无 device kernel，核心计算疑似在：{', '.join(cand.core_elsewhere)}")
        else:
            cand.verdict = "rejected"
            cand.reasons.append("本文件既无 device kernel 也无内部 header 引用，疑似纯 host 包装")
        return

    if cand.lines and cand.lines < 20:
        cand.verdict = "needs_review"
        cand.reasons.append(f"仅 {cand.lines} 行，疑似模板实例化，核心计算在别处")
        return

    review_notes = []
    if cand.needs_external_config:
        review_notes.append(
            "依赖外部 config JSON（@triton.autotune 或 AITER_TRITON_CONFIGS_PATH）："
            "独立复现时若缺该文件会失败或退化到默认配置"
        )
    if cand.unknown_imports:
        review_notes.append(f"依赖白名单外的三方库：{', '.join(cand.unknown_imports)}")
    if cand.multi_outputs:
        review_notes.append(
            f"公开函数返回多值（{', '.join(cand.multi_outputs)}）：model_class 契约要求 "
            "forward 返回单个张量，需确认主算子是否有单输出入口，否则应剔除"
        )
    if cand.void_publics:
        # 仅提示、不降级：1002_paged_attention 本身就是 void API，题目侧包一层即可适配。
        cand.advisories.append(
            f"void API（{', '.join(cand.void_publics[:3])}）：写预分配缓冲区。可适配——"
            "题目侧分配 out → 调用 → 返回 out；但需人工确认输出缓冲区只有 1 个、"
            "且全部元素都有定义（多缓冲区或填充区未定义则不适配）"
        )
    if cand.opus_header_include:
        review_notes.append(
            "include 了 aiter_opus_plus.h（内部 using namespace opus），"
            "可能以非限定名调用 opus 框架——需人工确认核心计算是否自持"
        )

    if review_notes:
        cand.verdict = "needs_review"
        cand.reasons.extend(review_notes)
        return

    cand.verdict = "admissible"
    cand.reasons.append(f"含 {len(cand.device_kernels)} 个 device kernel，未见闭源库/框架调用")


def cmd_scan(args) -> int:
    root = Path(args.aiter_root).resolve()
    if not root.exists():
        print(f"[scan] aiter 检出不存在: {root}", file=sys.stderr)
        return 2

    candidates: list[Candidate] = []
    if args.impl_lang in ("triton", "all"):
        for path in sorted((root / "aiter" / "ops" / "triton").rglob("*.py")):
            if path.name == "__init__.py" or "configs" in path.parts or "utils" in path.parts:
                continue
            c = _scan_triton_file(path, root, args.family)
            if c:
                candidates.append(c)
    if args.impl_lang in ("hip", "all"):
        csrc = root / "csrc"
        if csrc.exists():
            for path in sorted(list(csrc.rglob("*.cu")) + list(csrc.rglob("*.cuh"))):
                c = _scan_hip_file(path, root, args.family)
                if c:
                    candidates.append(c)

    order = {"admissible": 0, "needs_review": 1, "rejected": 2}
    candidates.sort(key=lambda c: (order[c.verdict], c.family, c.lines))

    counts = {v: sum(1 for c in candidates if c.verdict == v) for v in order}
    print(f"[scan] aiter-root = {root}")
    print(f"[scan] 候选 {len(candidates)} 个  admissible={counts['admissible']} "
          f"needs_review={counts['needs_review']} rejected={counts['rejected']}")
    print()
    for c in candidates:
        flag = {"admissible": "OK  ", "needs_review": "REVIEW", "rejected": "REJECT"}[c.verdict]
        print(f"  [{flag}] {c.impl_lang:6} {c.family:11} {c.source_path}")
        print(f"           kernels={len(c.device_kernels)} lines={c.lines} "
              f"difficulty_hint={c.difficulty_hint}")
        for r in c.reasons:
            print(f"           - {r}")
        for a in c.advisories:
            print(f"           ~ {a}")

    payload = {
        "schema": 1,
        "aiter_root": str(root),
        "aiter_remote": AITER_REMOTE,
        "generated_at": date.today().isoformat(),
        "counts": counts,
        "candidates": [asdict(c) for c in candidates],
    }
    out = Path(args.out) if args.out else REPO_ROOT / "benchmark" / "tools" / "scan_candidates.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[scan] 已写入 {out}")

    if args.for_admit:
        eligible = [c for c in candidates if c.verdict == "admissible"]
        print(f"\n[scan] admissible 候选 {len(eligible)} 个，建议下一步：")
        for c in eligible[:10]:
            print(f"  dcukb admit --id <序号>_{c.suggested_name} --file {c.source_path}:device_kernel ...")
    return 0


# --------------------------------------------------------------------------
# admit
# --------------------------------------------------------------------------

SOURCE_HEADER = """# 准入审核记录 — 由 benchmark/tools/dcukb.py admit 生成骨架，人工复核后填写 reviewer/note。
# 判据见 benchmark/sources/README.md：核心计算在可审查 device 源码中、不调闭源库、
# 能独立编译运行、I/O 与 dtype/shape 可描述、能写独立 reference。

"""


def _parse_file_spec(spec: str) -> tuple[str, str]:
    if ":" in spec:
        path, role = spec.rsplit(":", 1)
        return path, role
    return spec, "device_kernel"


def cmd_admit(args) -> int:
    root = Path(args.aiter_root).resolve()
    if not root.exists():
        print(f"[admit] aiter 检出不存在: {root}", file=sys.stderr)
        return 2

    files, problems = [], []
    for spec in args.file:
        rel_path, role = _parse_file_spec(spec)
        path = root / rel_path
        if not path.exists():
            problems.append(f"文件不存在: {rel_path}")
            continue
        code = strip_comments(path.read_text(encoding="utf-8", errors="replace"))
        closed = [label for pat, label in CLOSED_LIB_PATTERNS if re.search(pat, code, re.I)]
        frameworks = [label for pat, label in FRAMEWORK_PATTERNS if re.search(pat, code)]
        if closed:
            problems.append(f"{rel_path} 调用闭源库：{', '.join(closed)}")
        if frameworks:
            problems.append(f"{rel_path} 核心计算走模板框架：{', '.join(frameworks)}")
        files.append({
            "path": rel_path,
            "sha256": sha256_file(path),
            "role": role,
        })

    if problems:
        print("[admit] 准入前置检查未通过：", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    license_note = args.license
    payload = {
        "task_id": args.id,
        "source": {
            "repo": AITER_REPO,
            "remote": AITER_REMOTE,
            "commit": args.commit,
            "license": license_note,
            "files": files,
        },
        "admission": {
            "core_compute_in_source": True,
            "calls_closed_libs": False,
            "builds_standalone": None,
            "has_reference": None,
            "decision": "candidate",
            "reviewer": "",
            "date": date.today().isoformat(),
            "note": args.note or "TODO: 说明核心计算所在函数、排除项、任务范围界定",
        },
    }

    out = SOURCES / f"{args.id}.yaml"
    print(f"[admit] task_id = {args.id}")
    print(f"[admit] commit  = {args.commit}")
    for f in files:
        print(f"  {f['role']:14} {f['path']}")
        print(f"                 sha256 {f['sha256']}")
    print()
    print("[admit] 仍需人工确认（已留空）：")
    print("  - admission.builds_standalone : 能否在 environment.yaml 锁定环境独立编译")
    print("  - admission.has_reference     : 能否写出独立 PyTorch/CPU reference")
    print("  - admission.decision          : candidate → admitted")
    print("  - admission.note              : 核心计算位置、排除项、范围界定")

    if not args.write:
        print(f"\n[admit] 未落盘（加 --write 写入 {out}）")
        return 0
    if out.exists() and not args.force:
        print(f"[admit] 已存在，拒绝覆盖：{out}（加 --force 强制）", file=sys.stderr)
        return 2
    dump_yaml(out, payload, header=SOURCE_HEADER)
    print(f"\n[admit] 已写入 {out}")
    return 0


# --------------------------------------------------------------------------
# new
# --------------------------------------------------------------------------

TASK_YAML = Template('''id: "$id"
name: $name
family: $family
impl_lang: $impl_lang
$entry_line
difficulty: $difficulty

description: >
  TODO：语义描述。只引用 reference.py 的语义，不得粘贴 aiter 官方实现代码。
  来源：$source_path（$commit）

io:
  inputs:
    - {name: TODO, dtype: [float16, bfloat16], layout: contiguous, ndim: 1}
  outputs:
    - {name: TODO, dtype: same_as_input, layout: contiguous, ndim: 1}

shape:
  # TODO：填写各维取值范围与不变式
  dynamic: true

tolerance:
  float16:  {atol: 2.0e-2, rtol: 2.0e-2}
  bfloat16: {atol: 2.0e-2, rtol: 2.0e-2}
  compare: "reference 以 float32 计算后 cast 到输出 dtype 再比较"

# static_audit.py 强制执行：提交代码中出现以下符号即判违规
forbidden:
  - rocblas
  - hipblas
  - hipblaslt
  - miopen
  - hipdnn
  - cublas
  # TODO：补充本算子特有的禁用项（如高层 torch op）

source:
  repo: $repo
  commit: $commit
  path: $source_path   # 仅维护者溯源用；Agent 上下文中不得出现
''')

REFERENCE_PY = Template('''"""$id — 独立 PyTorch 参考实现（语义唯一依据）。

TODO：这是收集流程中唯一无法自动化的部分——用纯 PyTorch 独立重写算子语义。
禁止粘贴或改写 aiter 官方实现；只依据 task.yaml 的语义描述与数学定义。

来源（仅维护者溯源）：$source_path @ $commit
"""


def reference(# TODO: 与 task.yaml 的 io.inputs 对齐
              ):
    """TODO：按数学定义实现，保持 float32 中间累加。"""
    raise NotImplementedError("TODO: 实现 reference")


def make_inputs(# TODO: 与 task.yaml 的 shape 范围对齐
                seed: int = 0):
    """按公开/隐藏案例描述生成输入。仅在评测端运行。

    必须覆盖 task.yaml 声明的 dtype 与 shape 下界/上界，并保证同 seed 可复现。
    """
    import torch  # noqa: F401
    raise NotImplementedError("TODO: 实现输入生成器")
''')

REFERENCE_MODEL_CLASS = Template('''"""$id — 独立 PyTorch 参考实现（语义唯一依据），KernelBench 兼容形态。

本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块导入，
reference 实现与输入生成全部内联。语义对标 aiter 的 $source_path。

TODO：这是收集流程中唯一无法自动化的部分——用纯 PyTorch 独立重写算子语义。
只依据数学定义，禁止粘贴或改写 aiter 官方实现。
"""

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """TODO：语义描述（算子做什么、输入输出形状/dtype、关键约束）。

    实现约束（违规判负）：
      - 核心计算必须在提交文件内完成，禁止调用 TODO: 列出禁用项。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间累加用 float32；输出 cast 回输入 dtype。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    """

    def __init__(self, # TODO: init 参数
                 ):
        super().__init__()
        raise NotImplementedError("TODO: __init__")

    def forward(self, # TODO: 与 task.yaml 的 io.inputs 对齐
                ):
        raise NotImplementedError("TODO: forward")


def get_init_inputs():
    """Model(*get_init_inputs()) 即可构造。"""
    raise NotImplementedError("TODO: get_init_inputs")


def get_inputs():
    """评测器在 set_seed 后调用本函数；随机部分消费全局 RNG 以获得输入多样性。"""
    raise NotImplementedError("TODO: get_inputs")
''')

PUBLIC_CASES = Template('''{
  "task_id": "$id",
  "cases": [
    {"name": "TODO_typical", "seed": 0}
  ]
}
''')
HIDDEN_CASES = Template('''{
  "task_id": "$id",
  "cases": [
    {"name": "TODO_boundary_low",  "seed": 101},
    {"name": "TODO_boundary_high", "seed": 102},
    {"name": "TODO_unaligned",     "seed": 103}
  ]
}
''')

PERF_CASES = Template('''{
  "task_id": "$id",
  "cases": [
    {"name": "perf_TODO", "seed": 201}
  ],
  "timing": {"warmup_iters": 10, "repeat_iters": 30, "reduction": "median"}
}
''')

BASELINE_JSON = Template('''{
  "task_id": "$id",
  "status": "placeholder",
  "note": "由 benchmark/evaluator/record_baseline.py 在 DCU 真机采集 aiter 官方基线后填充。",
  "environment": null,
  "baselines": []
}
''')

AITER_IMPL = Template('''# aiter_impl.py — $id 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
# 契约：run(inputs, init_args, device) -> (out_tensor, ctx)
#   inputs 由 tasks/<继承任务>/reference.py::make_inputs 按 perf_cases 生成
#
# 来源：$source_path @ $commit
# aiter 必须已在 PYTHONPATH 上（见 tools/aiter_shim 的构造方式）。


def run(inputs, init_args, device):
    """TODO：按 aiter 官方函数签名绑定参数。

    核对清单：
      1. 参数顺序与官方测试 op_tests/ 下的用法一致
      2. 布局是否与 make_inputs 的输出一致（必要时要 permute/contiguous）
      3. scale / 量化 scale 等标量参数的取值来源（默认 1/sqrt(head_size) 等）
      4. 分派路径（v1/v2 等）是否落在 task.yaml 声明的范围内，超出应显式报错
    """
    raise NotImplementedError("TODO: 实现 aiter 调用适配器")
''')

CATALOG_ENTRY = Template('''  - id: "$id"
    name: $name
    family: $family
    source:
      repo: $repo
      commit: "$commit"
      path: "$source_path"
    impl_lang: $impl_lang
$catalog_entry_line
    core_compute: "TODO: 核心计算所在文件与函数（准入审核的关键证据）"
    external_calls: []
    status: candidate
    difficulty: $difficulty
    note: "TODO: 审核备注"
''')


def _ensure_writable(path: Path, force: bool) -> bool:
    if path.exists() and not force:
        print(f"  [skip] 已存在，不覆盖：{rel(REPO_ROOT, path)}")
        return False
    return True


def cmd_new(args) -> int:
    task_dir = TASKS / args.id
    priv_dir = PRIV / args.id
    if task_dir.exists() and not args.force:
        print(f"[new] 任务目录已存在：{task_dir}（加 --force 覆盖）", file=sys.stderr)
        return 2

    entry_line = f"entry: {args.entry}" if args.entry else ""
    catalog_entry_line = f"    entry: {args.entry}" if args.entry else ""
    ctx = dict(
        id=args.id, name=args.name, family=args.family, impl_lang=args.impl_lang,
        entry=args.entry or "", entry_line=entry_line, catalog_entry_line=catalog_entry_line,
        difficulty=args.difficulty, commit=args.commit, repo=AITER_REPO,
        source_path=args.source_path or "TODO",
    )

    artifacts = [
        (task_dir / "task.yaml", TASK_YAML.substitute(ctx)),
        (task_dir / "reference.py",
         (REFERENCE_MODEL_CLASS if args.entry == "model_class" else REFERENCE_PY).substitute(ctx)),
        (task_dir / "public_cases.json", PUBLIC_CASES.substitute(ctx)),
        (priv_dir / "hidden_cases.json", HIDDEN_CASES.substitute(ctx)),
        (priv_dir / "perf_cases.json", PERF_CASES.substitute(ctx)),
        (priv_dir / "baseline.json", BASELINE_JSON.substitute(ctx)),
        (priv_dir / "aiter_impl.py", AITER_IMPL.substitute(ctx)),
    ]
    # model_class 形态由框架 loader 从 Model 自动生成脚手架，不单独提供 starter/
    if args.entry != "model_class":
        artifacts.append((task_dir / "starter" / ("kernel.hip" if args.impl_lang == "hip" else "kernel.py"),
                          "# TODO: Starter 骨架\n"))

    print(f"[new] 生成任务骨架：{args.id}")
    print(f"      impl_lang={args.impl_lang} entry={args.entry or 'callable'} family={args.family}")
    print()
    for path, content in artifacts:
        exists = path.exists()
        mark = "skip" if (exists and not args.write) else ("over" if exists else "new ")
        print(f"  [{mark}] {rel(REPO_ROOT, path)}")
        if args.write and (not exists or args.force):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    print()
    print("[new] 需要人工补全的部分：")
    print("  1. tasks/<id>/task.yaml 的 io / shape / forbidden  ← 从 aiter 源码与语义提炼")
    print("  2. tasks/<id>/reference.py 的语义实现           ← ★ 唯一不可自动化的核心工作")
    print("  3. tasks/<id>/public_cases.json 与 private/*_cases.json 的具体 case")
    print("  4. private/<id>/aiter_impl.py 的调用绑定")
    print("  5. operator_catalog.yaml 登记（下面这段需手工追加到 operators: 下）")
    print()
    print(CATALOG_ENTRY.substitute(ctx))
    if not args.write:
        print("[new] 未落盘（加 --write 写入上述文件）")
    return 0


# --------------------------------------------------------------------------
# mirror
# --------------------------------------------------------------------------

def _mirror_plan(task_id: str) -> dict:
    """算出一个 model_class 任务的镜像计划；不合法则抛 ValueError。"""
    task_dir = TASKS / task_id
    if not (task_dir / "task.yaml").exists():
        raise ValueError(f"任务不存在或缺少 task.yaml：{task_dir}")

    scalars = read_task_scalars(task_dir, ("entry", "difficulty"))
    entry = scalars.get("entry")
    if entry != "model_class":
        raise ValueError(
            f"entry={entry or 'callable'}；只有 model_class 任务需要镜像到 kernelbench_compat"
            "（callable 任务用 tasks/<id>/starter/）"
        )
    difficulty = scalars.get("difficulty")
    if difficulty not in DIFFICULTY_TO_LEVEL:
        raise ValueError(
            f"difficulty={difficulty!r} 无法映射 level（应为 basic | medium | hard）"
        )
    src = task_dir / "reference.py"
    if not src.exists():
        raise ValueError(f"缺少 {rel(REPO_ROOT, src)}")
    return {
        "task_id": task_id,
        "entry": entry,
        "difficulty": difficulty,
        "level": DIFFICULTY_TO_LEVEL[difficulty],
        "src": src,
        "dst": compat_path(task_id, DIFFICULTY_TO_LEVEL[difficulty]),
    }


def cmd_mirror(args) -> int:
    if args.all:
        task_ids = model_class_tasks()
        if not task_ids:
            print("[mirror] 没有 entry: model_class 的任务", file=sys.stderr)
            return 2
    elif args.id:
        task_ids = [args.id]
    else:
        print("[mirror] 需要 --id 或 --all", file=sys.stderr)
        return 2

    mode = "check（只校验）" if args.check else ("write" if args.write else "dry-run")
    print("[mirror] tasks/<id>/reference.py  →  kernelbench_compat/level<N>/<id>.py")
    print(f"[mirror] 模式 = {mode}")
    print()

    pending, done = 0, 0
    for task_id in task_ids:
        try:
            plan = _mirror_plan(task_id)
        except ValueError as exc:
            print(f"  [ERR  ] {task_id:26} {exc}")
            pending += 1
            continue

        src_bytes = plan["src"].read_bytes()
        dst = plan["dst"]
        if not dst.exists():
            state, detail = "new", "目标不存在"
        elif dst.read_bytes() == src_bytes:
            state, detail = "ok", "字节一致"
        else:
            state, detail = "drift", "与 tasks 侧不一致"

        print(f"  [{state:5}] {task_id:26} level{plan['level']} ({plan['difficulty']})"
              f"  -> {rel(REPO_ROOT, dst)}")
        print(f"          {detail}（src {len(src_bytes)}B）")

        if state == "ok":
            done += 1
        elif args.write and not args.check:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src_bytes)
            print("          -> 已写入")
            done += 1
        else:
            pending += 1

        # 同 id 出现在别的 level 目录 = difficulty 改过但旧镜像没删
        for level_dir in sorted(COMPAT_ROOT.glob("level*")):
            stray = level_dir / f"{task_id}.py"
            if stray.exists() and stray.resolve() != dst.resolve():
                print(f"          [WARN] level 不匹配的残留镜像：{rel(REPO_ROOT, stray)}，应删除")
                pending += 1

    print()
    if pending == 0:
        print(f"[mirror] {len(task_ids)} 个任务全部一致")
        return 0
    if args.write and not args.check:
        print(f"[mirror] 完成：{done} 个一致/已写入，{pending} 处仍需人工处理")
        return 1 if pending else 0
    print(f"[mirror] {pending} 处需要处理（加 --write 修复）", file=sys.stderr)
    return 1


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="dcukb", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scan", help="扫 aiter 源码找候选算子")
    p.add_argument("--aiter-root", required=True, help="aiter 检出的根目录")
    p.add_argument("--impl-lang", choices=["triton", "hip", "all"], default="all")
    p.add_argument("--family", default=None, help="只保留指定家族，如 attention")
    p.add_argument("--out", default=None, help="候选清单输出路径（默认 benchmark/tools/scan_candidates.json）")
    p.add_argument("--for-admit", action="store_true", help="额外打印下一步 admit 命令示例")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("admit", help="生成准入记录 sources/<id>.yaml")
    p.add_argument("--id", required=True, help="任务 id，如 1004_paged_attention_v2")
    p.add_argument("--aiter-root", required=True)
    p.add_argument("--file", action="append", required=True,
                   help="相对 aiter 根的文件路径，可带 :role 后缀（device_kernel/host_launch/test/header）")
    p.add_argument("--commit", required=True, help="aiter commit（40 位）")
    p.add_argument("--license", default="LICENSE (MIT)")
    p.add_argument("--note", default=None)
    p.add_argument("--write", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_admit)

    p = sub.add_parser("new", help="生成任务骨架")
    p.add_argument("--id", required=True)
    p.add_argument("--name", required=True, help="算子名，如 paged_attention_decode")
    p.add_argument("--family", required=True)
    p.add_argument("--impl-lang", required=True, choices=["hip", "triton"])
    p.add_argument("--entry", choices=["callable", "model_class"], default=None,
                   help="缺省 callable；model_class 供 KernelBench 兼容形态")
    p.add_argument("--difficulty", default="medium", choices=["basic", "medium", "hard"])
    p.add_argument("--commit", default="", help="aiter commit")
    p.add_argument("--source-path", default=None, help="aiter 内的相对路径")
    p.add_argument("--write", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_new)

    p = sub.add_parser("mirror", help="把 model_class 任务的 reference.py 镜像到 kernelbench_compat")
    p.add_argument("--id", default=None, help="任务 id")
    p.add_argument("--all", action="store_true", help="处理所有 entry: model_class 的任务")
    p.add_argument("--write", action="store_true", help="实际写入（默认只报告）")
    p.add_argument("--check", action="store_true", help="只校验，绝不写入；不一致则退出 1（供 CI）")
    p.set_defaults(func=cmd_mirror)

    return ap


def main() -> int:
    # Windows 控制台默认非 UTF-8，中文输出会乱码
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
