"""生成 HIP / Triton 双轨题面树：同一道题，两条实现轨各一份题面。

为什么需要两份：`Model` docstring 里写着"目标硬件 + 实现语言 + 锚点区"，而
PyramidKernel 的 scaffold 也是按 backend 生成的（Triton 用 `helpers` /
`forward_stmt_N` 锚点，native-cuda 用 `CUDA_CU_SRC` / `CUDA_CPP_SRC` 锚点并要
`<hip/hip_runtime.h>`）。所以要按另一条轨跑同一道题，题面必须跟着换，否则
agent 会被告知"写 HIP"却拿到 Triton 锚点（或反之）。

本脚本只改 docstring 里与"轨"相关的句子，其余（语义、IO 契约、边界、禁用列表）
逐字保留；代码不动。

    python benchmark/tools/make_dualtrack.py --out .dcu_runs/dual --write
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import textwrap

REPO = pathlib.Path(__file__).resolve().parents[2]
TASKS = REPO / "benchmark" / "tasks"
COMPAT = REPO / "benchmark" / "kernelbench_compat"

TRITON_SENTENCE = "Target hardware: Hygon DCU (gfx936). Implementation language: Triton."
HIP_SENTENCE_RAW = (
    "Target hardware: Hygon DCU (gfx936/gfx938). Implementation language: HIP "
    "(compiled by hipcc; write inside the CUDA_CU_SRC / CUDA_CPP_SRC anchor regions; "
    "device code must include <hip/hip_runtime.h> -- <cuda_runtime.h> is not available "
    "on this machine)."
)

HIP_START = re.compile(r"(?:-\s*)?Target hardware:\s*Hygon DCU", re.I)
HIP_TAIL = re.compile(r"not available\s+on this machine\)\.", re.I)
TRITON_LINE = re.compile(r"Target hardware:\s*Hygon DCU \(gfx936\)\.\s*Implementation language: Triton\.", re.I)

HIP_ONLY_TOKENS = re.compile(r"hipcc|hip/hip_runtime|CUDA_CU_SRC|CUDA_CPP_SRC|hipLaunchKernel|__global__", re.I)


def _wrap(sentence: str, indent: str = "", width: int = 78) -> str:
    return textwrap.fill(sentence, width=width, initial_indent=indent, subsequent_indent=indent)


def _replace_track_sentence(doc: str, track: str) -> str:
    """把 docstring 里的「目标硬件/实现语言」句子换成 `track` 的版本。"""
    if track == "hip":
        # 先抹掉已有的 Triton 单行句，或整段 HIP 句（4 行）
        if TRITON_LINE.search(doc):
            doc = TRITON_LINE.sub(HIP_SENTENCE_RAW, doc)
        else:
            m = HIP_START.search(doc)
            t = HIP_TAIL.search(doc, m.start()) if m else None
            if m and t:
                doc = doc[: m.start()] + HIP_SENTENCE_RAW + doc[t.end():]
    else:
        m = HIP_START.search(doc)
        t = HIP_TAIL.search(doc, m.start()) if m else None
        if m and t:
            doc = doc[: m.start()] + TRITON_SENTENCE + doc[t.end():]
        else:
            doc = TRITON_LINE.sub(TRITON_SENTENCE, doc)
    # 实现约束里的 "HIP kernel" / "Triton kernel" 跟着换
    if track == "hip":
        doc = doc.replace("Triton kernel", "HIP kernel").replace("Triton code", "HIP code")
    else:
        doc = doc.replace("HIP kernel", "Triton kernel").replace("HIP code", "Triton code")
    return doc


def _rewrite_file(src_text: str, track: str) -> str:
    tree = ast.parse(src_text)
    model = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Model"), None)
    if model is None:
        return src_text
    doc_node = model.body[0]
    old_doc = ast.get_docstring(model) or ""
    if not old_doc:
        return src_text
    new_doc = _replace_track_sentence(old_doc, track)
    if new_doc == old_doc:
        return src_text
    lines = src_text.splitlines(keepends=True)
    indent = " " * 4
    body = "\n".join((indent + l).rstrip() if l.strip() else "" for l in new_doc.splitlines())
    block = indent + '"""' + "\n" + body + "\n" + indent + '"""\n'
    return "".join(lines[: doc_node.lineno - 1]) + block + "".join(lines[doc_node.end_lineno:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".dcu_runs/dual", type=pathlib.Path)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    manifest_text = (COMPAT / "manifest_dcukb.yaml").read_text(encoding="utf-8")
    entries = re.findall(
        r"- alias:\s*(\S+)\s*\n\s*level:\s*(\d+)\s*\n\s*problem_id:\s*(\d+)(?:\s*\n\s*backend:\s*(\w+))?",
        manifest_text,
    )
    if not entries:
        print("manifest 解析失败")
        return 1

    totals = {}
    problems = []
    for track in ("triton", "hip"):
        manifest_lines = [f"name: dcukb_dual_{track}", f"backend: {'cuda' if track == 'hip' else 'triton'}", "tasks:"]
        n = 0
        for alias, level, problem_id, _src_backend in entries:
            src = TASKS / alias / "reference.py"
            if not src.exists():
                problems.append((track, alias, "缺 reference.py"))
                continue
            text = src.read_text(encoding="utf-8")
            new_text = _rewrite_file(text, track)
            # 校验：语法、CJK、跨轨残留
            try:
                ast.parse(new_text)
            except SyntaxError as exc:
                problems.append((track, alias, f"SyntaxError {exc}"))
                continue
            tree = ast.parse(new_text)
            doc = next((ast.get_docstring(n) for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Model"), "")
            if re.search(r"[\u4e00-\u9fff]", doc):
                problems.append((track, alias, "docstring 含 CJK"))
            if track == "triton" and HIP_ONLY_TOKENS.search(doc):
                problems.append((track, alias, f"Triton 题面残留 HIP 字样: {HIP_ONLY_TOKENS.search(doc).group(0)}"))
            if track == "hip" and not HIP_ONLY_TOKENS.search(doc):
                problems.append((track, alias, "HIP 题面缺 HIP 说明"))
            if args.write:
                dst = args.out / track / "kernelbench_compat" / f"level{level}"
                dst.mkdir(parents=True, exist_ok=True)
                (dst / f"{alias}.py").write_text(new_text, encoding="utf-8", newline="\n")
            n += 1
            manifest_lines.append(f"- alias: {alias}")
            manifest_lines.append(f"  level: {level}")
            manifest_lines.append(f"  problem_id: {problem_id}")
            if track == "hip":
                manifest_lines.append("  backend: cuda")
        totals[track] = n
        if args.write:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"manifest_{track}.yaml").write_text("\n".join(manifest_lines) + "\n",
                                                             encoding="utf-8", newline="\n")

    print(f"triton 轨: {totals['triton']} 道   hip 轨: {totals['hip']} 道   问题: {len(problems)}")
    for p in problems[:12]:
        print("  ", p)
    if args.write:
        print(f"已写 {args.out}/{{triton,hip}}/kernelbench_compat + manifest_*.yaml")
    else:
        print("（未加 --write，只校验不落盘）")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
