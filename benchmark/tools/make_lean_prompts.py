"""生成"精简题面"变体：把可由参考实现直接看出来的段落从 `Model` docstring 里删掉。

为什么可以删：PyramidKernel 的 root 脚手架里放的**就是参考实现本身**（`Model.forward`
与它用到的辅助方法都在 `ModelNew` 里），所以 docstring 里逐字复述的数学定义、I/O 布局、
打包协议，对 Agent 来说是第二遍——它可以直接读代码。

必须保留的三类（代码里推不出来）：
  1. **禁用列表** —— 这是政策，静态审计会硬判负；
  2. **目标硬件 / 实现语言** —— DCU、Triton 还是 HIP；
  3. **隐藏 case 的 shape/dtype 域** —— 评测会用更大/非 2 次幂/非对齐的 shape，
     不告诉它就等于让 Agent 按公开 shape 写死（本轮 `1016`/`1022` 就是这么崩的）。

本脚本只改 docstring 文本，代码一字不动；产物写到独立目录 + 独立 manifest，
这样能和原题面并排跑同一模型、同一 profile 做 A/B。

用法：
    python benchmark/tools/make_lean_prompts.py --out .dcu_runs/lean/benchmark
    # 生成 <out>/tasks/*/reference.py、<out>/kernelbench_compat/levelN/*.py 与 manifest
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]
TASKS = REPO / "benchmark" / "tasks"
COMPAT = REPO / "benchmark" / "kernelbench_compat"
DIFFICULTY_TO_LEVEL = {"basic": 1, "medium": 2, "hard": 3}

# 「可由代码替代」的部分：从这些标题开始，到下一个 KEEP 标题结束
DROP_START = re.compile(
    r"^\s*(Mathematical definition|Math(?:ematical)? semantics|Semantics|"
    r"Input/output specification:|I/O specification:|Packing protocol)",
    re.I,
)
KEEP_START = re.compile(
    r"^\s*(Implementation constraints|Forbidden \(|Disallowed list|"
    r"Target hardware|Task scope:|Boundary behaviour:|Global-domain constraints|"
    r"Notes|Reference:)",
    re.I,
)


def trim_docstring(doc: str) -> str:
    """删掉 DROP 段落（数学定义 / I-O 规格），其余原样保留。"""
    lines = doc.splitlines()
    out, dropping = [], False
    for line in lines:
        if DROP_START.match(line):
            dropping = True
            continue
        if dropping and KEEP_START.match(line):
            dropping = False
        if not dropping:
            out.append(line)
    # 压掉连续空行
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text).strip("\n")
    return text


def replace_docstring(src: str, new_doc: str) -> str:
    """把 `Model` 类 docstring 换成 new_doc，代码其余部分逐字保留。"""
    tree = ast.parse(src)
    model = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Model"), None)
    if model is None:
        return src
    node = model.body[0]
    if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)):
        return src
    lines = src.splitlines(keepends=True)
    start = node.lineno - 1
    end = node.end_lineno            # 1-based inclusive
    indent = " " * 4
    body = "\n".join(indent + l if l.strip() else "" for l in new_doc.splitlines())
    new_block = f'{indent}"""{new_doc.splitlines()[0]}\n' if new_doc else f'{indent}"""\n'
    # 简单起见：整段重写为三引号块，保持缩进
    new_block = indent + '"""' + "\n" + body + "\n" + indent + '"""\n'
    return "".join(lines[:start]) + new_block + "".join(lines[end:])


def task_meta(task_dir: pathlib.Path) -> dict:
    text = (task_dir / "task.yaml").read_text(encoding="utf-8")
    meta = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z_]\w*):\s*(.*?)\s*$", line)
        if m and m.group(1) in ("entry", "difficulty"):
            meta[m.group(1)] = m.group(2).split("#", 1)[0].strip()
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=pathlib.Path,
                    help="输出根目录（会建 tasks/ 与 kernelbench_compat/）")
    ap.add_argument("--write", action="store_true", help="真正写文件（否则只报告）")
    args = ap.parse_args()

    out_tasks = args.out / "tasks"
    out_compat = args.out / "kernelbench_compat"
    total_before = total_after = 0
    kept = []
    for task_dir in sorted(TASKS.iterdir()):
        ref = task_dir / "reference.py"
        if not task_dir.is_dir() or not ref.exists():
            continue
        src = ref.read_text(encoding="utf-8")
        tree = ast.parse(src)
        model = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Model"), None)
        if model is None:
            continue
        doc = ast.get_docstring(model) or ""
        if not doc:
            continue
        lean = trim_docstring(doc)
        total_before += len(doc)
        total_after += len(lean)
        kept.append((task_dir.name, len(doc), len(lean)))
        if args.write:
            new_src = replace_docstring(src, lean)
            ast.parse(new_src)                      # 必须仍是合法 Python
            dst = out_tasks / task_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            (dst / "reference.py").write_text(new_src, encoding="utf-8", newline="\n")
            meta = task_meta(task_dir)
            if meta.get("entry") == "model_class" and meta.get("difficulty") in DIFFICULTY_TO_LEVEL:
                lvl = DIFFICULTY_TO_LEVEL[meta["difficulty"]]
                (out_compat / f"level{lvl}").mkdir(parents=True, exist_ok=True)
                (out_compat / f"level{lvl}" / f"{task_dir.name}.py").write_text(
                    new_src, encoding="utf-8", newline="\n")

    for name, before, after in sorted(kept, key=lambda x: -(x[1] - x[2]))[:8]:
        print(f"  {name:44} {before:5} -> {after:5}  (-{before - after})")
    print(f"\n{len(kept)} 道题：docstring {total_before} -> {total_after} 字符"
          f"（-{(total_before - total_after) / total_before * 100:.1f}%）")
    if args.write:
        # manifest：与 benchmark/kernelbench_compat/manifest_dcukb.yaml 同构
        src_man = (COMPAT / "manifest_dcukb.yaml").read_text(encoding="utf-8")
        out_man = re.sub(r"alias:\s*(\S+)", r"alias: \1", src_man)
        (args.out / "manifest_lean.yaml").write_text(out_man, encoding="utf-8", newline="\n")
        print(f"已写 {args.out}（tasks/ + kernelbench_compat/ + manifest_lean.yaml）")
    else:
        print("（未加 --write，只报告不落盘）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
