"""审计 66 道题的 Agent 可见题面（`Model` 的 docstring），找三类问题。

题面的唯一载体是 `benchmark/tasks/<id>/reference.py` 里 `Model` 类的 docstring；
镜像树 `kernelbench_compat/level{N}/` 里那份与之逐字节一致（tests/test_compat_mirrors.py 强制）。
一个设计良好的题面应当清楚说明「**做什么**」（语义、输入输出规格、数值契约、边界行为、
实现约束与禁用列表），而**不说明「怎么做」**（算法选型、分块尺寸、实现步骤）。

本工具只读、只报告，不做判定也不改文件。三类检查：

  A 后端口径不一致：题面声明的实现语言与 manifest 的 backend 对不上，或题面里
    残留另一条 track 的 API（HIP 题禁用清单里留 `tl.*`、Triton 题里说 `hipcc`），
    Agent 会被引到错误的实现路径。
  B 溯源泄漏：题面出现 aiter / OpenDAS / hipdnn_samples / third_party / private/ /
    commit sha / sourcefind 等 —— 违反铁律 1（源码隔离）。
  C 解法暴露：题面点名了具体算法、分块尺寸、线程组织或实现步骤。

用法：
    python benchmark/tools/audit_prompts.py                 # 汇总 + 逐题表
    python benchmark/tools/audit_prompts.py --detail A       # 展开某一类的明细
    python benchmark/tools/audit_prompts.py --json out.json
"""

import argparse
import ast
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]

# ---- B 溯源词 ---------------------------------------------------------------
TRACE_PATTERNS = {
    "aiter": r"aiter",
    "OpenDAS": r"OpenDAS",
    "hipdnn_samples": r"hipdnn_samples",
    "third_party": r"third_party",
    "private_path": r"\bprivate/",
    # 必须含至少一个十六进制**字母**，否则会误命中数学常量：
    # 0.7071067811865476（1/√2）、0.7978845608028654（√(2/π)）等纯数字串
    "commit_sha": r"\b(?=[0-9a-f]{7,40}\b)[0-9a-f]*[a-f][0-9a-f]*\b",
    "sourcefind": r"sourcefind",
    "operator_catalog": r"operator_catalog",
}

# 「禁止/禁用」段落里的词是**题面应该写**的内容（禁用列表），不算溯源泄漏。
# 注意必须按**段落**判断而非按行：禁用清单经常换行，关键词落在上一行。
FORBIDDEN_HINT = re.compile(r"禁止|禁用|不得|不许|forbidden|违规", re.I)


def _forbidden_spans(doc: str) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for block in re.split(r"(\n\s*\n)", doc):
        if FORBIDDEN_HINT.search(block):
            spans.append((pos, pos + len(block)))
        pos += len(block)
    return spans


def _in_spans(index: int, spans: list[tuple[int, int]]) -> bool:
    return any(lo <= index < hi for lo, hi in spans)

# ---- C 解法暴露词（点名"怎么做"，而非"做什么"）------------------------------
PRESCRIPTION_PATTERNS = {
    "算法名": r"Hillis[- ]?Steele|Blelloch|Kahan|Welford|Kogge[- ]?Stone|flash\s*attention\s*算法",
    "分块/线程组织": r"BLOCK_[A-Z_]+\s*=|num_warps|num_stages|block_size\s*=|线程块|grid\s*=\s*\(",
    "实现步骤": r"第一步|第二步|步骤\s*[1-9]|先.{0,12}再.{0,12}最后",
    "性能技巧": r"双缓冲|软件流水|向量化加载|合并访存|shared\s*memory\s*复用|bank\s*conflict",
    "推荐做法": r"推荐用|建议使用|应当采用|最优做法",
}


def load_manifest_backends() -> dict:
    import yaml

    man = yaml.safe_load(
        (REPO / "benchmark/kernelbench_compat/manifest_dcukb.yaml").read_text(encoding="utf-8")
    )
    out = {}
    for t in man["tasks"]:
        out[t["alias"]] = (t.get("backend") or "triton")
    return out


def get_docstring(path: pathlib.Path) -> str:
    src = path.read_text(encoding="utf-8")
    module = ast.parse(src, filename=str(path))
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            return ast.get_docstring(node) or ""
    return ""


def audit_one(alias: str, backend: str) -> dict:
    path = REPO / "benchmark" / "tasks" / alias / "reference.py"
    rec = {"alias": alias, "backend": backend, "problems": [], "path": str(path)}
    if not path.is_file():
        rec["problems"].append(("A", "缺 reference.py"))
        return rec
    doc = get_docstring(path)
    rec["doc_chars"] = len(doc)
    rec["docstring"] = doc
    if not doc:
        rec["problems"].append(("A", "Model 没有 docstring（Agent 看不到题面）"))
        return rec

    low = doc.lower()
    # --- A 后端口径 ---------------------------------------------------------
    m = re.search(r"实现语言\s*([A-Za-z]+)", doc)
    lang = m.group(1).lower() if m else None
    rec["declared_lang"] = lang
    if lang is None:
        rec["problems"].append(("A", "题面未声明实现语言"))
    else:
        want = "hip" if backend == "cuda" else backend
        if want not in lang and lang not in want:
            rec["problems"].append(("A", f"题面声明「{lang}」但 manifest backend={backend}"))
    # 跨 track 残留
    if backend == "cuda":
        hits = sorted(set(re.findall(r"tl\.[a-z_]+|triton[a-z_.]*", low)))
        if hits:
            rec["problems"].append(("A", f"HIP 题里出现 Triton API/字样: {hits[:6]}"))
    else:
        hits = sorted(set(re.findall(r"hip/hip_runtime|hipcc|hipLaunchKernel|__global__", low)))
        if hits:
            rec["problems"].append(("A", f"Triton 题里出现 HIP 字样: {hits[:6]}"))
    if "目标硬件" not in doc:
        rec["problems"].append(("A", "题面缺「目标硬件」声明"))

    # --- B 溯源泄漏 ---------------------------------------------------------
    exempt = _forbidden_spans(doc)
    for name, pat in TRACE_PATTERNS.items():
        found = sorted({m.group(0) for m in re.finditer(pat, doc) if not _in_spans(m.start(), exempt)})
        if found:
            rec["problems"].append(("B", f"溯源词 {name}: {found[:4]}"))

    # --- C 解法暴露 ---------------------------------------------------------
    for name, pat in PRESCRIPTION_PATTERNS.items():
        found = re.findall(pat, doc)
        if found:
            rec["problems"].append(("C", f"{name}: {found[:4]}"))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", default="", help="展开某一类明细：A / B / C 或 all")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    backends = load_manifest_backends()
    recs = [audit_one(a, b) for a, b in sorted(backends.items(), key=lambda kv: int(kv[0].split("_")[0]))]

    print("=" * 118)
    print(f"题面审计：{len(recs)} 道题   （manifest: {sum(1 for b in backends.values() if b == 'cuda')} HIP / "
          f"{sum(1 for b in backends.values() if b != 'cuda')} Triton）")
    print("=" * 118)
    print(f"{'题名':<40} {'backend':<8} {'字数':<6} {'A后端':<6} {'B溯源':<6} {'C解法':<6} 说明")
    print("-" * 118)
    counts = {"A": 0, "B": 0, "C": 0}
    for r in recs:
        kinds = [k for k, _ in r["problems"]]
        for k in counts:
            if k in kinds:
                counts[k] += 1
        first = r["problems"][0][1] if r["problems"] else ""
        print(f"{r['alias'][:39]:<40} {r['backend']:<8} {r.get('doc_chars', 0):<6} "
              f"{('有' if 'A' in kinds else '-'):<6} {('有' if 'B' in kinds else '-'):<6} "
              f"{('有' if 'C' in kinds else '-'):<6} {first[:44]}")
    print("-" * 118)
    print(f"有问题的题: A(后端口径) {counts['A']} 道   B(溯源泄漏) {counts['B']} 道   C(解法暴露) {counts['C']} 道")

    if args.detail:
        want = [args.detail.upper()] if args.detail.upper() in counts else list(counts)
        for kind in want:
            rows = [(r, p) for r in recs for p in r["problems"] if p[0] == kind]
            print()
            print("=" * 118)
            titles = {"A": "A 后端口径不一致", "B": "B 溯源泄漏", "C": "C 解法暴露"}
            print(f"{titles[kind]}：{len(rows)} 处")
            print("=" * 118)
            for r, (_, msg) in rows:
                print(f"  {r['alias']:<40} {msg}")

    if args.json:
        slim = [{k: v for k, v in r.items() if k != "docstring"} for r in recs]
        pathlib.Path(args.json).write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n明细已写 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
