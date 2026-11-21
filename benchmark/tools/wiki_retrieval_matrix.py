"""开跑前验收：逐题打印 DCU Wiki 会发给生成侧的页面，并给出覆盖面统计。

判据来自框架自己的检索实现（`PublicKernelWiki.retrieve`），所以这不是另写一套规则，
而是把「这一轮实际会看到什么知识」提前摊开：

  python benchmark/tools/wiki_retrieval_matrix.py \
    --kw-root ../KernelWeaver --tasks benchmark/tasks --json /tmp/wiki_matrix.json

退出码：有题目拿不到硬件页、或某题一条页面都没有，就以非零退出（开跑前应当为 0）。
`--allow <task_id>` 可把已知「按设计走通用检查单」的题排除在硬失败之外。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HARDWARE_ID = "dcu-runtime-contract-928"
COMPLIANCE_ID = "public-compliance"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kw-root", type=Path, default=Path(__file__).resolve().parents[3] / "KernelWeaver")
    ap.add_argument("--wiki", type=Path, help="语料目录，缺省为 <kw-root>/knowledge/kernelwiki_dcu")
    ap.add_argument("--tasks", type=Path, default=Path(__file__).resolve().parents[1] / "tasks")
    ap.add_argument("--target-arch", default="gfx928")
    ap.add_argument("--backend", default="triton")
    ap.add_argument("--json", type=Path, help="把逐题结果写成 JSON")
    ap.add_argument("--allow", nargs="*", default=[], help="允许只拿通用检查单的题")
    args = ap.parse_args()

    kw_root = args.kw_root.resolve()
    wiki_path = (args.wiki or kw_root / "knowledge" / "kernelwiki_dcu").resolve()
    sys.path.insert(0, str(kw_root.parent))
    from KernelWeaver.src.knowledge.public_wiki import PublicKernelWiki  # noqa: PLC0415

    wiki = PublicKernelWiki(str(wiki_path), target_arch=args.target_arch)

    class Task:
        backend = args.backend
        name = "probe"
        description = ""

        def __init__(self, code: str):
            self.reference_code = code

    print(f"corpus {wiki.corpus_sha256}  pages {len(wiki.pages)}  max_entries {wiki.max_entries}  "
          f"backend {args.backend}  arch {args.target_arch}")

    rows, missing_hw, no_page, contract_only = [], [], [], []
    for task_dir in sorted(p for p in args.tasks.iterdir() if p.is_dir()):
        reference = task_dir / "reference.py"
        if not reference.exists():
            continue
        task = Task(reference.read_text(encoding="utf-8"))
        families = sorted(wiki.task_families(task))
        evidence = wiki.retrieve(task)
        ids = [item["id"] for item in evidence]
        rows.append({"task": task_dir.name, "families": families, "pages": ids})
        if HARDWARE_ID not in ids:
            missing_hw.append(task_dir.name)
        if not ids:
            no_page.append(task_dir.name)
        elif [i for i in ids if i not in (HARDWARE_ID, COMPLIANCE_ID)] in ([], ["public-contract"]):
            contract_only.append(task_dir.name)

    width = max((len(row["task"]) for row in rows), default=8)
    for row in rows:
        print(f"{row['task']:<{width}}  {','.join(row['families']) or '-':<44}  {' | '.join(row['pages'])}")

    allowed = set(args.allow)
    print(f"\n共 {len(rows)} 题：硬件页 {len(rows) - len(missing_hw)}/{len(rows)}，"
          f"退通用检查单 {len(contract_only)} 题{('（' + ', '.join(contract_only) + '）') if contract_only else ''}")
    if args.json:
        args.json.write_text(json.dumps({"corpus_sha256": wiki.corpus_sha256,
                                         "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"明细已写入 {args.json}")

    failed = []
    if missing_hw:
        failed.append(f"缺硬件页：{', '.join(missing_hw)}")
    if no_page:
        failed.append(f"完全无页面：{', '.join(no_page)}")
    unexpected = [t for t in contract_only if t not in allowed]
    if unexpected:
        failed.append(f"退通用检查单但未在 --allow 里声明：{', '.join(unexpected)}")
    for line in failed:
        print("FAIL", line)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
