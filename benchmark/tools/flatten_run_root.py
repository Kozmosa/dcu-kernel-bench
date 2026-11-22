#!/usr/bin/env python3
"""把「分片 → 题目」两层结构拍平成一层的「题目目录」，供 collect_metrics / audit_batch 使用。

这两个工具都按 `<alias>_l<L>_p<P>/` 直接找 `run.json` / `best_code.py`，而批量跑出来的结构是
`<run_root>/<shard>/<alias>_l<L>_p<P>/`。本工具用**硬链接**铺一层，不复制大文件、不动原数据。

    python benchmark/tools/flatten_run_root.py \
        --run-root /root/runs/kb_l1 --out /root/runs/kb_l1_flat \
        --shards "kn_l1_triton_s*"

之后：
    python benchmark/tools/collect_metrics.py --run-root /root/runs/kb_l1_flat
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys

FILES = ("run.json", "best_code.py", "candidate_code.py", "response.md")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", required=True, type=pathlib.Path, help="分片产物的根目录")
    ap.add_argument("--out", required=True, type=pathlib.Path, help="拍平后的输出目录（会被重建）")
    ap.add_argument("--shards", default="*", help="分片目录 glob，默认全部子目录")
    ap.add_argument("--copy", action="store_true", help="复制而不是硬链接（跨文件系统时用）")
    args = ap.parse_args()

    if not args.run_root.is_dir():
        print(f"[FAIL] run-root 不存在: {args.run_root}", file=sys.stderr)
        return 2
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    count = 0
    for shard in sorted(p for p in args.run_root.glob(args.shards) if p.is_dir()):
        for task_dir in sorted(p for p in shard.iterdir() if p.is_dir()):
            if not (task_dir / "run.json").exists():
                continue
            dest = args.out / task_dir.name
            if dest.exists():
                continue
            dest.mkdir()
            for name in FILES:
                src = task_dir / name
                if not src.exists():
                    continue
                if args.copy:
                    shutil.copy2(src, dest / name)
                else:
                    try:
                        os.link(src, dest / name)
                    except OSError:
                        shutil.copy2(src, dest / name)
            count += 1

    print(f"拍平 {count} 题 -> {args.out}")
    return 0 if count else 1


if __name__ == "__main__":
    sys.exit(main())
