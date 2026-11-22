#!/usr/bin/env python3
"""从上游 KernelBench 官方题目目录生成清单（含分片），供 `run-kernelbench-batch` 使用。

只扫 `<kernelbench_root>/KernelBench/level{N}/`，alias 取文件名（去掉结尾下划线），
problem_id 取文件名前缀数字——和框架 loader 的 `_resolve_problem_path`（`glob "<pid>_*.py"`）
一一对应，因此不需要另建镜像树。

    python benchmark/tools/make_kernelbench_manifests.py \
        --kernelbench /path/to/KernelBench-main --level 1 \
        --tracks triton,cuda --shards 8 --out-dir /root/manifests/kb_l1

产出：
    l1_triton_all.json / l1_triton_s0.json ... / l1_cuda_all.json / l1_cuda_s0.json ...

分片按 problem_id 轮转切（不按文件名排序切），避免同一类算子全挤在一个分片里。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

PID_RE = re.compile(r"^(\d+)_")


def discover(level_dir: pathlib.Path) -> list[dict]:
    tasks = []
    for path in sorted(level_dir.glob("*.py")):
        match = PID_RE.match(path.name)
        if not match:
            continue
        tasks.append({
            "problem_id": int(match.group(1)),
            "alias": path.stem.rstrip("_"),
            "file": path.name,
        })
    tasks.sort(key=lambda item: item["problem_id"])
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernelbench", required=True, type=pathlib.Path, help="KernelBench 仓库根（含 KernelBench/levelN）")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--tracks", default="triton,cuda", help="逗号分隔，默认两轨")
    ap.add_argument("--shards", type=int, default=8, help="每轨分片数")
    ap.add_argument("--out-dir", required=True, type=pathlib.Path)
    ap.add_argument("--name-prefix", default="", help="清单名前缀，默认 l<level>")
    args = ap.parse_args()

    level_dir = args.kernelbench / "KernelBench" / f"level{args.level}"
    if not level_dir.is_dir():
        level_dir = args.kernelbench / f"level{args.level}"
    if not level_dir.is_dir():
        print(f"[FAIL] 找不到 level 目录: {level_dir}", file=sys.stderr)
        return 2

    tasks = discover(level_dir)
    if not tasks:
        print(f"[FAIL] {level_dir} 下没有 <id>_<name>.py", file=sys.stderr)
        return 2
    ids = [t["problem_id"] for t in tasks]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        print(f"[FAIL] problem_id 重复: {dupes}（loader 用 glob 取第一个，会串题）", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.name_prefix or f"l{args.level}"
    tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]
    shards = max(1, args.shards)

    for track in tracks:
        payload = {
            "name": f"{prefix}_{track}_all",
            "backend": track,
            "source": "upstream_kernelbench",
            "kernelbench_level_dir": str(level_dir),
            "tasks": [{"level": args.level, "problem_id": t["problem_id"], "alias": t["alias"]}
                      for t in tasks],
        }
        all_path = args.out_dir / f"{prefix}_{track}_all.json"
        all_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")

        groups: list[list[dict]] = [[] for _ in range(shards)]
        for index, task in enumerate(payload["tasks"]):
            groups[index % shards].append(task)
        for k, group in enumerate(groups):
            shard_path = args.out_dir / f"{prefix}_{track}_s{k}.json"
            shard_path.write_text(json.dumps({
                "name": f"{prefix}_{track}_s{k}",
                "backend": track,
                "tasks": group,
            }, indent=1, ensure_ascii=False), encoding="utf-8")
        sizes = [len(g) for g in groups]
        print(f"{track}: {len(payload['tasks'])} 题 → {shards} 片 {sizes}  (all={all_path.name})")

    print(f"首题 {tasks[0]['alias']}，末题 {tasks[-1]['alias']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
