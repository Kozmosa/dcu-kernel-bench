#!/usr/bin/env python3
"""migrate_target_arch.py —— 把题面里的"目标硬件架构"整体迁移到另一张卡。

用途：换机器/换架构时，`Model` docstring 里那句
    Target hardware: Hygon DCU (<arch>). Implementation language: Triton.
必须跟着改，否则题面在骗模型（不同卡的计算单元数、共享内存、带宽都不同）。本脚本只改
**括号里的架构串**，其余一字不动；并同时改 `benchmark/kernelbench_compat/` 的镜像，
保持"镜像与 tasks 逐字节一致"（tests/test_compat_mirrors.py 强制）。

同时改掉生成侧模板里的架构串：`make_dualtrack.py`（双轨题面替换）、`pure_bench.py`
（纯 API 提示）、`dcukb.py`（新任务模板）。

用法：
    python benchmark/tools/migrate_target_arch.py gfx928 --dry-run
    python benchmark/tools/migrate_target_arch.py gfx928
    python benchmark/tools/migrate_target_arch.py gfx928 --only-tools   # 只改工具模板
"""
import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SENTENCE = re.compile(r"(?i)(Target hardware:\s*Hygon DCU \()([^)]*)(\))")
TOOL_FILES = ("benchmark/tools/make_dualtrack.py", "benchmark/tools/pure_bench.py", "benchmark/tools/dcukb.py")


def rewrite(path: Path, arch: str) -> tuple[str, int]:
    text = path.read_text(encoding="utf-8")
    count = 0

    def repl(m):
        nonlocal count
        if m.group(2).strip().lower() == arch.lower():
            return m.group(0)          # 已经是目标架构，不计入改动
        count += 1
        return m.group(1) + arch + m.group(3)

    return SENTENCE.sub(repl, text), count


def mirror_for(task_id: str) -> Path | None:
    base = ROOT / "benchmark" / "kernelbench_compat"
    hits = sorted(p for p in base.glob("level*/*.py") if p.name.startswith(task_id + "_"))
    return hits[0] if len(hits) == 1 else (None if not hits else hits[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arch", help="新的架构串，例如 gfx928")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-tools", action="store_true", help="只改工具模板，不动题目文件")
    args = parser.parse_args()
    arch = args.arch

    changed_files = changed_sentences = 0
    if not args.only_tools:
        for task_dir in sorted((ROOT / "benchmark" / "tasks").iterdir()):
            ref = task_dir / "reference.py"
            if not ref.is_file():
                continue
            new, count = rewrite(ref, arch)
            if not count:
                continue
            mirror = mirror_for(task_dir.name.split("_")[0])
            targets = [(ref, new)]
            if mirror is not None:
                m_new, m_count = rewrite(mirror, arch)
                if m_count:
                    targets.append((mirror, m_new))
            for path, content in targets:
                print(f"  {path.relative_to(ROOT)}: {count} 处")
                if not args.dry_run:
                    path.write_text(content, encoding="utf-8")
            changed_files += len(targets)
            changed_sentences += count

    for rel in TOOL_FILES:
        path = ROOT / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        new = text
        # 工具模板里的架构串不是"Target hardware"句，按字面替换两种旧写法
        for old in ("gfx936/gfx938", "gfx936", "gfx938"):
            new = new.replace(old, arch)
        if new != text:
            print(f"  {rel}: 字面替换架构串")
            if not args.dry_run:
                path.write_text(new, encoding="utf-8")
            changed_files += 1

    print(f"\n{'[dry-run] ' if args.dry_run else ''}改了 {changed_files} 个文件、{changed_sentences} 处题面句")
    if not args.dry_run and not args.only_tools:
        print("提醒：跑一遍 `python -m pytest tests/test_compat_mirrors.py tests/test_prompt_english.py`"
              "（或本机垫片）确认镜像字节一致、题面无 CJK。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
