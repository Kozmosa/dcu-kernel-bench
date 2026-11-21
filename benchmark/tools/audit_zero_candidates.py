#!/usr/bin/env python3
"""audit_zero_candidates.py —— 扫一遍 run 目录，找出"没有合法候选"的题。

为什么要它：一次全量跑的失败模式里，最贵的是"零候选"——不是模型写不出，而是
provider 报错（隧道断、429、超时）把候选预算吃光，最后 champion 退回 root（脚手架直通版）。
这种题在聚合指标里看起来像"模型不行"，实际是链路问题，必须单独揪出来重跑。

判据（每题一个 run.json）：
  candidates        除 root / phase2_root 之外的节点数
  provider_errors   run.json 的 stats.provider_errors
  champion          best_node_id；等于 root 说明没有候选胜过脚手架

用法：
    python3 benchmark/tools/audit_zero_candidates.py /root/runs2
    python3 benchmark/tools/audit_zero_candidates.py /root/runs2 --json /tmp/zero.json
退出码：0 = 没有零候选题；1 = 存在零候选题（供 supervisor / CI 用）。
"""
import argparse
import json
import pathlib
import sys

ROOT_NODES = ("root", "phase2_root")


def scan_run(path: pathlib.Path):
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    nodes = doc.get("nodes") or {}
    if isinstance(nodes, dict):
        items = list(nodes.items())
    else:
        items = [(n.get("node_id"), n) for n in nodes]
    candidates = [(nid, n) for nid, n in items if nid not in ROOT_NODES]
    stats = doc.get("stats") or {}
    correct = [nid for nid, n in candidates if n.get("correct")]
    champion = doc.get("best_node_id")
    return {
        "task": path.parent.name,
        "candidates": len(candidates),
        "correct_candidates": len(correct),
        "provider_errors": stats.get("provider_errors", 0),
        "attempts": stats.get("attempt_count"),
        "champion": champion,
        "champion_is_root": champion in ROOT_NODES,
        "failure_types": sorted({(n.get("failure_type") or "") for _, n in candidates if n.get("failure_type")}),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=pathlib.Path)
    ap.add_argument("--json", type=pathlib.Path, help="把逐题结果写到这个文件（JSON）")
    ap.add_argument("--only-bad", action="store_true", help="只打印有问题的题")
    args = ap.parse_args()

    rows = []
    for run_json in sorted(args.root.rglob("run.json")):
        rows.append(scan_run(run_json))
    if not rows:
        print(f"[warn] {args.root} 下没有 run.json", file=sys.stderr)
        return 0

    zero = [r for r in rows if r.get("candidates", 0) == 0]
    no_correct = [r for r in rows if r.get("candidates", 1) > 0 and r.get("correct_candidates", 0) == 0]
    provider = [r for r in rows if (r.get("provider_errors") or 0) > 0]
    root_champ = [r for r in rows if r.get("champion_is_root")]

    print(f"== 共 {len(rows)} 个 run")
    print(f"   零候选（root 之外没有任何节点）      : {len(zero)}")
    print(f"   有候选但都不正确                     : {len(no_correct)}")
    print(f"   含 provider 错误（隧道/429/超时）     : {len(provider)}")
    print(f"   champion = root（没有候选胜过脚手架） : {len(root_champ)}")

    def show(title, items):
        if not items:
            return
        print(f"\n-- {title}")
        for r in items:
            if "error" in r:
                print(f"   {r.get('task')}: 读文件失败 {r['error']}")
                continue
            print(f"   {r['task']:<40s} 候选={r['candidates']:<2d} 正确候选={r['correct_candidates']:<2d} "
                  f"provider_err={r['provider_errors']} attempts={r['attempts']} champion={r['champion']} "
                  f"types={','.join(t.split('.')[-1] for t in r['failure_types'][:3])}")

    if zero or not args.only_bad:
        show("零候选（必重跑）", zero)
    show("有候选但不正确（属模型问题，可重跑也可不跑）", no_correct)
    show("provider 错误（链路问题，优先重跑）", provider)

    if args.json:
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"\n逐题结果写到 {args.json}")
    return 1 if zero else 0


if __name__ == "__main__":
    raise SystemExit(main())
