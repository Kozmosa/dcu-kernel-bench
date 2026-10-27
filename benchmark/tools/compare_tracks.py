"""生成 HIP / Triton 双轨对照报告。

同一批题在两条实现轨上各跑一次（题面按轨改写，见 `make_dualtrack.py`），
本脚本把两边的 `run.json` 汇总成一张对照表 + 两轨指标 + 结论。

    python benchmark/tools/compare_tracks.py \
        --hip .dcu_runs/dual_hip --triton .dcu_runs/dual_triton \
        --out benchmark/tools/metrics_dual.md

口径与 `collect_metrics.py` 一致：候选排除 root；speedup 分母 `torch_eager`；
best = 所有 correct 候选里的最大 speedup；尝试序号 root=0、n1..nN、phase2_nN。
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics


def load(root: pathlib.Path) -> dict:
    out = {}
    for p in sorted(root.rglob("run.json")):
        alias = p.parent.name.rsplit("_l", 1)[0]
        d = json.loads(p.read_text(encoding="utf-8"))
        nodes = d.get("nodes") or {}
        nl = list(nodes.values()) if isinstance(nodes, dict) else nodes
        order = {}
        seq = 0
        for n in nl:
            nid = n.get("node_id") or n.get("id") or ""
            import re
            m2 = re.fullmatch(r"phase2_n(\d+)", nid)
            m1 = re.fullmatch(r"n(\d+)", nid)
            if nid in ("root", "phase2_root"):
                order[nid] = 0
            elif m2 or m1:
                seq += 1
                order[nid] = seq
            else:
                seq += 1
                order[nid] = seq
        cor = [(order.get(n.get("node_id") or n.get("id")), n) for n in nl
               if n.get("correct") and (n.get("node_id") or n.get("id")) not in ("root", "phase2_root")]
        cor.sort(key=lambda x: x[0])
        sp = [n.get("speedup") for _, n in cor if isinstance(n.get("speedup"), (int, float))]
        best = max(sp) if sp else None
        first_idx = cor[0][0] if cor else None
        first_sp = next((n.get("speedup") for _, n in cor if isinstance(n.get("speedup"), (int, float))), None)
        comp = any(n.get("compile_ok") for n in nl if (n.get("node_id") or n.get("id")) not in ("root", "phase2_root"))
        out[alias] = dict(compile_ok=comp, correct=bool(cor), best=best,
                          first_idx=first_idx, first_sp=first_sp,
                          attempts=max(order.values()) if order else 0)
    return out


def geomean(v):
    v = [x for x in v if isinstance(x, (int, float)) and x > 0]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def summary(D: dict) -> dict:
    cor = [a for a, r in D.items() if r["correct"]]
    sp = [D[a]["best"] for a in cor if isinstance(D[a]["best"], (int, float))]
    idx = [D[a]["first_idx"] for a in cor if D[a]["first_idx"] is not None]
    gains = [D[a]["best"] / D[a]["first_sp"] for a in cor
             if isinstance(D[a]["best"], (int, float)) and isinstance(D[a]["first_sp"], (int, float))
             and D[a]["first_sp"] > 0]
    n = len(D)

    def succ(k):
        return sum(1 for a in D if D[a]["first_idx"] is not None and D[a]["first_idx"] <= k)

    def fast(k):
        return sum(1 for a in D if isinstance(D[a]["best"], (int, float)) and D[a]["best"] > 1
                   and D[a]["first_idx"] is not None and D[a]["first_idx"] <= k)

    return dict(
        n=n, comp=sum(1 for a in D if D[a]["compile_ok"]), cor=len(cor),
        s1=succ(1), s5=succ(5), s10=succ(10),
        f1=fast(1), f5=fast(5), f10=fast(10),
        fast_all=sum(1 for a in D if isinstance(D[a]["best"], (int, float)) and D[a]["best"] > 1),
        gm=geomean(sp), med=statistics.median(sp) if sp else float("nan"),
        mean=statistics.mean(sp) if sp else float("nan"), best=max(sp) if sp else float("nan"),
        fcm=statistics.median(idx) if idx else float("nan"),
        rg=statistics.median(gains) if gains else float("nan"),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hip", required=True)
    ap.add_argument("--triton", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    H = load(pathlib.Path(args.hip))
    T = load(pathlib.Path(args.triton))
    SH, ST = summary(H), summary(T)
    L = []
    w = L.append
    w("# HIP / Triton 双轨对照")
    w("")
    out_d = f"  —  输出目录：`{args.out}`"
    if args.note:
        w(f"- {args.note}")
    w("- 每道题按同一份语义在两条轨上各跑一次；题面只改「目标硬件 + 实现语言 + 锚点」那几句"
      "（`benchmark/tools/make_dualtrack.py` 生成），语义/IO 契约/边界/禁用列表逐字相同。")
    w("- 口径与 `metrics.md` 一致：候选排除 root；分母 `torch_eager`；best = 所有 correct "
      "候选里的最大 speedup；尝试序号 root=0。**每格只跑了一次，逐题差异含采样噪声。**")
    w("")
    w("## 两轨总表")
    w("")
    w("| 指标 | HIP 轨 | Triton 轨 | 谁更好 |")
    w("|---|---|---|---|")
    def row(label, a, b, better="high", fmt="{:.2f}"):
        fa = fmt.format(a) if isinstance(a, (int, float)) else str(a)
        fb = fmt.format(b) if isinstance(b, (int, float)) else str(b)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not math.isnan(a) and not math.isnan(b):
            tag = "HIP" if ((a > b) if better == "high" else (a < b)) else ("Triton" if a != b else "持平")
        else:
            tag = "-"
        w(f"| {label} | **{fa}** | **{fb}** | {tag} |")
    def rate(label, kh, kt):
        ah, at = SH[kh], ST[kt]
        tag = "HIP" if ah > at else ("Triton" if at > ah else "持平")
        w(f"| {label} | **{ah}/{SH['n']}** | **{at}/{ST['n']}** | {tag} |")
    rate("Compile Rate", "comp", "comp")
    rate("Correct Rate", "cor", "cor")
    rate("Success@1", "s1", "s1")
    rate("Success@5", "s5", "s5")
    rate("Fast1@all", "fast_all", "fast_all")
    row("Geomean Speedup", SH["gm"], ST["gm"], fmt="{:.2f}x")
    row("Median Speedup", SH["med"], ST["med"], fmt="{:.2f}x")
    row("Mean Speedup", SH["mean"], ST["mean"], fmt="{:.2f}x")
    row("Best Speedup", SH["best"], ST["best"], fmt="{:.2f}x")
    row("First Correct Median", SH["fcm"], ST["fcm"], "low")
    row("Refine Gain Median", SH["rg"], ST["rg"], fmt="{:.2f}x")
    w("")
    both = [a for a in T if a in H and T[a]["correct"] and H[a]["correct"]]
    only_t = [a for a in T if a in H and T[a]["correct"] and not H[a]["correct"]]
    only_h = [a for a in T if a in H and not T[a]["correct"] and H[a]["correct"]]
    neither = [a for a in T if a in H and not T[a]["correct"] and not H[a]["correct"]]
    w(f"两轨都写对 **{len(both)}** 道；只 Triton 对 {len(only_t)} 道"
      f"{('（' + '、'.join(f'`{x}`' for x in only_t) + '）') if only_t else ''}；"
      f"只 HIP 对 {len(only_h)} 道"
      f"{('（' + '、'.join(f'`{x}`' for x in only_h) + '）') if only_h else ''}；"
      f"都不对 {len(neither)} 道"
      f"{('（' + '、'.join(f'`{x}`' for x in neither) + '）') if neither else ''}。")
    w("")
    wins_h = sum(1 for a in both if isinstance(H[a]["best"], (int, float)) and isinstance(T[a]["best"], (int, float))
                 and H[a]["best"] > T[a]["best"])
    w(f"两轨都写对的 {len(both)} 道里，**HIP 更快 {wins_h} 道、Triton 更快 {len(both) - wins_h} 道**。")
    w("")
    w("## 逐题对照")
    w("")
    w("| 题 | HIP 正确 | HIP best | Triton 正确 | Triton best | Triton/HIP |")
    w("|---|---|---|---|---|---|")
    for a in sorted(T, key=lambda x: -(H.get(x, {}).get("best") or 0)):
        h = H.get(a, {})
        t = T[a]
        hb = f"{h.get('best'):.2f}x" if isinstance(h.get("best"), (int, float)) else "-"
        tb = f"{t['best']:.2f}x" if isinstance(t.get("best"), (int, float)) else "-"
        ratio = (t["best"] / h["best"]) if isinstance(t.get("best"), (int, float)) and isinstance(h.get("best"), (int, float)) and h["best"] else None
        w(f"| `{a}` | {'✓' if h.get('correct') else '✗'} | {hb} | {'✓' if t['correct'] else '✗'} | {tb} | "
          f"{f'{ratio:.2f}' if ratio else '-'} |")
    w("")
    pathlib.Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"已写 {args.out}（{len(L)} 行）")
    print(f"  HIP   : correct {SH['cor']}/{SH['n']}  median {SH['med']:.2f}x  best {SH['best']:.2f}x")
    print(f"  Triton: correct {ST['cor']}/{ST['n']}  median {ST['med']:.2f}x  best {ST['best']:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
