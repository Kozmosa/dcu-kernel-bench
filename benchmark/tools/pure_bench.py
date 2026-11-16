"""纯 API 直出基线：不套任何搜索框架，一次调用生成完整 ModelNew，然后用同一评测器打分。

与 r3（KernelWeaver d 组）严格对齐的部分：
  - 同一批题（manifest 里 66 题，HIP/Triton 按 backend 分别跑）
  - 同一评测器（KernelBenchPaperEvaluator）、同一测量配置（configs/evaluation 下的 main）
  - 同一指标口径（collect_metrics 直接吃这里产出的 run.json）

不同之处：没有 Plan/Code/Debug 三 agent、没有记忆、没有 Wiki、没有多轮候选 —— 每题只调用一次。

用法：
  python3 pure_bench.py --manifest manifest_hip.yaml --backend cuda --out pure/hip --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CODE_ROOT = Path(os.environ.get("KW_CODE", str(Path(__file__).resolve().parents[3] / "KernelWeaver")))
sys.path.insert(0, str(CODE_ROOT))

import yaml  # noqa: E402

PROMPT_SYSTEM = (
    "You are a high-performance kernel engineer targeting a Hygon DCU accelerator (gfx936, DTK 26.04). "
    "You write complete, correct, runnable code. Reply with code only."
)

PROMPT_USER = """Rewrite the following operator so it runs faster on the target accelerator.

Requirements:
- Produce a complete, self-contained Python module.
- It must define `class ModelNew` with **the same `__init__` signature** as `Model` below, and a `forward`
  that accepts exactly the same arguments and returns a single torch.Tensor with the same shape/dtype/values
  (preserve native input dtypes; use the DCUKB task tolerance contract below).
- Do NOT call torch.matmul / torch.softmax / torch.nn.functional.softmax / conv2d / torch.mm / F.linear /
  torch.cumsum / torch.autograd in the core computation. Write the compute yourself
  ({lang}), and you may use torch only for allocation, shape queries and launch bookkeeping.
- Assume CUDA-style device naming (torch.cuda); the runtime is HIP on this device.
- Output ONE ```python code block and nothing else. No explanation.

## Reference implementation (semantics to preserve)

```python
{source}
```
"""


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.S)
    code = max(blocks, key=len) if blocks else text
    return code.strip() + "\n"


def build_candidate(code: str) -> str:
    """保证候选里确有 ModelNew；缺了就原样返回，由评测器判负。"""
    return code


def wiki_context(task) -> str:
    """取 DCU Wiki 的检索结果（含常驻的合规页），拼成可注入提示词的文本。"""
    from src.knowledge.public_wiki import PublicKernelWiki

    wiki = PublicKernelWiki(CODE_ROOT / "knowledge/kernelwiki_dcu", target_arch="gfx936")
    pages = wiki.retrieve(task)
    blocks = []
    for page in pages:
        f = page.get("frontmatter") or {}
        body = page.get("body") or ""
        blocks.append("### " + str(f.get("title", f.get("id", "?"))) + "\n" + body.strip())
    return "\n\n".join(blocks)


def call_llm(task, lang: str, timeout: int, temperature: float, extra: str = "") -> tuple[str, dict]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"],
                    timeout=timeout, max_retries=0)
    body = {
        "model": os.environ.get("OPENAI_MODEL", "gpt-6-astra"),
        "messages": [
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": PROMPT_USER.format(source=task.reference_code, lang=lang) + "\n## Task contract\n" + task.description
             + (("\n## Platform knowledge (DCU Wiki)\n" + extra) if extra else "")},
        ],
        "temperature": temperature,
        "max_tokens": 8192,
    }
    # 必须流式：经 Clash 的隧道对非流式请求 60 秒无数据即掐断（Server disconnected）。
    last = None
    for attempt in range(6):
        try:
            t0 = time.time()
            stream = client.chat.completions.create(**body, stream=True,
                                                    stream_options={"include_usage": True})
            parts: list[str] = []
            usage_obj = None
            for chunk in stream:
                u = getattr(chunk, "usage", None)
                if u is not None:
                    usage_obj = u
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                piece = getattr(delta, "content", None) if delta is not None else None
                if piece:
                    parts.append(piece)
            text = "".join(parts)
            break
        except Exception as exc:  # noqa: BLE001 - 隧道抖动时重试
            last = exc
            if attempt == 5:
                raise
            time.sleep(15 * (attempt + 1))
    else:
        raise last
    usage = {
        "seconds": round(time.time() - t0, 1),
        "input_tokens": getattr(usage_obj, "prompt_tokens", None),
        "output_tokens": getattr(usage_obj, "completion_tokens", None),
    }
    return text, usage


def eval_config():
    from src.models import StarkConfig

    measurement = yaml.safe_load((CODE_ROOT / "configs/evaluation/measurement/main.yaml").read_text())
    evaluator = yaml.safe_load((CODE_ROOT / "configs/evaluation/evaluators/main.yaml").read_text())
    return StarkConfig(
        max_attempts=1,
        benchmark_loops=int(measurement["benchmark_loops"]),
        warmup_loops=int(measurement["warmup_loops"]),
    ), measurement, evaluator


def run_one(item: dict, args, kernelbench_root: Path, pure_root: Path) -> dict:
    from src.core.loader import KernelBenchLoader
    from src.evaluation.evaluator_paper import KernelBenchPaperEvaluator

    alias, level, pid = item["alias"], int(item["level"]), int(item["problem_id"])
    backend = item.get("backend") or args.backend
    out_dir = pure_root / alias
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"task_name": alias, "level": level, "problem_id": pid, "backend": backend,
              "entry_kind": "model_class", "workflow": "pure_api_single_shot",
              "nodes": {}, "best_node_id": None, "primary_reference": "torch_eager"}

    expected_root = Path(__file__).resolve().parents[2]
    if Path(os.environ.get("DCUKB_ROOT", "")).resolve() != expected_root:
        raise ValueError(f"Set DCUKB_ROOT={expected_root} to activate native public evaluation")
    loader = KernelBenchLoader()
    task = loader.load_official_problem(kernelbench_root, level, pid, backend=backend,
                                        semantics_enabled=False)
    lang = "HIP C++ via load_inline" if backend == "cuda" else "Triton"
    extra = wiki_context(task) if args.wiki else ""
    text, usage = call_llm(task, lang, args.timeout, args.temperature, extra)
    (out_dir / "response.md").write_text(text, encoding="utf-8")
    code = extract_code(text)
    (out_dir / "candidate_code.py").write_text(code, encoding="utf-8")
    (out_dir / "best_code.py").unlink(missing_ok=True)

    cfg, measurement, evaluator_cfg = args._cfg
    ev = KernelBenchPaperEvaluator()
    evaluation = ev.evaluate(task, code, cfg)

    node = {
        "node_id": "n1", "parent_id": "root", "depth": 1, "code": code,
        "origin": "plan_code", "plan_strategy_name": "direct_one_shot",
        "compile_ok": evaluation.compile_ok, "correct": evaluation.correct,
        "runtime": evaluation.runtime, "score": evaluation.score,
        "failure_type": evaluation.failure_type, "failure_stage": evaluation.failure_stage,
        "reference_runtime": evaluation.reference_runtime, "speedup": evaluation.speedup,
        "reference_runtimes": evaluation.reference_runtimes, "speedups": evaluation.speedups,
        "primary_reference": evaluation.primary_reference,
        "debug_attempts": 0, "llm_usage": usage,
    }
    result["nodes"] = {"n1": node}
    result["best_node_id"] = "n1" if evaluation.compile_ok and evaluation.correct else None
    result["stats"] = {"search_status": "ok" if result["best_node_id"] else "no_valid_candidate"}
    if result["best_node_id"]:
        (out_dir / "best_code.py").write_text(code, encoding="utf-8")
    result["config"] = {"max_attempts": 1, "benchmark_loops": measurement["benchmark_loops"],
                        "warmup_loops": measurement["warmup_loops"]}
    result["measurement"] = measurement
    result["evaluator"] = evaluator_cfg
    (out_dir / "run.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"task": alias, "compile_ok": evaluation.compile_ok, "correct": evaluation.correct,
            "speedup": evaluation.speedup, "failure": evaluation.failure_type, "usage": usage}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--kernelbench-root", required=True)
    ap.add_argument("--backend", default="cuda", choices=["cuda", "triton"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--missing", action="store_true", help="只跑还没有 run.json 的题（补跑）")
    ap.add_argument("--wiki", action="store_true", help="把 DCU Wiki 检索结果注入提示词（隔离知识 vs 搜索的对照）")
    args = ap.parse_args()

    manifest = yaml.safe_load(Path(args.manifest).read_text())
    items = manifest["tasks"]
    if args.limit:
        items = items[:args.limit]
    pure_root = Path(args.out)
    pure_root.mkdir(parents=True, exist_ok=True)
    if args.missing:
        items = [it for it in items if not (pure_root / it["alias"] / "run.json").exists()]
        print("补跑模式：剩余 %d 题" % len(items), flush=True)
    args._cfg = eval_config()

    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, item, args, Path(args.kernelbench_root), pure_root): item
                   for item in items}
        for fut in as_completed(futures):
            item = futures[fut]
            done += 1
            try:
                row = fut.result()
                print("[%d/%d] %-46s compile=%-5s correct=%-5s speedup=%s %s"
                      % (done, len(items), row["task"], row["compile_ok"], row["correct"],
                         ("%.2fx" % row["speedup"]) if row["speedup"] else "-", row["failure"] or ""),
                      flush=True)
            except Exception as exc:  # noqa: BLE001
                import traceback
                print("[%d/%d] %-46s ERROR %s: %s" % (done, len(items), item["alias"],
                                                      type(exc).__name__, str(exc)[:160]), flush=True)
                print("  cause:", repr(getattr(exc, "__cause__", None))[:300], flush=True)
                traceback.print_exc()
    print("总耗时 %.1f 分钟" % ((time.time() - t0) / 60), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
