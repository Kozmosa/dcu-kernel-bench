#!/usr/bin/env python3
"""supervise_shards.py —— 框架臂分片监督器（崩了自动重启、卡死自动重拉、可续跑）。

设计要点（对应这一轮踩过的坑）：
  · 单 GPU：评测端已有 gpu_evaluation_lock，多分片会排队而不是互相污染计时；LLM 调用是 IO 型，
    所以并发主要受 prompt 与出口带宽限制。默认 6 分片/轨（共 12 条流）。
  · 每个分片一个 `run-kernelbench-batch` 进程；summary.json 出现即视为该分片完成，不再重启。
  · 日志 mtime 超过 STALL_MIN 分钟没有更新 → 判定卡死，kill 后重拉（最多 MAX_RESTARTS 次）。
  · 记忆池放在 RUN_ROOT/pools/memory_pool.json（全新目录 = 全新记忆，不和旧轮混）。
  · 所有参数用环境变量覆盖，不改代码即可调并发/候选数。

用法：
    python3 /root/supervise_fw.py            # 前台跑（建议 nohup/setsid）
    RUN_ROOT=/root/runs2 SHARDS_PER_TRACK=6 python3 /root/supervise_fw.py
"""
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = pathlib.Path(os.environ.get("RUN_ROOT", "/root/runs2"))
CODE = pathlib.Path(os.environ.get("CODE_DIR", "/root/KernelWeaver"))
DCUKB = pathlib.Path(os.environ.get("DCUKB_DIR", "/root/dcukb"))
MANIFESTS = pathlib.Path(os.environ.get("MANIFEST_DIR", "/root/manifests"))
SEARCH = os.environ.get("SEARCH_CONFIG", "dcu_d_combined")
ATTEMPTS = os.environ.get("MAX_ATTEMPTS", "10")
SHARDS = int(os.environ.get("SHARDS_PER_TRACK", "6"))
STALL_MIN = float(os.environ.get("STALL_MIN", "50"))
MAX_RESTARTS = int(os.environ.get("MAX_RESTARTS", "8"))
POLL = float(os.environ.get("POLL_SECONDS", "60"))

LOG_DIR = ROOT / "logs"
POOL = ROOT / "pools" / "memory_pool.json"
STATUS = ROOT / "supervisor_status.json"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def shard_specs():
    for track in ("triton", "cuda"):
        for i in range(SHARDS):
            name = f"fw_{track}_s{i}"
            yield name, track, MANIFESTS / f"{name}.json", ROOT / name


def launch(name, track, manifest, out_dir, restarts):
    log = LOG_DIR / f"{name}.log"
    cmd = [
        sys.executable, "-m", "src.cli", "run-kernelbench-batch",
        "--kernelbench-root", str(DCUKB / "benchmark" / "kernelbench_compat"),
        "--manifest", str(manifest),
        "--backend", track,
        "--search-config", SEARCH,
        "--provider", "openai-compatible",
        "--env-file", str(CODE / ".env"),
        "--max-attempts", ATTEMPTS,
        "--output-dir", str(out_dir),
    ]
    env = dict(os.environ)
    env["KW_MEMORY_POOL"] = str(POOL)
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL",
                "OPENAI_WIRE_API", "OPENAI_HTTP_STREAM", "OPENAI_MAX_RETRIES", "PYTHONPATH"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    with open(log, "a", buffering=1) as fh:
        fh.write(f"\n===== [{now()}] launch #{restarts} {name} ({track}) attempts={ATTEMPTS} search={SEARCH}\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=str(CODE), env=env, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
    return proc


def main() -> int:
    for path in (ROOT, LOG_DIR, POOL.parent):
        path.mkdir(parents=True, exist_ok=True)
    if not POOL.exists():
        POOL.write_text("{}\n", encoding="utf-8")
    state = {}
    for name, track, manifest, out_dir in shard_specs():
        if not manifest.exists():
            print(f"[fatal] 缺 manifest: {manifest}", file=sys.stderr)
            return 2
        state[name] = {"track": track, "proc": None, "restarts": 0, "done": False,
                       "manifest": str(manifest), "out": str(out_dir)}

    def stop(signum, _frame):
        print(f"[{now()}] 收到信号 {signum}，停止所有分片", flush=True)
        for item in state.values():
            proc = item["proc"]
            if proc and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:  # noqa: BLE001
                    pass
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    for name, item in state.items():
        if (pathlib.Path(item["out"]) / "summary.json").exists():
            item["done"] = True
            print(f"[{now()}] {name} 已有 summary.json，跳过", flush=True)
            continue
        item["proc"] = launch(name, item["track"], pathlib.Path(item["manifest"]), pathlib.Path(item["out"]), 0)
        print(f"[{now()}] 启动 {name} pid={item['proc'].pid}", flush=True)

    while True:
        changed = False
        for name, item in state.items():
            if item["done"]:
                continue
            out_dir = pathlib.Path(item["out"])
            log_path = LOG_DIR / f"{name}.log"
            if (out_dir / "summary.json").exists():
                item["done"] = True
                changed = True
                log_path.open("a").write(f"===== [{now()}] {name} 完成（summary.json 出现）\n")
                continue
            proc = item["proc"]
            alive = proc is not None and proc.poll() is None
            stalled = False
            if alive and log_path.exists():
                age_min = (time.time() - log_path.stat().st_mtime) / 60.0
                stalled = age_min > STALL_MIN
            if alive and not stalled:
                continue
            reason = "卡死" if stalled else f"退出码 {proc.poll() if proc else 'n/a'}"
            if stalled:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    pass
            item["restarts"] += 1
            changed = True
            if item["restarts"] > MAX_RESTARTS:
                item["done"] = True
                log_path.open("a").write(f"===== [{now()}] {name} 重启超限（{reason}），放弃\n")
                continue
            log_path.open("a").write(f"===== [{now()}] {name} {reason} → 重启 #{item['restarts']}\n")
            item["proc"] = launch(name, item["track"], pathlib.Path(item["manifest"]), out_dir, item["restarts"])
        if changed or not STATUS.exists():
            STATUS.write_text(json.dumps({
                "updated": now(),
                "root": str(ROOT),
                "shards_per_track": SHARDS,
                "search": SEARCH,
                "max_attempts": ATTEMPTS,
                "done": sorted(n for n, i in state.items() if i["done"]),
                "running": sorted(n for n, i in state.items()
                                  if i["proc"] and i["proc"].poll() is None and not i["done"]),
                "restarts": {n: i["restarts"] for n, i in state.items() if i["restarts"]},
            }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        if all(i["done"] for i in state.values()):
            print(f"[{now()}] 全部分片结束", flush=True)
            return 0
        time.sleep(POLL)


if __name__ == "__main__":
    raise SystemExit(main())
