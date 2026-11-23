#!/usr/bin/env python3
"""supervise_shards.py —— 框架臂分片看护：自动重启 + 断点续跑 + 只跑没跑完的题。

设计要点（对应 2026-11-21/22 那一轮踩过的坑）：

  · **完成判据是「已完成题数 == 清单题数」**，不是 `summary.json` 是否存在。
    框架的 batch 在 provider 预检失败时会 abort 整批（`batch_aborted`）并照常写
    summary.json——用 summary.json 判完成，会把"abort 掉一半的分片"当成跑完了，
    实测因此白等 8 小时。
  · **进程存活用 pid 文件 + `/proc/<pid>/cmdline` 双重校验**。只看 `os.kill(pid,0)`
    会被 pid 复用骗到（旧 pid 被别的活进程占用 → 以为分片还在跑，于是永不重启）。
  · **每轮把剩余题重写成 `run_<name>.json` 再起**，已完成的不重跑；分片清单本体的写法：
    `<MANIFEST_DIR>/<prefix>_<track>_s<k>.json`。
  · 单 GPU：评测端有 `gpu_evaluation_lock`（设置 `STARK_GPU_EVALUATION_LOCK` 才生效）。
  · 记忆池在 `RUN_ROOT/pools/memory_pool.json`（全新目录 = 全新记忆）。
  · `DCUKB_ROOT` 只作为**透传**：设了就把 1116 在线契约带进子进程，没设就走上游
    KernelBench 语义（fp32 + allclose 1e-4）。上游 KernelBench L1 这种没有 task.yaml
    的题目必须不设。

用法：

    # 1) 跑 DCUKB 自己的题（1116 契约）
    RUN_ROOT=/root/runs2 MANIFEST_DIR=/root/manifests python3 benchmark/tools/supervise_shards.py

    # 2) 跑上游 KernelBench L1（上游语义，不要设 DCUKB_ROOT）
    RUN_ROOT=/root/runs/kb_l1 MANIFEST_DIR=/root/manifests/kb_l1 SHARD_PREFIX=kn_l1 \
      KERNELBENCH_ROOT=/root/KernelBench-main python3 benchmark/tools/supervise_shards.py
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
KB_ROOT = pathlib.Path(os.environ.get("KERNELBENCH_ROOT", str(DCUKB / "benchmark" / "kernelbench_compat")))
PREFIX = os.environ.get("SHARD_PREFIX", "fw")
TRACKS = [t.strip() for t in os.environ.get("TRACKS", "triton,cuda").split(",") if t.strip()]
SHARDS = int(os.environ.get("SHARDS_PER_TRACK", "6"))
SEARCH = os.environ.get("SEARCH_CONFIG", "dcu_d_combined")
ATTEMPTS = os.environ.get("MAX_ATTEMPTS", "10")
POLL = float(os.environ.get("POLL_SECONDS", "90"))
STALL_MIN = float(os.environ.get("STALL_MIN", "50"))
MAX_RESTARTS = int(os.environ.get("MAX_RESTARTS", "8"))
# 内存闸门：容器 cgroup v1 上限 115 GiB，每个分片峰值约 4 GiB（父进程 + 隔离评测子进程）。
# 2026-11-22 用 28 分片直接把 cgroup 打爆（oom-kill 掉 python3 与 sshd，机器要重启）。
MEM_GUARD_GIB = float(os.environ.get("MEM_GUARD_GIB", "95"))
LAUNCH_STAGGER = float(os.environ.get("LAUNCH_STAGGER", "6"))
CGROUP_CURRENT = pathlib.Path("/sys/fs/cgroup/memory.current")
CGROUP_V1_CURRENT = pathlib.Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
CGROUP_V1_LIMIT = pathlib.Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")

LOG_DIR = ROOT / "logs"
POOL = ROOT / "pools" / "memory_pool.json"
STATUS = ROOT / "supervisor_status.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def shard_specs():
    for track in TRACKS:
        for index in range(SHARDS):
            name = f"{PREFIX}_{track}_s{index}"
            yield name, track, MANIFESTS / f"{name}.json", ROOT / name


def memory_gib() -> float:
    for path in (CGROUP_CURRENT, CGROUP_V1_CURRENT):
        try:
            return int(path.read_text().strip()) / 2 ** 30
        except (OSError, ValueError):
            continue
    return 0.0


def memory_limit_gib() -> float:
    try:
        value = int(CGROUP_V1_LIMIT.read_text().strip())
    except (OSError, ValueError):
        return 0.0
    return value / 2 ** 30 if value < 2 ** 60 else 0.0


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def shard_pid(out_dir: pathlib.Path):
    path = out_dir / ".supervisor.pid"
    if not path.exists():
        return None
    try:
        pid = int(path.read_text().strip())
    except ValueError:
        return None
    if not pid_alive(pid):
        return None
    try:
        cmdline = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "replace")
    except OSError:
        return None
    return pid if "run-kernelbench-batch" in cmdline else None


def done_aliases(out_dir: pathlib.Path, tasks: list[dict]) -> set:
    done = set()
    if not out_dir.is_dir():
        return done
    for task in tasks:
        rel = "%s_l%s_p%s" % (task["alias"], task["level"], task["problem_id"])
        if (out_dir / rel / "run.json").exists():
            done.add(task["alias"])
    return done


def launch(name, track, manifest: pathlib.Path, out_dir: pathlib.Path):
    env = dict(os.environ)
    env["KW_MEMORY_POOL"] = str(POOL)
    log = LOG_DIR / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(log, "a", buffering=1) as handle:
        handle.write("\n===== [%s] launch %s (%s) attempts=%s manifest=%s contract=%s\n"
                     % (now(), name, track, ATTEMPTS, manifest.name,
                        "dcukb" if os.environ.get("DCUKB_ROOT") else "upstream"))
        handle.flush()
        proc = subprocess.Popen(
            [sys.executable, "-m", "src.cli", "run-kernelbench-batch",
             "--kernelbench-root", str(KB_ROOT),
             "--manifest", str(manifest),
             "--backend", track,
             "--search-config", SEARCH,
             "--provider", "openai-compatible",
             "--env-file", str(CODE / ".env"),
             "--max-attempts", ATTEMPTS,
             "--output-dir", str(out_dir)],
            cwd=str(CODE), env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    (out_dir / ".supervisor.pid").write_text(str(proc.pid), encoding="utf-8")
    return proc


def main() -> int:
    for path in (ROOT, LOG_DIR, POOL.parent):
        path.mkdir(parents=True, exist_ok=True)
    if not POOL.exists():
        POOL.write_text("{}\n", encoding="utf-8")

    state = {}
    for name, track, manifest, out_dir in shard_specs():
        if not manifest.exists():
            print("[fatal] 缺 manifest: %s" % manifest, file=sys.stderr)
            return 2
        state[name] = {"track": track, "manifest": manifest, "out": out_dir,
                       "restarts": 0, "proc": None}

    def write_status():
        STATUS.write_text(json.dumps({
            "updated": now(),
            "run_root": str(ROOT),
            "manifest_dir": str(MANIFESTS),
            "kernelbench_root": str(KB_ROOT),
            "contract": "dcukb" if os.environ.get("DCUKB_ROOT") else "upstream",
            "memory_gib": round(memory_gib(), 1),
            "memory_limit_gib": round(memory_limit_gib(), 1),
            "memory_guard_gib": MEM_GUARD_GIB,
            "shards": {name: {"running": bool(shard_pid(item["out"])),
                              "restarts": item["restarts"],
                              "done": item.get("done"), "total": item.get("total")}
                       for name, item in state.items()},
        }, indent=1), encoding="utf-8")

    def stop(signum, _frame):
        print("[%s] 收到信号 %s，停止所有分片" % (now(), signum), flush=True)
        for item in state.values():
            proc = item["proc"]
            if proc and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:  # noqa: BLE001
                    pass
        write_status()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    print("[%s] supervisor up: %d 分片 run_root=%s manifest_dir=%s kernelbench=%s contract=%s"
          % (now(), len(state), ROOT, MANIFESTS, KB_ROOT,
             "dcukb" if os.environ.get("DCUKB_ROOT") else "upstream"), flush=True)

    while True:
        for name, item in state.items():
            manifest = item["manifest"]
            out_dir = item["out"]
            tasks = json.loads(manifest.read_text(encoding="utf-8"))["tasks"]
            done = done_aliases(out_dir, tasks)
            remaining = [t for t in tasks if t["alias"] not in done]
            item["done"] = len(done)
            item["total"] = len(tasks)
            if not remaining:
                if item["proc"] is not None and item["proc"].poll() is not None:
                    item["proc"] = None
                continue

            pid = shard_pid(out_dir)
            if pid:
                # 卡死判定：日志长时间不更新且进程还在
                log = LOG_DIR / f"{name}.log"
                if log.exists() and (time.time() - log.stat().st_mtime) > STALL_MIN * 60:
                    if item["restarts"] < MAX_RESTARTS:
                        print("[%s] %s 卡死（日志 %d 分钟未更新），重启 #%d"
                              % (now(), name, STALL_MIN, item["restarts"] + 1), flush=True)
                        try:
                            os.killpg(os.getpgid(pid), signal.SIGTERM)
                        except Exception:  # noqa: BLE001
                            pass
                        item["restarts"] += 1
                continue

            if item["restarts"] > MAX_RESTARTS:
                continue
            used = memory_gib()
            if used and used > MEM_GUARD_GIB:
                print("[%s] 内存 %.1f GiB 已超过闸门 %.1f GiB，暂缓启动 %s"
                      % (now(), used, MEM_GUARD_GIB, name), flush=True)
                continue
            resume = MANIFESTS / ("run_%s.json" % name)
            resume.write_text(json.dumps({
                "name": "run_%s" % name, "backend": item["track"], "tasks": remaining,
            }, indent=1, ensure_ascii=False), encoding="utf-8")
            item["proc"] = launch(name, item["track"], resume, out_dir)
            print("[%s] %s 启动：剩 %d/%d 题 pid=%s 内存=%.1f/~%.1f GiB"
                  % (now(), name, len(remaining), len(tasks), item["proc"].pid,
                     memory_gib(), memory_limit_gib()), flush=True)
            if LAUNCH_STAGGER:
                time.sleep(LAUNCH_STAGGER)
        write_status()
        time.sleep(POLL)


if __name__ == "__main__":
    sys.exit(main())
