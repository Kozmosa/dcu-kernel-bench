#!/usr/bin/env bash
# 上游 KernelBench（L1）在 DCU 上跑框架臂的一键启动器 —— A 方案：**上游 KernelBench 语义**。
#
# 口径（和 DCUKB 的 66 题不同，不要混表）：
#   · DCUKB_ROOT **不设** → 框架不走 1116 桥：输入统一 cast fp32、allclose(atol=rtol=1e-4)、
#     只跑 get_inputs() 的单一 shape、没有执行前静态审计、root 可以当冠军。
#   · 上游 L1 没有 task.yaml，设了 DCUKB_ROOT 会直接 dcukb_contract_error，所以这里显式 unset。
#
# 用法：
#   bash benchmark/tools/launch_kernelbench.sh                 # 默认 L1 + Triton 单轨 + 8 分片
#   TRACKS=triton,cuda bash benchmark/tools/launch_kernelbench.sh   # 双轨
#   LEVEL=2 SHARDS=6 bash benchmark/tools/launch_kernelbench.sh     # 换 level
#   ATTEMPTS=1 bash benchmark/tools/launch_kernelbench.sh           # 对齐 KernelBench 官方 pass@1
set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# ---- 可按需覆盖 ----
LEVEL=${LEVEL:-1}
TRACKS=${TRACKS:-triton}
SHARDS=${SHARDS:-8}
ATTEMPTS=${ATTEMPTS:-10}
SEARCH_CONFIG=${SEARCH_CONFIG:-dcu_d_combined}
KERNELBENCH_ROOT=${KERNELBENCH_ROOT:-/root/KernelBench-main}
CODE_DIR=${CODE_DIR:-/root/KernelWeaver}
RUN_ROOT=${RUN_ROOT:-/root/runs/kb_l${LEVEL}}
MANIFEST_DIR=${MANIFEST_DIR:-/root/manifests/kb_l${LEVEL}}
SHARD_PREFIX=${SHARD_PREFIX:-kn_l${LEVEL}}
POLL_SECONDS=${POLL_SECONDS:-90}
STALL_MIN=${STALL_MIN:-50}

set -a
source /opt/dtk/env.sh
[ -f /root/llm.env ] && . /root/llm.env
set +a
# A 方案的关键：不要带上 1116 在线契约
unset DCUKB_ROOT
# HIP 评测超时必须够（240s 对 hipcc 编译太紧）
export STARK_PAPER_EVAL_TIMEOUT_SECONDS=${STARK_PAPER_EVAL_TIMEOUT_SECONDS:-900}

# 出口代理：容器里可能同时存在两套——
#   · 平台注入的小写 https_proxy=http://preset:...@10.15.100.43:3120（实测可用）
#   · 我们自己开的小写/大写隧道 http://127.0.0.1:17890（会随 ssh 会话掉）
# httpx/openai SDK 大小写都会读，谁生效取决于实现细节 → 这里**只留一个**并统一大小写，
# 由 KB_PROXY 显式指定；没有 KB_PROXY 时优先用平台 preset，最后才回落到隧道。
export KB_PROXY=${KB_PROXY:-${https_proxy:-${HTTPS_PROXY:-}}}
if [ -n "$KB_PROXY" ]; then
  export HTTPS_PROXY="$KB_PROXY" HTTP_PROXY="$KB_PROXY" https_proxy="$KB_PROXY" http_proxy="$KB_PROXY"
fi
export NO_PROXY=${NO_PROXY:-localhost,127.0.0.1,::1}
export no_proxy="$NO_PROXY"
# litellm 启动时会去 raw.githubusercontent.com 拉价目表；无外网环境下每次启动都等超时。
export LITELLM_LOCAL_MODEL_COST_MAP=${LITELLM_LOCAL_MODEL_COST_MAP:-True}

# 起跑前必须确认出口可用，否则一开跑就是整片 APIConnectionError。
# 出口本身也会抖（实测代理偶发 000/5s），单次探测会误杀整轮 → 重试式预检。
PROBE_URL="${OPENAI_BASE_URL:-}/models"
if [ -n "${OPENAI_BASE_URL:-}" ]; then
  code=000
  for attempt in 1 2 3 4 5; do
    code=$(curl -s -o /dev/null -m 15 -w "%{http_code}" "$PROBE_URL"       -H "Authorization: Bearer ${OPENAI_API_KEY:-}") || code=000
    [ "$code" = "200" ] && break
    echo "预检第 $attempt 次失败（api=$code），3s 后重试"
    sleep 3
  done
  echo "预检出口: proxy=${KB_PROXY:-direct} api=$code"
  if [ "$code" != "200" ]; then
    echo "[FAIL] LLM 出口不可用（连续 5 次非 200），先修代理再跑：KB_PROXY=http://host:port bash $0" >&2
    exit 3
  fi
fi

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/pools" "$MANIFEST_DIR"

# 1) 清单（缺了才生成，已有则保留——分片看护会自己维护 run_*.json）
FIRST_TRACK=${TRACKS%%,*}
if [ ! -f "$MANIFEST_DIR/${SHARD_PREFIX}_${FIRST_TRACK}_all.json" ]; then
  python3 "$SCRIPT_DIR/make_kernelbench_manifests.py" \
    --kernelbench "$KERNELBENCH_ROOT" --level "$LEVEL" \
    --tracks "$TRACKS" --shards "$SHARDS" --out-dir "$MANIFEST_DIR" \
    --name-prefix "$SHARD_PREFIX" || exit 2
else
  echo "清单已存在，复用: $MANIFEST_DIR/${SHARD_PREFIX}_${FIRST_TRACK}_*.json"
fi

# 2) 看护器（后台）
cd "$CODE_DIR" || exit 2
RUN_ROOT="$RUN_ROOT" MANIFEST_DIR="$MANIFEST_DIR" SHARD_PREFIX="$SHARD_PREFIX" \
KERNELBENCH_ROOT="$KERNELBENCH_ROOT" TRACKS="$TRACKS" SHARDS_PER_TRACK="$SHARDS" \
MAX_ATTEMPTS="$ATTEMPTS" SEARCH_CONFIG="$SEARCH_CONFIG" \
POLL_SECONDS="$POLL_SECONDS" STALL_MIN="$STALL_MIN" \
setsid nohup python3 "$SCRIPT_DIR/supervise_shards.py" \
  >> "$RUN_ROOT/supervisor.log" 2>&1 &
echo "supervisor 已启动 pid=$!  run_root=$RUN_ROOT  分片=$((SHARDS * $(echo "$TRACKS" | tr ',' '\n' | wc -l)))"
sleep 25
echo "--- 状态 ---"
cat "$RUN_ROOT/supervisor_status.json" 2>/dev/null | head -30
echo "--- 进程 ---"
pgrep -af "run-kernelbench-batch" | sed 's/.*--manifest //;s/ .*//' | sort | tr '\n' ' '; echo
