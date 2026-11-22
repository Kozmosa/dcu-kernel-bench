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

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/pools" "$MANIFEST_DIR"

# 1) 清单（缺了才生成，已有则保留——分片看护会自己维护 run_*.json）
if [ ! -f "$MANIFEST_DIR/l${LEVEL}_${TRACKS%%,*}_all.json" ]; then
  python3 "$(dirname "$0")/make_kernelbench_manifests.py" \
    --kernelbench "$KERNELBENCH_ROOT" --level "$LEVEL" \
    --tracks "$TRACKS" --shards "$SHARDS" --out-dir "$MANIFEST_DIR" || exit 2
else
  echo "清单已存在，复用: $MANIFEST_DIR"
fi

# 2) 看护器（后台）
cd "$CODE_DIR" || exit 2
RUN_ROOT="$RUN_ROOT" MANIFEST_DIR="$MANIFEST_DIR" SHARD_PREFIX="$SHARD_PREFIX" \
KERNELBENCH_ROOT="$KERNELBENCH_ROOT" TRACKS="$TRACKS" SHARDS_PER_TRACK="$SHARDS" \
MAX_ATTEMPTS="$ATTEMPTS" SEARCH_CONFIG="$SEARCH_CONFIG" \
POLL_SECONDS="$POLL_SECONDS" STALL_MIN="$STALL_MIN" \
setsid nohup python3 benchmark/tools/supervise_shards.py \
  >> "$RUN_ROOT/supervisor.log" 2>&1 &
echo "supervisor 已启动 pid=$!  run_root=$RUN_ROOT  分片=$((SHARDS * $(echo "$TRACKS" | tr ',' '\n' | wc -l)))"
sleep 25
echo "--- 状态 ---"
cat "$RUN_ROOT/supervisor_status.json" 2>/dev/null | head -30
echo "--- 进程 ---"
pgrep -af "run-kernelbench-batch" | sed 's/.*--manifest //;s/ .*//' | sort | tr '\n' ' '; echo
