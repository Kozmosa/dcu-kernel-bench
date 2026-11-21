#!/usr/bin/env bash
# 框架臂启动器：全新目录（runs2）+ 全新记忆池 + supervisor 自动重启
set -u
source /opt/dtk/env.sh
source /root/llm.env
export RUN_ROOT=${RUN_ROOT:-/root/runs2}
export SHARDS_PER_TRACK=${SHARDS_PER_TRACK:-6}
export MAX_ATTEMPTS=${MAX_ATTEMPTS:-10}
export SEARCH_CONFIG=${SEARCH_CONFIG:-dcu_d_combined}
export STALL_MIN=${STALL_MIN:-50}
# 1116 在线契约开关：原生 dtype + 任务容差 + 执行前静态合规 + oracle root 不参与选优。
# 不设这个变量时框架保持上游 KernelBench 语义（fp32 输入 + allclose 1e-4）。
export DCUKB_ROOT=${DCUKB_ROOT:-/root/dcukb}
mkdir -p "$RUN_ROOT"
cd /root
nohup python3 /root/supervise_fw.py >> "$RUN_ROOT/supervisor.log" 2>&1 &
echo "supervisor 已启动 pid=$!  root=$RUN_ROOT  分片=$((SHARDS_PER_TRACK*2))"
sleep 20
echo "--- 状态"
cat "$RUN_ROOT/supervisor_status.json" 2>/dev/null | head -20
