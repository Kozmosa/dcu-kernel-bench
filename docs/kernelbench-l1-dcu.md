# 上游 KernelBench L1 在 DCU 上跑（A 方案：上游语义）

> 目的：把**上游 KernelBench 官方 L1（100 题）**接到 KernelWeaver 上在 gfx928 上跑，
> 用于和 KernelBench 官方榜单口径对齐。
> **本页是 A 方案**：不启用 DCUKB 的 1116 在线契约，回到**上游 KernelBench 语义**。
> 若要和本仓库 66 题的表并列（含静态审计与真解门槛），走 B 方案（onboard 成 dcukb 任务，另文）。

---

## 1. 与 DCUKB 66 题的口径差异（不要混表）

| 维度 | 本页（A，上游语义） | 本仓库 66 题（1116 契约） |
|---|---|---|
| 输入 dtype | 统一 cast **fp32** | 原生 dtype（含 int32 索引） |
| 正确性 | `allclose(atol=rtol=1e-4)`，单一 `get_inputs()` shape | `task.yaml` 分档容差 + 分段容差 |
| 执行前静态审计 | **无** | 有（禁用算子 + 平台符号） |
| oracle root | 可以当冠军（`collect_metrics` 默认排除 root 后再统计） | 框架层排除，不导出 |
| 离线终审 / 真解门槛 | **没有**（上游题没有 task.yaml / 隐藏 case） | 有 |
| 题面硬件提示 | 上游 docstring（不含 gfx928 提示） | 题面写明 DCU 目标 + 注入公开契约 |
| DCU Wiki | **仍然生效**（按 `reference_code` 的 API 符号检索，与 `DCUKB_ROOT` 无关） | 生效 |

> 因此：`--max-attempts 1` 时最接近 KernelBench 官方 pass@1；`10` 时是"给框架 10 次机会"的口径，
> 两者不可互换引用。

## 2. 前置条件（机器上）

| 项 | 要求 |
|---|---|
| KernelBench 源码 | `/root/KernelBench-main`（含 `KernelBench/level1/` 100 个 `.py`） |
| kernelbench 包可导入 | `site-packages/kernelbench_src.pth` → `/root/KernelBench-main/src`（否则 `paper_evaluator_import_error`） |
| 框架 | `/root/KernelWeaver`（`cd5d85c` + 两个补丁序列，见 `docs/machine-2026-11-21-k500sm-gfx928.md`） |
| LLM 出口 | `/root/llm.env`（含 `OPENAI_*` 与反代 `HTTPS_PROXY`），反向隧道 `-R 17890` |
| HIP 编译 | `PYTORCH_ROCM_ARCH=gfx928`（否则每个候选按 5 个架构编译，慢 5×） |
| 评测超时 | `STARK_PAPER_EVAL_TIMEOUT_SECONDS=900`（240s 对 hipcc 太紧） |

## 3. 一条命令开跑

```bash
cd /root/dcukb
bash benchmark/tools/launch_kernelbench.sh
```

默认：**L1 + Triton 单轨 + 8 分片 + 10 attempts**，输出 `/root/runs/kb_l1/`，清单 `/root/manifests/kb_l1/`。

常见变体：

```bash
TRACKS=triton,cuda bash benchmark/tools/launch_kernelbench.sh   # 双轨（HIP 编译慢，建议先 Triton）
ATTEMPTS=1             bash benchmark/tools/launch_kernelbench.sh   # 对齐官方 pass@1
LEVEL=2                bash benchmark/tools/launch_kernelbench.sh   # 换 level
SHARDS=10              bash benchmark/tools/launch_kernelbench.sh   # 加并发
```

脚本做四件事：① `unset DCUKB_ROOT`（**A 方案的关键**，设了会 `dcukb_contract_error`）；
② 生成清单（缺了才生成）；③ 后台起 `benchmark/tools/supervise_shards.py`；
④ 打印 `supervisor_status.json` 与在跑进程。

## 4. 清单与产物

```
/root/manifests/kb_l1/
├── l1_triton_all.json / l1_triton_s0..s7.json      # 100 题，按 problem_id 轮转切 8 片
└── run_kn_l1_triton_s0.json ...                    # 看护器每轮写的"剩余题"清单

/root/runs/kb_l1/
├── kn_l1_triton_s0/<alias>_l1_p<pid>/{run.json,best_code.py}
├── logs/kn_l1_triton_s0.log
├── pools/memory_pool.json                          # 全新目录 = 全新记忆
└── supervisor_status.json                          # 每 90s 刷新：running/done/total/restarts
```

## 5. 看指标

```bash
# 在线指标（默认排除 root 候选；与 66 题同一套定义）
python3 benchmark/tools/collect_metrics.py --run-root /root/runs/kb_l1_flat

# 零候选 / 有候选但不正确 / provider 错误分开列名
python3 benchmark/tools/audit_zero_candidates.py --root /root/runs/kb_l1
```

注意：`collect_metrics.py` 需要"一题目录"这一层，而分片产物是"分片→题目"两层，
所以统计前先拍平（硬链接，不动原数据）：

```bash
python3 benchmark/tools/flatten_run_root.py \
  --run-root /root/runs/kb_l1 --out /root/runs/kb_l1_flat --shards "kn_l1_triton_s*"
# 拍平 100 题 -> /root/runs/kb_l1_flat
```

## 6. 看护器的行为（为什么能直接放着跑）

`benchmark/tools/supervise_shards.py`（本轮重写）：

- **完成判据 = 已完成 `run.json` 数 == 清单题数**。框架在 provider 预检失败时会 abort 整批并**照常写
  `summary.json`**，所以不能用 `summary.json` 判完成（上一轮因此白等 8 小时）。
- **进程存活 = pid 文件 + `/proc/<pid>/cmdline` 双重校验**：只看 `os.kill(pid,0)` 会被 pid 复用骗到。
- 每轮把**剩余题**写进 `run_<name>.json` 再起，已完成的题不重跑。
- 日志超过 `STALL_MIN`（默认 50 分钟）没更新且进程还在 → 判卡死，kill 后重拉（`MAX_RESTARTS` 次上限）。

本机已做的离线自检（`WIRING PASS`）：① loader 的 `glob "<pid>_*.py"` 在 100 题上不串题（1 / 10 / 11 / 100 互不冲突）；
② `summary.json` 不被当完成；③ 死 pid 与 pid 复用都判"没在跑"。

## 7. 已知限制（引用结果时必须说明）

1. **没有离线终审**：上游题没有 `task.yaml` 与隐藏 case，所以拿不到"真解（在线 ∩ 离线）"，
   也抓不到"在线侥幸通过、离线编不过"的情况（本仓库 66 题里后者有 7–15 题）。
2. **没有执行前静态审计**：候选可以内部调 `torch.matmul`/`F.conv2d` 而不被判负——这属于上游语义的原貌。
3. **root 可以当冠军**：`collect_metrics.py` 默认把 root 排除在候选池外，但 `run.json` 里
   `best_node_id="root"` 的题要在报告里单独标注（这些题等于"框架没产出超过脚手架的候选"）。
4. **题面没有 gfx928 硬件提示**：L1 docstring 是上游原文；DCU Wiki 仍在上下文里。
5. 纯 API 臂（`pure_bench.py`）在 A 方案下**不能直接跑**——它硬性要求 `DCUKB_ROOT`，
   设了就会走 1116 桥并要求 `task.yaml`。要出纯 API 基线就得上 B 方案，或放宽该检查。

## 8. 预计耗时（按本轮实测速率）

- 准备：0（脚本已就绪，清单自动生成）
- Triton 单轨、8 分片、10 attempts：单体调用 ~2–4 分钟/次、每题最多 10 次 → **约 4–7 小时**
- 若 `ATTEMPTS=1`：**约 40–60 分钟**
- 双轨会把 HIP 编译开销叠上来（本轮 HIP 轨修复后约 21 题/小时），建议先只跑 Triton。
