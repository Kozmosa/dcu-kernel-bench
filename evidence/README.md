# evidence/ — 指标证据与复现资产

本目录只放**可复算指标所需的原始数据**与**部署脚本**，不放构建产物、不放大文件。
所有 `.tar.gz` 都是打平的静态快照，解压即可用。

## 1. `metrics_runs.tar.gz`（4.3 MB）

各轮全量跑的**逐题 `run.json`**（含 `best_code.py`）与离线终审缓存：

| 目录 | 内容 |
|---|---|
| `en66/` | 单轨全量（英文题面 + 11 道去做法），66 题 —— `benchmark/tools/metrics.md` 的数据源 |
| `dual_hip/` | 双轨 HIP 轨，66 题 —— `metrics_dual.md` 的 HIP 列 |
| `dual_triton/` | 双轨 Triton 轨，66 题 —— `metrics_dual.md` 的 Triton 列 |
| `offline_audit_clean/` | 单轨 66 题的离线终审缓存 + `offline_audit.json` 汇总 |

解压与复算：

```bash
tar xzf evidence/metrics_runs.tar.gz -C /tmp/runs
python3 benchmark/tools/collect_metrics.py --run-root /tmp/runs/dual_hip \
        --out /tmp/dual_hip.md --csv /tmp/dual_hip.csv
python3 benchmark/tools/collect_metrics.py --run-root /tmp/runs/en66 \
        --offline-audit /tmp/runs/offline_audit_clean/offline_audit.json \
        --audit-cache /tmp/runs/offline_audit_clean/offline_audit \
        --out /tmp/en66_gated.md
python3 benchmark/tools/compare_tracks.py --hip /tmp/runs/dual_hip \
        --triton /tmp/runs/dual_triton --out /tmp/compare.md
```

## 2. `deploy_scripts.tar.gz`（60 KB，56 个脚本）

DCU 机器上的部署与运行脚本（已剔除一次性诊断脚本与 189 MB 二进制资产）：

| 类别 | 文件 |
|---|---|
| 部署 | `deploy.sh`、`verify.sh`、`_setup_llm.sh`、`local_direct_proxy.py`、`tunnel_daemon.ps1`、`DEPLOY.md` |
| 生成 | `run_generation.sh`、`run_parallel.sh`、`run_dualtrack.sh`、`run_lean.sh`、`supervise_r3.py`、`start8.sh`、`start9.sh`、`clean_rerun.sh`、`fresh_start.sh` |
| 离线终审 | `audit_pure*.sh`、`audit_subset.sh`、`finalize_run.sh`、`finalize_final2.sh` |
| 指标 | `collect_baselines.sh`、`merge_metrics.sh`、`metrics_three_groups.sh`、`strict_speedup.sh`、`cmp_two_groups.sh`、`breakdown_failures.sh`、`mismatch_tasks.sh`、`fill_metrics.sh` |
| 纯 API 基线 | `pure_bench.py`（支持 `--wiki`、`--missing`）、`pure_all.sh`、`pure_repair.sh`、`run_pure_wiki.sh`、`run_subset_triton.sh` |
| 框架补丁 | `patch_streaming_any.py`（流式传输）、`patch_retry_overloaded.py`（过载可重试）、`patch_memory_pool.py`（共享记忆池）、`migrate_pool.py`、`setup_r3.py`、`test_pool_locking.py` |
| 运维 | `monitor_once.sh`、`parallel_status.sh`、`pack_run.sh`、`make_flat.sh`、`cleanup_disk.sh`、`RUN_R3.md` |

## 3. 口径提醒

- `run.json` 的 `nodes` 里 `root` 是脚手架直通节点，**指标计算一律排除**；
- 加速比分母在线为 `torch_eager`，离线为 `aiter`（部分题无基线，见 `metrics.md` 的说明）；
- **真解 = 在线正确 ∩ 离线通过**，两套判据的容差不同（在线固定 `1e-4`，离线按 `task.yaml` 分档），
  因此存在"离线过但在线挂"的题，属于口径差异而非实现错误。
