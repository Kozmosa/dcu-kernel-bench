# 上游 KernelBench L1 在 DCU 上的试跑（2026-11-23，**仅有效样本 17 题**）

> ⚠️ **这不是一次完整评测**。本轮原计划 100 题 × 2 轨 = 200 条，实际只有 **17 条**跑到了模型，
> 其余 183 条的 10 次尝试全部被网关 `HTTP 403 insufficient_user_quota` 挡掉（账户余额用尽：
> 剩余 $0.0435，单次预扣需 $0.0461）。本报告只统计 `provider_errors == 0` 的题。
>
> 口径：**上游 KernelBench 语义**（输入 cast fp32、`allclose(atol=rtol=1e-4)`、单一 `get_inputs()` shape、
> 无执行前静态审计、root 可当冠军）；分母为同轮现测 `torch_eager`。**无离线终审**（上游题没有 task.yaml / 隐藏 case）。

---

## 1. 结论

| 项 | 结果 |
|---|---|
| 有效样本 | **17 题**（Triton 16 / HIP 1） |
| 解出（候选 correct） | **13 / 17（76%）** |
| 跑赢 eager（Fast1） | **3 / 13** |
| 加速比 中位 / 几何 / 最好 | **0.81× / 0.68× / 2.77×** |

**核心发现：在这套 L1 题目上，框架生成的 kernel 大多比 PyTorch eager 慢。**
L1 的算子都很小（参考实现耗时 3.2–117 ms 量级，多数 5–18 ms），瓶颈是**发射开销与访存**而不是算力：
eager 已经接近最优，而 Triton kernel 每层都要额外 launch + 读写中间张量，于是普遍落后。
只有 3 题靠更好的分块/向量化跑赢（1.50×、2.49×、2.77×），另有 1 题严重退步（`29_Softplus` 0.03×，慢 33 倍）。

这与本仓库 66 题（大 attention / GEMM，geomean 4–7×）形成鲜明对比 —— **L1 适合测"能否写对"，不适合测"能否写快"**。

## 2. 逐题明细（有效样本）

| 轨 | 题 | correct | 加速比 | champion | 首次 correct 的尝试 | 参考实现 | 失败类型（尝试内） |
|---|---|---|---|---|---|---|---|
| triton | 1_Square_matrix_multiplication | ✓ | 0.81× | n2 | root(0) | 8.34 ms | — |
| triton | 2_Standard_matrix_multiplication | ✓ | 0.86× | n2 | root(0) | 9.23 ms | — |
| triton | 3_Batched_matrix_multiplication | ✓ | **1.50×** | n4 | root(0) | 27.5 ms | — |
| triton | 4_Matrix_vector_multiplication | ✓ | **2.77×** | n10 | root(0) | 117 ms | — |
| triton | 6_Matmul_with_large_K_dimension | ✓ | 0.74× | n2 | root(0) | 13.2 ms | — |
| triton | 8_Matmul_with_irregular_shapes | ✓ | 0.22× | n4 | root(0) | 17.9 ms | — |
| triton | 9_Tall_skinny_matrix_multiplication | ✓ | 0.81× | n10 | root(0) | 12.2 ms | — |
| triton | 10_3D_tensor_matrix_multiplication | ✓ | 0.77× | n7 | root(0) | 3.2 ms | — |
| triton | 15_Matmul_for_lower_triangular_matrices | ✓ | **2.49×** | n7 | root(0) | 8.89 ms | — |
| triton | 18_Matmul_with_transposed_both | ✓ | 0.59× | n10 | root(0) | 8.67 ms | — |
| triton | 20_LeakyReLU | ✓ | —（无 champion 速度） | n5 | 第 **5** 次 | — | — |
| triton | 29_Softplus | ✓ | 0.03× | n10 | 第 **10** 次 | 57 ms | — |
| triton | 25_Swish | ✗ | — | 无 | — | — | OOM 7 / Triton 编译错 2 / 不支持的构造 1 |
| triton | 30_Softsign | ✗ | — | 无 | — | — | OOM 9 / Triton 编译错 1 |
| triton | 23_Softmax | ✗ | — | 无 | — | — | **OOM 11** |
| triton | 24_LogSoftmax | ✗ | — | 无 | — | — | OOM 7 / Triton 编译错 4 |
| cuda | 2_Standard_matrix_multiplication | ✓ | 0.33× | n1 | root(0) | 10.6 ms | — |

> `root(0)` 表示 root（框架 scaffold 直通参考实现）本来就是 correct，最终 champion 仍是生成候选；
> 没有 champion 速度表示该题 best 节点没有有效计时。

## 3. 聚合（按轨）

| 指标 | Triton（16 题） | HIP（1 题） |
|---|---|---|
| Correct | **12/16（75%）** | 1/1 |
| Fast1（correct 且 >1×） | 3/12 | 0 |
| 加速比 中位 / 几何 / 最好 | 0.81× / 0.68× / 2.77× | 0.33× |

## 4. 失败原因：4 题全是评测侧抢显存，不是候选写错

4 个失败全部集中在**小算子**（Swish / Softsign / Softmax / LogSoftmax），`failure_counts` 以
`torch.OutOfMemoryError` 为主（7/9/11/7 次）。当时 12 个分片同时评测、单卡 64 GiB，
几个分片的候选 + eager 参考同时驻留就把显存打满 —— **这是并发评测造成的 harness 伪失败，不是候选缺陷**。
另有零星 `triton.compiler.errors.CompilationError`（候选真编译不过）。

**修法**（下次跑前）：把评测段串行化（`STARK_GPU_EVALUATION_LOCK`）或把分片降到 6 片以内；
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 已开，可缓解碎片但不能根治并发峰值。

## 5. 这轮暴露的运维问题（都已修，写在这里避免重犯）

1. **余额用尽的表现极像"网关卡流"**：网关接受请求回 `200`，然后只发 `: PING` 心跳不出字。
   必须读 `403 insufficient_user_quota` 的错误体才能定位。→ 已在 provider 里加 **preflight 重试**，
   并把 `provider_error_details` 落进 `run.json`。
2. **内容空闲超时不能设太短**：`gpt-6-astra` 会先静默推理很久才吐字，90s 上限会把正常调用误杀成
   `provider_error`（实测 57 题全 10/10 失败）。现在默认 **600s**，且把 `reasoning_content`
   等推理增量也算作"活跃"。
3. **分片数要按 cgroup 内存反推**：容器上限 **115 GiB**，每分片峰值约 4 GB（父进程 + 隔离评测子进程）；
   28 片直接 OOM 到 sshd 被杀、机器要重启。现在看护器带 **内存闸门 + 错峰启动**。
4. **代理要显式统一大小写**：平台注入的小写 `https_proxy` 会被 SDK 优先采用，与我们的大写
   `HTTPS_PROXY` 打架 → 表现为进程 0 CPU、卡在 `poll_schedule_timeout`。现在由 `KB_PROXY` 统一，
   并在起跑前做 5 次重试式出口预检。

## 6. 复现（充值后接着跑，已完成的题会自动跳过）

```bash
cd /root/dcukb
TRACKS=triton,cuda SHARDS=6 ATTEMPTS=10 KB_PROXY=http://127.0.0.1:17890 \
  bash benchmark/tools/launch_kernelbench.sh
```

清单：`/root/manifests/kb_l1/kn_l1_{triton,cuda}_{all,s0..s5}.json`（100 题/轨，6 片）；
产物：`/root/runs/kb_l1/kn_l1_*_s*/<alias>_l1_p<pid>/{run.json,best_code.py}`；
被余额挡掉的 183 题目录已移到 `/root/runs/kb_l1/_rerun_needed/`，续跑时会重新排队。

## 7. 一句话总结

**链路是通的**（17 题里 13 题写对，含 3 题比 eager 快），**但 L1 这种小算子上框架打不过 eager**；
要得到有统计意义的结论，需要先把额度补齐、把评测并发压下来，再跑完整 200 条。
