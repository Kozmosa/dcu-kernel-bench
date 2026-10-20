# 题目级指标（.dcu_runs\en66）

- accepted_tasks = **66**（solved = 60）
- 候选口径：**排除 root**（root 是框架直通脚手架，不算 Agent 产出）
- 参考 baseline（speedup 分母）：`torch_eager`
- 数据来源：英文题面全量跑 runs/dcukb_en_1019_1027（66 道，16 分片）+ 3 道补跑 + 6001 HIP 重跑 + 11 道题面清理后重跑

## 简要描述
- **能编译**：66/66 道题至少产出一个可编译候选，脚手架不是瓶颈。
- **能写对**：Correct Rate 60/66（90.9%）；6 道始终没写出正确实现：`1026_sage_attention_qk_int8_per_block`、`1027_sage_attention_qk_int8_per_block_causal`、`2013_groupnorm_bwd`、`4008_gemm_a16w4`、`4009_gemm_a8w8`、`4017_group_quant_int8`。
- **收敛速度**：首次 correct 的中位尝试序号 2.0，Success@1 29/66、Success@5 56/66。
- **跑得比 baseline 快**：Fast1@all 50/66（75.8%）；solved 题 best speedup 几何平均 8.01x、中位 6.39x、最好 900.66x；另有 10 道虽写对但整个预算内没跑赢 eager。
- **精修收益有限**：Refine Gain 中位 1.10x —— 第一次写对之后，剩下的尝试只能再提升这么多，难点在「第一次对不对」。

## 指标定义

| 指标 | 定义 |
|---|---|
| Compile Rate | 题目级：至少产出 1 个 `compile_ok` 候选的题目数 / accepted_tasks |
| Correct Rate | 题目级：至少产出 1 个 `correct` 候选的题目数 / accepted_tasks |
| Success@K | 每题 K 次尝试内至少出现 1 个 correct 候选的题目比例 |
| Fast1@K | 每题 K 次尝试内至少出现 1 个 correct 且快于参考 baseline（speedup > 1）的候选的题目比例 |
| Geomean Speedup | 每题 best correct 候选 speedup 的几何平均（只在 solved tasks 上算） |
| Median Speedup | 每题 best correct 候选 speedup 的中位数 |
| Best Speedup | 所有题里最高的 best correct speedup |
| Mean Speedup | 每题 best correct 候选 speedup 的算术平均 |
| First Correct Median | 每个 solved task 第一次出现 correct 的尝试序号的中位数 |
| Refine Gain Median | 每题（最终 best correct speedup / 第一次 correct 的 speedup）的中位数 |

两点口径说明：**① root 不在候选池内**（否则 Correct Rate 恒为 100%）；**② best = 所有 correct 候选里的最大 speedup**，与框架按 score 挑的 `best_node_id` 可能不同，逐题明细最后一列给出 champion 供对照。

| 指标 | 值 |
|---|---|
| Compile Rate | **66/66（100.0%）** |
| Correct Rate | **60/66（90.9%）** |
| Success@1 | 29/66（43.9%） |
| Success@5 | 56/66（84.8%） |
| Success@10 | 60/66（90.9%） |
| Fast1@1 | 26/66（39.4%） |
| Fast1@5 | 43/66（65.2%） |
| Fast1@10 | 50/66（75.8%） |
| Fast1@all（整个预算内） | 50/66（75.8%） |

K 曲线（K = 每题允许的尝试次数上限）：

| K | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Success@K | 29 | 47 | 52 | 55 | 56 | 58 | 58 | 59 | 60 | 60 |
| Fast1@K | 26 | 37 | 40 | 41 | 43 | 48 | 48 | 49 | 50 | 50 |

在 60 道 solved 题上统计 speedup。**best 的定义是「所有 correct 候选里的最大 speedup」**，与框架自己选的 `best_node_id`（按 score 挑）可能不同；下表最后一列给出 框架 champion 的 speedup 供对照：


| 指标 | 值 |
|---|---|
| Geomean Speedup | **8.01x** |
| Median Speedup | **6.39x** |
| Mean Speedup | 64.94x |
| Best Speedup | **900.66x** |
| First Correct Median（尝试序号） | **2.0** |
| Refine Gain Median | **1.10x** |

首次 correct 的尝试序号分布：1→29 道、2→18 道、3→5 道、4→3 道、5→1 道、6→2 道、8→1 道、9→1 道

## 逐题明细

| 题 | backend | 尝试数 | 首次 correct | 首次快于 baseline | best speedup（max correct） | best 节点 | 框架 champion |
|---|---|---|---|---|---|---|---|
| `7001_ctc_loss` | triton | 4 | 1 | 1 | 900.66x | n1 | 900.66x（n1） |
| `1032_deform_attention_bwd` | cuda | 10 | 1 | 1 | 634.36x | phase2_n3 | 634.36x（phase2_n3） |
| `1031_deform_attention_fwd` | cuda | 10 | 3 | 3 | 541.35x | phase2_n4 | 541.35x（phase2_n4） |
| `1016_lean_atten_paged` | triton | 10 | 2 | 2 | 388.39x | phase2_n2 | 378.18x（phase2_n5） |
| `5006_spconv_bwd` | cuda | 10 | 1 | 1 | 286.10x | n5 | 265.71x（phase2_n2） |
| `5004_deform_conv2d_bwd` | cuda | 10 | 2 | 2 | 246.03x | phase2_n4 | 246.03x（phase2_n4） |
| `1030_unified_attention` | triton | 10 | 1 | 1 | 169.23x | phase2_n3 | 143.00x（phase2_n1） |
| `2001_activation` | triton | 10 | 1 | 1 | 88.81x | phase2_n4 | 88.81x（phase2_n4） |
| `3007_moe_op_silu_fused` | triton | 10 | 2 | 6 | 70.15x | phase2_n1 | 70.15x（phase2_n1） |
| `1019_mha_onekernel_bwd` | triton | 10 | 2 | 2 | 63.95x | phase2_n4 | 63.95x（phase2_n4） |
| `3006_moe_op_mxfp4` | triton | 10 | 1 | 1 | 57.06x | n5 | 57.06x（n5） |
| `3003_moe_op` | triton | 10 | 4 | 5 | 43.98x | phase2_n1 | 34.17x（phase2_n4） |
| `1002_paged_attention` | triton | 10 | 1 | 1 | 41.85x | phase2_n3 | 40.34x（phase2_n1） |
| `2004_fused_qk_concat` | triton | 10 | 1 | 1 | 40.88x | phase2_n2 | 26.19x（phase2_n5） |
| `5009_conv_depthtospace` | cuda | 10 | 4 | 6 | 40.15x | phase2_n1 | 5.10x（root） |
| `2011_batchnorm_bwd` | cuda | 10 | 1 | 1 | 33.29x | phase2_n4 | 4.54x（n5） |
| `1029_triton_decode_attention` | triton | 10 | 1 | 1 | 27.14x | phase2_n1 | 27.14x（phase2_n1） |
| `5005_spconv_fprop` | cuda | 10 | 1 | 1 | 22.00x | n1 | 8.93x（n4） |
| `1014_hstu_attention` | triton | 10 | 1 | 1 | 20.57x | phase2_n1 | 20.57x（phase2_n1） |
| `1012_flash_attention_forward` | triton | 10 | 1 | 1 | 19.93x | n2 | 13.78x（phase2_n2） |
| `4005_fused_mxfp4_quant` | triton | 10 | 9 | 9 | 11.57x | n9 | 11.57x（n9） |
| `2008_rope` | triton | 10 | 2 | 2 | 11.01x | phase2_n4 | 11.01x（phase2_n4） |
| `3002_moe_align_block_size` | triton | 10 | 3 | 5 | 9.83x | phase2_n1 | 9.83x（phase2_n1） |
| `2006_norm` | triton | 10 | 2 | 2 | 9.55x | n2 | 9.55x（n2） |
| `1011_extend_attention` | triton | 10 | 1 | 1 | 9.51x | n1 | 9.51x（n1） |
| `1024_prefill_attention` | triton | 10 | 1 | 1 | 9.49x | n3 | 9.33x（n1） |
| `1021_pa_decode` | triton | 10 | 1 | 1 | 9.17x | phase2_n1 | 9.17x（phase2_n1） |
| `5003_deform_conv2d_fwd` | cuda | 10 | 1 | 1 | 7.91x | n1 | 7.33x（n4） |
| `2014_instancenorm_fwd` | cuda | 10 | 1 | 1 | 6.79x | n4 | 6.79x（n4） |
| `2010_batchnorm_fwd` | cuda | 10 | 1 | 1 | 6.51x | n4 | 5.46x（phase2_n4） |
| `1010_chunked_pa_prefill` | triton | 10 | 2 | 2 | 6.27x | n2 | 6.27x（n2） |
| `2002_add_swiglu` | triton | 10 | 2 | 2 | 5.52x | n3 | 5.52x（n3） |
| `5007_spconv_indice_pairs` | cuda | 10 | 3 | 3 | 5.18x | n4 | 5.04x（phase2_n3） |
| `6001_fused_adamw` | cuda | 10 | 1 | 1 | 4.93x | phase2_n5 | 4.93x（phase2_n5） |
| `2005_moe_activation` | triton | 10 | 2 | 2 | 4.37x | phase2_n1 | 4.37x（phase2_n1） |
| `4001_batched_gemm_a8w8` | triton | 10 | 1 | 1 | 4.03x | phase2_n3 | 4.03x（phase2_n3） |
| `1017_mha` | triton | 10 | 3 | 3 | 4.01x | n3 | 4.01x（n3） |
| `2007_rmsnorm` | triton | 10 | 1 | 1 | 3.93x | phase2_n1 | 3.92x（n1） |
| `1015_lean_atten` | triton | 10 | 2 | 2 | 3.33x | phase2_n2 | 3.32x（n3） |
| `3010_topk` | triton | 10 | 1 | 1 | 3.18x | n3 | 3.18x（n3） |
| `2009_softmax` | triton | 10 | 1 | 1 | 2.65x | n4 | 2.65x（n4） |
| `4016_gemm_w8a8` | triton | 10 | 8 | 8 | 2.34x | n9 | 2.34x（n9） |
| `4004_batched_gemm_bf16` | triton | 10 | 1 | 1 | 2.10x | phase2_n1 | 2.10x（phase2_n1） |
| `1022_pa_prefill` | triton | 10 | 2 | 2 | 1.94x | phase2_n4 | 1.90x（phase2_n3） |
| `1018_mha_fused_bwd` | triton | 10 | 6 | 6 | 1.92x | n7 | 1.92x（n7） |
| `2003_fused_mul_add` | triton | 10 | 4 | 4 | 1.78x | n4 | 1.78x（n4） |
| `8002_gather` | cuda | 4 | 1 | 1 | 1.72x | n2 | 1.71x（n3） |
| `8003_scatter_nd` | cuda | 4 | 2 | 2 | 1.65x | n3 | 1.65x（n3） |
| `5002_conv2d_bwd` | cuda | 10 | 2 | 6 | 1.35x | phase2_n1 | 1.29x（root） |
| `5008_conv_bias_act` | cuda | 10 | 2 | 6 | 1.17x | phase2_n3 | 1.17x（phase2_n3） |
| `4018_quant` | triton | 8 | 5 | - | 0.99x | n5 | 0.99x（root） |
| `4006_gemm_a16w16` | triton | 10 | 2 | - | 0.93x | n2 | 1.00x（root） |
| `3004_moe_op_e2e` | triton | 10 | 2 | - | 0.80x | phase2_n4 | 1.01x（root） |
| `4010_gemm_a8w8_blockscale` | triton | 10 | 6 | - | 0.78x | n6 | 0.97x（root） |
| `1020_mla_decode_rope` | triton | 10 | 2 | - | 0.65x | phase2_n1 | 1.01x（root） |
| `5001_conv2d_fwd` | cuda | 10 | 1 | - | 0.64x | phase2_n3 | 0.98x（root） |
| `8001_cumsum` | cuda | 5 | 1 | - | 0.28x | n2 | 1.00x（root） |
| `4007_gemm_a16w16_atomic` | triton | 10 | 3 | - | 0.26x | n3 | 0.99x（root） |
| `2012_groupnorm_fwd` | cuda | 10 | 1 | - | 0.21x | phase2_n3 | 1.02x（root） |
| `3009_routing` | triton | 10 | 2 | - | 0.16x | n2 | 0.97x（root） |
| `1026_sage_attention_qk_int8_per_block` | triton | 10 | - | - | - | - | 2.16x（root） |
| `1027_sage_attention_qk_int8_per_block_causal` | triton | 10 | - | - | - | - | 1.01x（root） |
| `2013_groupnorm_bwd` | cuda | 10 | - | - | - | - | 1.00x（root） |
| `4008_gemm_a16w4` | triton | 10 | - | - | - | - | 0.99x（root） |
| `4009_gemm_a8w8` | triton | 10 | - | - | - | - | 0.99x（root） |
| `4017_group_quant_int8` | triton | 10 | - | - | - | - | 1.00x（root） |

## 离线终审（隐藏 case + 静态审计，分母 = `aiter`）

- **66 道全审，通过 43 道**
- 未通过 23：静态判负 19、隐藏 case 跑不起来 3、数值不达标 1
- `aiter` 有测值 25 道：中位 **1.29x**、均值 1.46x、最大 3.69x；MI 兜底基线 6 道（分母偏大，加速比偏高，引用须标注）

**为什么不能只看在线**：在线只跑 `get_inputs()` 的单一 shape、不执行禁用列表，
所以在线的 60 道 solved 实现里，有 19 道把核心计算交给了被禁的算子、3 道换一批隐藏 shape 就崩、1 道数值对不上。离线才是判据的硬边。

| 静态判负的题 |
|---|
| `1018_mha_fused_bwd` |
| `1020_mla_decode_rope` |
| `1026_sage_attention_qk_int8_per_block` |
| `1027_sage_attention_qk_int8_per_block_causal` |
| `1031_deform_attention_fwd` |
| `1032_deform_attention_bwd` |
| `2013_groupnorm_bwd` |
| `3004_moe_op_e2e` |
| `3009_routing` |
| `4006_gemm_a16w16` |
| `4007_gemm_a16w16_atomic` |
| `4008_gemm_a16w4` |
| `4009_gemm_a8w8` |
| `4010_gemm_a8w8_blockscale` |
| `5001_conv2d_fwd` |
| `5002_conv2d_bwd` |
| `5004_deform_conv2d_bwd` |
| `5009_conv_depthtospace` |
| `8001_cumsum` |

| 隐藏 case 跑不起来的题 | 崩溃摘要 |
|---|---|
| `1016_lean_atten_paged` | `: out of resource: shared memory, Required: 98304, Hardware limit: 65536. Reducing block sizes or `num_stages` may help.` |
| `1022_pa_prefill` | `d_m = tl.program_id(0) h = tl.program_id(1) rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) d = tl.arange(0, HEAD_SIZE) ^` |
| `3007_moe_op_silu_fused` | ` + rk av = tl.load(A + token[:, None]*A0 + kk[None, :]*A1, valid[:, None] & (kk[None, :] < K), other=0).to(tl.float32) ^` |

### 加离线门槛后的口径（对外引用建议用这一栏）

| 指标 | 只算在线 | **在线 ∩ 离线通过** |
|---|---|---|
| 算 solved 的题数 | 60/66 | **42/66** |
| 加速比中位 | 6.39x | **7.35x** |
| 加速比均值 | 64.94x | **40.00x** |
| 加速比几何平均 | 8.01x | **8.79x** |
| 加速比最大 | 900.66x | 900.66x |

被门槛剔掉的 18 道：`1016_lean_atten_paged`、`1018_mha_fused_bwd`、`1020_mla_decode_rope`、`1022_pa_prefill`、`1031_deform_attention_fwd`、`1032_deform_attention_bwd`、`3004_moe_op_e2e`、`3007_moe_op_silu_fused`、`3009_routing`、`4006_gemm_a16w16`、`4007_gemm_a16w16_atomic`、`4010_gemm_a8w8_blockscale`、`5001_conv2d_fwd`、`5002_conv2d_bwd`、`5004_deform_conv2d_bwd`、`5006_spconv_bwd`、`5009_conv_depthtospace`、`8001_cumsum`

注意：`Success@K` / `First Correct Median` / `Refine Gain Median` **不能**这样加门槛——
离线终审只审最终 `best_code.py`，不审每一次尝试；要按尝试门槛化就得把每个候选都拿去审计
（约 66×12≈790 次，单题 1–2 分钟），代价过高，所以这三项保留在线口径并在结论里注明。

| 题 | vs aiter | vs eager | 备注 |
|---|---|---|---|
| `4001_batched_gemm_a8w8` | **3.69x** | 1.75x | MI 兜底† |
| `2002_add_swiglu` | **2.98x** | 11.42x |  |
| `2007_rmsnorm` | **2.90x** | 5.27x |  |
| `2006_norm` | **2.77x** | 12.15x |  |
| `2004_fused_qk_concat` | **2.68x** | 3.14x |  |
| `3002_moe_align_block_size` | **2.26x** | 1.12x |  |
| `4004_batched_gemm_bf16` | **2.20x** | 1.25x | MI 兜底† |
| `1024_prefill_attention` | **1.98x** | 9.43x |  |
| `2005_moe_activation` | **1.59x** | 4.73x |  |
| `1017_mha` | **1.58x** | 4.17x | MI 兜底† |
| `2003_fused_mul_add` | **1.47x** | 3.04x |  |
| `2009_softmax` | **1.34x** | 4.56x |  |
| `3010_topk` | **1.29x** | 1.52x |  |
| `3003_moe_op` | **1.06x** | 18.73x |  |
| `1021_pa_decode` | **1.04x** | 17.39x |  |
| `2008_rope` | **1.01x** | 4.13x |  |
| `1015_lean_atten` | **0.92x** | 1.24x |  |
| `1010_chunked_pa_prefill` | **0.77x** | 1.60x |  |
| `4018_quant` | **0.69x** | 1.01x |  |
| `1012_flash_attention_forward` | **0.66x** | 1.24x |  |
| `4016_gemm_w8a8` | **0.57x** | 1.03x |  |
| `1002_paged_attention` | **0.44x** | 11.97x |  |
| `4017_group_quant_int8` | **0.40x** | 1.02x |  |
| `4005_fused_mxfp4_quant` | **0.11x** | 2.25x |  |
| `1014_hstu_attention` | **0.09x** | 4.06x | MI 兜底† |

## 两点观察

- **Correct Rate 60/66 但 Fast1@all 只有 50/66**：差出来的 10 道是「能写对、但整个预算内没跑赢 `torch_eager`」（`1020_mla_decode_rope`、`2012_groupnorm_fwd`、`3004_moe_op_e2e`、`3009_routing`、`4006_gemm_a16w16`、`4007_gemm_a16w16_atomic`…）——这 10 道横跨 quant/GEMM、MoE、decode-attention 与 conv，共同点是 eager 分母本身已经是 ATen/MIOpen 的高度优化实现，生成实现很难超过。
- **Refine Gain 中位只有 1.10x**：一旦第一次写对，后续尝试平均只再提升这么多。也就是说瓶颈在「第一次能不能写对」（Success@1 29/66），而不是精修阶段——把预算从 10 次加到 20 次，预期收益有限。

## 最高加速比为什么这么高（`7001_ctc_loss` 900.66x）

分母是题面给的**朴素参考实现**：`_ctc_one` 对每个 batch 元素各跑一遍 Python 循环（前向 `for t in range(1, tb)` + 反向再来一遍），每一步临时造 6~7 个张量、发 6~7 次 kernel。在线 shape（T=500、N=4、tb≤200）下约 **1.1 万次 launch**，272 ms 几乎全是主机侧开销。

生成的实现是一个 Triton kernel：一个 program 处理一个 batch 元素，状态维 S 放寄存器（`tl.arange(0,B)`），**时间维递推写在 kernel 内部**，对数域三分支 logsumexp 手写（`_ctc_lse3`），前向存 alpha、反向在同一 kernel 里算 gamma 并 `atomic_add` 写梯度 —— 整个 500 步递推只有 **1 次 launch**，0.30 ms。

所以这 900x 的绝大部分来自**消除上万次 launch + Python 开销**，算法复杂度两边一样（都是 O(T·S) 对数域 CTC），没有任何渐进优势。

**引用时必须注明分母**：不能写成“比生产级 CTC 快 900 倍”。真正对位的生产实现是 `torch.nn.functional.ctc_loss`，而它正是本题的禁用算子（所以不会出现在分母里）。离线那 4319x 更是同一原因在更大 perf case（T=1024、N=16、C=32）上的放大：参考实现成本 ∝ T·N，生成实现把 N 摊到多个 program、T 留在 kernel 内循环。

