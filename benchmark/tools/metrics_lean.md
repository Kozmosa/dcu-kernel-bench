# 题目级指标（.dcu_runs\lean66）

- accepted_tasks = **66**（solved = 60）
- 候选口径：**排除 root**（root 是框架直通脚手架，不算 Agent 产出）
- 参考 baseline（speedup 分母）：`torch_eager`
- 数据来源：B 臂：精简题面（-51.8%），runs/dcukb_lean_1020_1030 61 道 + 补跑 5 道

## 简要描述
- **能编译**：63/66 道题至少产出一个可编译候选，脚手架不是瓶颈。
- **能写对**：Correct Rate 60/66（90.9%）；6 道始终没写出正确实现：`1026_sage_attention_qk_int8_per_block`、`2013_groupnorm_bwd`、`2014_instancenorm_fwd`、`4008_gemm_a16w4`、`4009_gemm_a8w8`、`5004_deform_conv2d_bwd`。
- **收敛速度**：首次 correct 的中位尝试序号 2.0，Success@1 24/66、Success@5 56/66。
- **跑得比 baseline 快**：Fast1@all 54/66（81.8%）；solved 题 best speedup 几何平均 10.02x、中位 7.45x、最好 872.61x；另有 6 道虽写对但整个预算内没跑赢 eager。
- **精修收益有限**：Refine Gain 中位 1.13x —— 第一次写对之后，剩下的尝试只能再提升这么多，难点在「第一次对不对」。

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
| Compile Rate | **63/66（95.5%）** |
| Correct Rate | **60/66（90.9%）** |
| Success@1 | 24/66（36.4%） |
| Success@5 | 56/66（84.8%） |
| Success@10 | 60/66（90.9%） |
| Fast1@1 | 20/66（30.3%） |
| Fast1@5 | 50/66（75.8%） |
| Fast1@10 | 54/66（81.8%） |
| Fast1@all（整个预算内） | 54/66（81.8%） |

K 曲线（K = 每题允许的尝试次数上限）：

| K | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Success@K | 24 | 41 | 48 | 53 | 56 | 56 | 57 | 58 | 59 | 60 |
| Fast1@K | 20 | 36 | 42 | 47 | 50 | 50 | 52 | 53 | 54 | 54 |

在 60 道 solved 题上统计 speedup。**best 的定义是「所有 correct 候选里的最大 speedup」**，与框架自己选的 `best_node_id`（按 score 挑）可能不同；下表最后一列给出 框架 champion 的 speedup 供对照：


| 指标 | 值 |
|---|---|
| Geomean Speedup | **10.02x** |
| Median Speedup | **7.45x** |
| Mean Speedup | 52.17x |
| Best Speedup | **872.61x** |
| First Correct Median（尝试序号） | **2.0** |
| Refine Gain Median | **1.13x** |

首次 correct 的尝试序号分布：1→24 道、2→17 道、3→7 道、4→5 道、5→3 道、7→1 道、8→1 道、9→1 道、10→1 道

## 逐题明细

| 题 | backend | 尝试数 | 首次 correct | 首次快于 baseline | best speedup（max correct） | best 节点 | 框架 champion |
|---|---|---|---|---|---|---|---|
| `7001_ctc_loss` | triton | 10 | 1 | 1 | 872.61x | phase2_n1 | 872.61x（phase2_n1） |
| `1016_lean_atten_paged` | triton | 10 | 1 | 1 | 337.30x | phase2_n1 | 331.20x（phase2_n4） |
| `3003_moe_op` | triton | 10 | 2 | 2 | 266.52x | n2 | 266.52x（n2） |
| `5006_spconv_bwd` | cuda | 10 | 3 | 3 | 264.83x | phase2_n1 | 264.83x（phase2_n1） |
| `1031_deform_attention_fwd` | cuda | 10 | 1 | 1 | 230.92x | phase2_n5 | 230.92x（phase2_n5） |
| `1032_deform_attention_bwd` | cuda | 10 | 7 | 7 | 152.22x | n8 | 142.14x（n7） |
| `5007_spconv_indice_pairs` | cuda | 10 | 5 | 5 | 148.08x | phase2_n4 | 146.15x（phase2_n1） |
| `1030_unified_attention` | triton | 10 | 2 | 2 | 136.99x | phase2_n1 | 136.99x（phase2_n1） |
| `2001_activation` | triton | 10 | 1 | 1 | 79.82x | n5 | 79.31x（phase2_n1） |
| `3007_moe_op_silu_fused` | triton | 10 | 1 | 1 | 65.85x | n3 | 62.07x（n1） |
| `5003_deform_conv2d_fwd` | cuda | 10 | 3 | 3 | 58.33x | phase2_n1 | 58.33x（phase2_n1） |
| `3006_moe_op_mxfp4` | triton | 10 | 2 | 2 | 44.39x | phase2_n1 | 43.59x（phase2_n2） |
| `4005_fused_mxfp4_quant` | triton | 10 | 3 | 3 | 41.73x | n3 | 39.46x（phase2_n1） |
| `1002_paged_attention` | triton | 10 | 1 | 1 | 41.41x | phase2_n1 | 41.41x（phase2_n1） |
| `1010_chunked_pa_prefill` | triton | 10 | 2 | 2 | 32.03x | phase2_n4 | 32.03x（phase2_n4） |
| `1022_pa_prefill` | triton | 10 | 1 | 1 | 30.04x | phase2_n2 | 29.42x（phase2_n4） |
| `1029_triton_decode_attention` | triton | 10 | 2 | 2 | 29.31x | n2 | 29.31x（n2） |
| `2004_fused_qk_concat` | triton | 10 | 1 | 1 | 27.32x | phase2_n1 | 26.97x（n3） |
| `1019_mha_onekernel_bwd` | triton | 10 | 9 | 9 | 25.68x | n9 | 25.68x（n9） |
| `3002_moe_align_block_size` | triton | 10 | 1 | 1 | 24.42x | phase2_n1 | 24.42x（phase2_n1） |
| `1011_extend_attention` | triton | 10 | 3 | 3 | 19.17x | n5 | 18.80x（n3） |
| `1017_mha` | triton | 10 | 3 | 3 | 18.94x | n3 | 17.69x（phase2_n2） |
| `1014_hstu_attention` | triton | 10 | 1 | 1 | 17.19x | phase2_n2 | 17.05x（phase2_n1） |
| `3004_moe_op_e2e` | triton | 10 | 4 | 4 | 13.64x | phase2_n1 | 12.93x（phase2_n2） |
| `1012_flash_attention_forward` | triton | 10 | 2 | 2 | 12.68x | phase2_n1 | 12.36x（phase2_n4） |
| `2012_groupnorm_fwd` | cuda | 10 | 1 | 2 | 11.64x | phase2_n5 | 11.36x（phase2_n4） |
| `1024_prefill_attention` | triton | 10 | 1 | 1 | 10.00x | phase2_n2 | 10.00x（phase2_n2） |
| `2006_norm` | triton | 10 | 2 | 2 | 9.30x | phase2_n1 | 9.30x（phase2_n1） |
| `1027_sage_attention_qk_int8_per_block_causal` | triton | 10 | 4 | 4 | 8.41x | phase2_n3 | 8.41x（phase2_n3） |
| `4018_quant` | triton | 10 | 2 | 2 | 8.05x | phase2_n5 | 8.05x（phase2_n5） |
| `6001_fused_adamw` | cuda | 10 | 2 | 2 | 6.84x | phase2_n2 | 6.84x（phase2_n2） |
| `2008_rope` | triton | 10 | 1 | 1 | 6.64x | phase2_n1 | 6.64x（phase2_n1） |
| `4017_group_quant_int8` | triton | 10 | 2 | 2 | 6.44x | phase2_n1 | 6.44x（phase2_n1） |
| `1015_lean_atten` | triton | 10 | 1 | 1 | 5.60x | n2 | 5.52x（n3） |
| `2002_add_swiglu` | triton | 10 | 4 | 4 | 5.42x | n5 | 5.42x（n5） |
| `2010_batchnorm_fwd` | cuda | 10 | 3 | 3 | 4.63x | phase2_n2 | 4.61x（n5） |
| `2005_moe_activation` | triton | 10 | 1 | 1 | 4.29x | n3 | 4.29x（n3） |
| `4010_gemm_a8w8_blockscale` | triton | 10 | 2 | 2 | 4.24x | n5 | 3.22x（phase2_n1） |
| `4016_gemm_w8a8` | triton | 10 | 2 | 2 | 4.14x | phase2_n1 | 4.14x（phase2_n1） |
| `5005_spconv_fprop` | cuda | 10 | 8 | 8 | 3.85x | n8 | 3.85x（n8） |
| `3010_topk` | triton | 10 | 1 | 2 | 3.79x | phase2_n5 | 3.79x（phase2_n5） |
| `1021_pa_decode` | triton | 10 | 2 | 2 | 3.56x | phase2_n1 | 3.56x（phase2_n1） |
| `4004_batched_gemm_bf16` | triton | 10 | 1 | 1 | 3.26x | n1 | 3.26x（n1） |
| `8003_scatter_nd` | cuda | 10 | 1 | 1 | 3.21x | phase2_n2 | 2.80x（phase2_n1） |
| `4001_batched_gemm_a8w8` | triton | 10 | 1 | 1 | 2.91x | phase2_n1 | 2.81x（phase2_n2） |
| `2007_rmsnorm` | triton | 10 | 2 | 2 | 2.82x | n2 | 2.82x（n2） |
| `2009_softmax` | triton | 10 | 1 | 1 | 2.60x | phase2_n4 | 2.60x（phase2_n4） |
| `8001_cumsum` | cuda | 10 | 1 | 1 | 2.38x | phase2_n4 | 1.68x（n2） |
| `5008_conv_bias_act` | cuda | 10 | 5 | 5 | 2.37x | phase2_n4 | 2.37x（phase2_n4） |
| `2011_batchnorm_bwd` | cuda | 10 | 1 | 1 | 2.26x | phase2_n5 | 2.26x（phase2_n5） |
| `8002_gather` | cuda | 10 | 4 | 4 | 1.87x | n4 | 1.85x（phase2_n1） |
| `2003_fused_mul_add` | triton | 10 | 4 | 4 | 1.78x | phase2_n4 | 1.78x（phase2_n4） |
| `1018_mha_fused_bwd` | triton | 10 | 2 | 7 | 1.53x | phase2_n4 | 1.50x（phase2_n5） |
| `5002_conv2d_bwd` | cuda | 10 | 5 | 5 | 1.20x | phase2_n4 | 0.82x（phase2_n1） |
| `5009_conv_depthtospace` | cuda | 10 | 10 | - | 0.86x | n10 | 0.99x（root） |
| `3009_routing` | triton | 10 | 1 | - | 0.81x | n1 | 0.94x（root） |
| `1020_mla_decode_rope` | triton | 10 | 2 | - | 0.71x | phase2_n3 | 0.99x（root） |
| `4007_gemm_a16w16_atomic` | cuda | 10 | 2 | - | 0.46x | phase2_n5 | 0.99x（root） |
| `4006_gemm_a16w16` | triton | 10 | 3 | - | 0.42x | n3 | 1.00x（root） |
| `5001_conv2d_fwd` | cuda | 10 | 1 | - | 0.32x | phase2_n5 | 1.01x（root） |
| `1026_sage_attention_qk_int8_per_block` | triton | 10 | - | - | - | - | 0.99x（root） |
| `2013_groupnorm_bwd` | cuda | 10 | - | - | - | - | 1.00x（root） |
| `2014_instancenorm_fwd` | cuda | 10 | - | - | - | - | 0.99x（root） |
| `4008_gemm_a16w4` | triton | 10 | - | - | - | - | 1.02x（root） |
| `4009_gemm_a8w8` | triton | 10 | - | - | - | - | 1.11x（root） |
| `5004_deform_conv2d_bwd` | cuda | 10 | - | - | - | - | 0.95x（root） |

## 两点观察

- **Correct Rate 60/66 但 Fast1@all 只有 54/66**：差出来的 6 道是「能写对、但整个预算内没跑赢 `torch_eager`」（`1020_mla_decode_rope`、`3009_routing`、`4006_gemm_a16w16`、`4007_gemm_a16w16_atomic`、`5001_conv2d_fwd`、`5009_conv_depthtospace`）——这 10 道横跨 quant/GEMM、MoE、decode-attention 与 conv，共同点是 eager 分母本身已经是 ATen/MIOpen 的高度优化实现，生成实现很难超过。
- **Refine Gain 中位只有 1.13x**：一旦第一次写对，后续尝试平均只再提升这么多。也就是说瓶颈在「第一次能不能写对」（Success@1 24/66），而不是精修阶段——把预算从 10 次加到 20 次，预期收益有限。

## 最高加速比为什么这么高（`7001_ctc_loss` 872.61x）

分母是题面给的**朴素参考实现**：`_ctc_one` 对每个 batch 元素各跑一遍 Python 循环（前向 `for t in range(1, tb)` + 反向再来一遍），每一步临时造 6~7 个张量、发 6~7 次 kernel。在线 shape（T=500、N=4、tb≤200）下约 **1.1 万次 launch**，272 ms 几乎全是主机侧开销。

生成的实现是一个 Triton kernel：一个 program 处理一个 batch 元素，状态维 S 放寄存器（`tl.arange(0,B)`），**时间维递推写在 kernel 内部**，对数域三分支 logsumexp 手写（`_ctc_lse3`），前向存 alpha、反向在同一 kernel 里算 gamma 并 `atomic_add` 写梯度 —— 整个 500 步递推只有 **1 次 launch**，0.30 ms。

所以这 900x 的绝大部分来自**消除上万次 launch + Python 开销**，算法复杂度两边一样（都是 O(T·S) 对数域 CTC），没有任何渐进优势。

**引用时必须注明分母**：不能写成“比生产级 CTC 快 900 倍”。真正对位的生产实现是 `torch.nn.functional.ctc_loss`，而它正是本题的禁用算子（所以不会出现在分母里）。离线那 4319x 更是同一原因在更大 perf case（T=1024、N=16、C=32）上的放大：参考实现成本 ∝ T·N，生成实现把 N 摊到多个 program、T 留在 kernel 内循环。

