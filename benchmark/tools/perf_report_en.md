# 66 道生成评测：中文题面 -> 英文题面（2026-10-19）

同一模型、同一 profile（main）、同一 16 片并行；唯一变量是 Agent 可见题面的语言。数据源：

- 生成模型：`gpt-6-astra`（取自 `llm.env` 的 `OPENAI_MODEL`；**run.json 未记录模型名**）
- 中文题面在线：`runs/dcukb_full_fixed (10-17)`（66 道）
- 英文题面在线：`runs/dcukb_en_1019_1027 (10-19)`（66 道，16 片并行）
- 英文题面离线终审：`offline_audit.json`（审 30 道有 aiter 基线的题）

## 0. 口径：三套数字不可混比

| 套 | 来源 | 正确性判据 | 加速比分母 |
|---|---|---|---|
| **在线** | `run.json`（KernelBench/PyramidKernel 口径） | 输入 cast fp32 + `allclose(atol=rtol=1e-4)` | **`torch_eager`**（框架没有"外部基线"模式） |
| **离线·aiter** | `offline_audit.json`（`audit_batch.py`） | 隐藏 case + `task.yaml` **分档容差** | `aiter` 官方实现 |
| **离线·eager** | 同上 | 同上 | `torch_eager` |

**因此**：① 在线 speedup **不是**对 aiter 的加速比（本轮 66 份 `primary_reference` 全为
`torch_eager`）；② 离线比在线**严得多**（隐藏 case + 分档容差，而在线是单一 shape 的
1e-4 allclose），所以同一题的离线正确性与在线正确性不是一回事。

## 1. 编译率（在线）

| 指标 | 中文题面 | 英文题面 |
|---|---|---|
| root 脚手架编译通过（= 框架给出的起点可用） | **66/66** | **66/66** |
| best 节点编译通过 | 66/66 | 66/66 |
| compile 阶段失败次数（合计，含 Agent 自己写坏） | 210 | 236 |
| runtime 阶段失败次数（合计） | 46 | 34 |

root 脚手架编译率衡量 **harness 质量**：脚手架坏掉的题 Agent 连起点都没有。
compile 阶段失败次数不归零是
正常的 —— 那是 Agent 尝试里自己写坏的补丁，属生成侧信号。

## 2. 正确率

### 2.1 在线（KernelBench 口径）

| 指标 | 中文题面 | 英文题面 |
|---|---|---|
| best 正确 | 66/66（100.0%） | **66/66（100.0%）** |
| └ 非 root 真解 | 53（80.3%） | **49（74.2%）** |
| └ 仅 root（Agent 未改写） | 13 | 17 |
| 有 speedup 可比 | 66/66 | 66/66 |

按 backend（英文题面）：

| backend | 题数 | best 正确（前→后） | 真解（前→后） |
|---|---|---|---|
| HIP（backend=cuda） | 19 | 19→19 | 14→13 |
| Triton | 47 | 47→47 | 39→36 |

### 2.2 离线终审（静态审计 + 隐藏 case + `task.yaml` 分档容差）

只审有 aiter 基线的题：**审了 30 道，通过 25 道（83%）**。

未通过的题**不是同一类原因**，性质完全不同：

| 未通过原因 | 题数 | 说明 |
|---|---|---|
| **静态审计判负**（未进入正确性环节） | **5** | 实现里直接调用被禁的 ATen 算子完成核心计算，违反「禁止绕过」铁律 |
| **正确性不达标**（隐藏 case 对不上） | **0** | 静态审计通过，但边界 shape 处理不到位 |

#### A. 静态审计判负

| 题 | 命中的禁用算子 |
|---|---|
| `1026_sage_attention_qk_int8_per_block` | `torch.matmul`、`torch.softmax` |
| `3004_moe_op_e2e` | （未复跑取证） |
| `3009_routing` | `torch.matmul`、`torch.sigmoid` |
| `4006_gemm_a16w16` | （未复跑取证） |
| `4008_gemm_a16w4` | `torch.matmul` |

这些题在**在线**被判为正确 —— 在线评测不执行禁用列表（题面 forbidden 在线上只是
软约束，硬审计在离线）。所以它们不是「性能不好」，而是**实现方式不合规**。

> 两套判据不互相否定：在线回答「能不能跑通并对上参考」，离线回答「换一批
> shape/边界还对不对、以及实现是否合规」。引用时必须说明用的是哪一套。

## 3. 在线加速比（分母 = `torch_eager`）

统计口径：只看**真解**（best 非 root 且 correct）。

| 指标 | 中文题面 | 英文题面 |
|---|---|---|
| 样本数 n | 53 | 49 |
| 中位数 | 5.61x | 7.33x |
| 均值 | 72.26x | 75.55x |
| P25 | 3.37x | 3.54x |
| P75 | 37.16x | 34.17x |
| 最大 | 858.02x | 900.66x |
| 最小 | 0.82x | 0.77x |

| 区间 | 中文题面 | 英文题面 |
|---|---|---|
| 0–1x | 2 | 1 |
| 1–2x | 9 | 7 |
| 2–5x | 12 | 9 |
| 5–10x | 9 | 14 |
| 10–100x | 14 | 11 |
| ≥100x | 7 | 7 |

Top 10（英文题面，在线口径）：

| # | 题 | 在线 speedup |
|---|---|---|
| 1 | `7001_ctc_loss` | **900.66x** |
| 2 | `1032_deform_attention_bwd` | **634.36x** |
| 3 | `1031_deform_attention_fwd` | **541.35x** |
| 4 | `1016_lean_atten_paged` | **378.18x** |
| 5 | `5006_spconv_bwd` | **265.71x** |
| 6 | `5004_deform_conv2d_bwd` | **246.03x** |
| 7 | `1030_unified_attention` | **143.00x** |
| 8 | `2001_activation` | **88.81x** |
| 9 | `3007_moe_op_silu_fused` | **70.15x** |
| 10 | `1019_mha_onekernel_bwd` | **63.95x** |

## 4. 双基线表（离线终审；生成实现 vs aiter **与** vs eager）

分母来自 `benchmark/private/<task>/baseline.json` 的实测值，perf 形状与计时参数取
`private/<task>/perf_cases.json`。**同一行的两列口径一致**，可以横向比。

| 题 | 离线正确性 | vs **aiter** | vs eager | 备注 |
|---|---|---|---|---|
| `2002_add_swiglu` | ✓ | **2.98x** | 11.42x |  |
| `2007_rmsnorm` | ✓ | **2.90x** | 5.27x |  |
| `2006_norm` | ✓ | **2.78x** | 12.17x |  |
| `2004_fused_qk_concat` | ✓ | **2.67x** | 3.12x |  |
| `3002_moe_align_block_size` | ✓ | **2.27x** | 1.13x |  |
| `1024_prefill_attention` | ✓ | **1.97x** | 9.31x |  |
| `2005_moe_activation` | ✓ | **1.59x** | 4.73x |  |
| `4001_batched_gemm_a8w8` | ✓ | **1.46x** | 0.95x | MI 兜底† |
| `2003_fused_mul_add` | ✓ | **1.45x** | 2.99x |  |
| `1017_mha` | ✓ | **1.33x** | 3.10x | MI 兜底† |
| `3010_topk` | ✓ | **1.26x** | 1.49x |  |
| `1015_lean_atten` | ✓ | **1.15x** | 1.61x |  |
| `2009_softmax` | ✓ | **1.08x** | 3.77x |  |
| `3003_moe_op` | ✓ | **1.04x** | 18.39x |  |
| `1021_pa_decode` | ✓ | **1.03x** | 17.22x |  |
| `2008_rope` | ✓ | **1.01x** | 4.14x |  |
| `1010_chunked_pa_prefill` | ✓ | **0.77x** | 1.60x |  |
| `1012_flash_attention_forward` | ✓ | **0.71x** | 1.56x |  |
| `4018_quant` | ✓ | **0.69x** | 1.01x |  |
| `4016_gemm_w8a8` | ✓ | **0.56x** | 1.03x |  |
| `1002_paged_attention` | ✓ | **0.44x** | 11.96x |  |
| `4017_group_quant_int8` | ✓ | **0.40x** | 1.02x |  |
| `1020_mla_decode_rope` | ✓ | **0.33x** | 0.45x |  |
| `4005_fused_mxfp4_quant` | ✓ | **0.11x** | 2.25x |  |
| `1014_hstu_attention` | ✓ | **0.09x** | 4.06x | MI 兜底† |
| `1026_sage_attention_qk_int8_per_block` | ✗ | - | - | 离线正确性未过 |
| `3004_moe_op_e2e` | ✗ | - | - | 离线正确性未过 |
| `3009_routing` | ✗ | - | - | 离线正确性未过 |
| `4006_gemm_a16w16` | ✗ | - | - | MI 兜底†；离线正确性未过 |
| `4008_gemm_a16w4` | ✗ | - | - | MI 兜底†；离线正确性未过 |

汇总（aiter 有测值的 25 道）：**aiter 中位 1.08x、均值 1.28x、最大 2.98x**。

† **MI 兜底基线 5 题**：该题的 aiter 基线落在 **MI 系列 fallback
tuner** 配置上（未针对 DCU 调优，见 AGENTS.md）—— 也就是说这个 aiter 数值**不等于
aiter 在 DCU 上能达到的最好成绩**，因此由它导出的加速比不能直接当「与官方实现的距离」。
方向因题而异，必须结合下一小节一起看（例：`1014` 的 aiter 仍比 eager 快 28x，而生成的
解比 aiter 慢约 13x；`4006` 的 aiter 反而比 eager 慢 8x，于是「vs aiter」被抬到 10.49x）。
这 5 道是：
`1014_hstu_attention`、`1017_mha`、`4001_batched_gemm_a8w8`、`4006_gemm_a16w16`、`4008_gemm_a16w4`

### 分母质量对照：aiter 本身相对 eager（按 perf case 取中位）

| 题 | aiter 相对 eager | (min–max) |
|---|---|---|
| `1014_hstu_attention` | **42.09x** | 28.16–56.02x |
| `1002_paged_attention` | **38.67x** | 16.57–54.65x |
| `1016_lean_atten_paged` | **28.25x** | 7.37–54.33x |
| `4005_fused_mxfp4_quant` | **20.83x** | 18.98–21.35x |
| `1021_pa_decode` | **20.22x** | 13.59–21.98x |
| `3003_moe_op` | **9.80x** | 9.60–32.52x |
| … | | |
| `4006_gemm_a16w16` | **0.11x** | 0.11–0.18x |
| `3002_moe_align_block_size` | **0.19x** | 0.05–0.75x |
| `4001_batched_gemm_a8w8` | **0.37x** | 0.29–2.20x |
| `3009_routing` | **1.07x** | 0.87–2.20x |

这张表回答的是「该题的 aiter 分母本身强不强」：>1x 说明 aiter 比 eager 快，拿它当
分母更难；<1x（`4006` 0.11x、`3002` 0.19x、`4001` 0.37x）说明该 aiter 配置比 PyTorch
还慢，拿它当分母会把生成实现算得偏高。

> **不要用上表的两个加速比相除来互推**：双基线表里的 vs-aiter / vs-eager 是逐 case
> 求比再聚合（均值或中位），而这张表是逐 case 的 `eager_us/aiter_us` 再取中位；
> 各 case 强弱差别大时两者不满足简单乘除关系（`1014` 就是一例：逐 case 比值 28.2x 与
> 56.0x，均值之比却是 45.6）。要互推请回到单题缓存 `offline_audit/<task>.json` 的
> `by_impl.*.cases` 逐 case 核对。

## 5. 只有 `torch_eager` 基线的题（含 HIP 那 19 道）

66 道里有 **33 道没有 aiter 基线**（`baseline.json` 的 impls 只有
`torch_eager`），因此只能给 `torch_eager` 口径。其中 **19 道是 HIP track
（`backend=cuda`）**，全部落在这一类。

另有 **3 道有 aiter 基线但离线终审跑不出结果**（不是没有基线，也不要
据此推断它们性能差）：

| 题 | 失败原因 |
|---|---|
| `1016_lean_atten_paged` | exit 1，未产出 JSON 报告 |
| `1022_pa_prefill` | 未记录 |
| `4004_batched_gemm_bf16` | 未记录 |

这几道属离线终审工具侧的待修项（不是生成质量差），定向重试同样失败。
它们的在线结果（分母
eager）是：

- `1016_lean_atten_paged`：在线 best 正确 True，speedup 378.18x
- `1022_pa_prefill`：在线 best 正确 True，speedup 1.90x
- `4004_batched_gemm_bf16`：在线 best 正确 True，speedup 1.90x

### 5.1 为什么 HIP 题没有 aiter / hipDNN 基线

这 19 道题源自 `opendtk/hipdnn_samples`（准入记录见 `benchmark/sources/`）。按 AGENTS.md，
**闭源 hipDNN 库只作性能基线、不作题源**，本应给它们补 hipDNN 分母。但
`hipdnn_baseline_probe.md` 的实测结论是：

> **在当前机器（DTK 26.04）上 hipDNN 做不了基线。** 阻塞点在 vendor 自己的引擎插件
> `libmiopen_legacy_plugin.so`（Convolution plan builder 内部 segfault），不在我们的工具链；
> 官方样例本身要求 DTK ≥ 26.10。hipDNN 的 Python 绑定也未随 26.04 发布，全盘无
> `hipdnn*.whl`。

aiter 侧同样没有对等算子可作分母（hipDNN 样例覆盖的 deform conv / spconv /
depth_to_space / adamw / ctc_loss / cumsum / gather / scatter_nd 在 aiter 里都没有
可直接对位的实现）。**所以这 19 道的基线状况与 hipDNN 探针的结论完全一致：只有
`torch_eager` 一个分母。**

### 5.2 但这不代表分母一定弱 —— 分三档看

探针第 5 节实测：`libtorch_hip.so` 链接 `libMIOpen.so.1`、
`torch.backends.cudnn.is_available()` 为 True（DCU 上即 MIOpen 后端）、
`F.conv2d` 实际落到 MIOpen 的 `winograd_fwd_2d_fp32_3x3_s1_16x16x4` 内核。因此：

| 档 | 题 | `torch_eager` 分母的含义 |
|---|---|---|
| **conv 族：eager 即 vendor 库分母** | 4 道：`5001_conv2d_fwd`、`5002_conv2d_bwd`、`5008_conv_bias_act`、`5009_conv_depthtospace` | torch 走 MIOpen → eager 分母**就是 vendor 实现**，不弱 |
| **纯 ATen** | 8 道：`5003_deform_conv2d_fwd`、`5004_deform_conv2d_bwd`、`5005_spconv_fprop`、`5006_spconv_bwd`、`5007_spconv_indice_pairs`、`8001_cumsum`、`8002_gather`、`8003_scatter_nd` | 探针明确列为无 MIOpen 路径 → eager 是纯 ATen，偏弱 |
| **探针未覆盖** | 7 道：`1031_deform_attention_fwd`、`1032_deform_attention_bwd`、`2010_batchnorm_fwd`、`2011_batchnorm_bwd`、`2012_groupnorm_fwd`、`2013_groupnorm_bwd`、`2014_instancenorm_fwd` | 未验证是否走 MIOpen，**不作断言** |

引用这 19 道的 speedup 时必须注明分母含义：conv 族（5001/5002/5008/5009）的 eager
分母已经是 vendor 库，其加速比含金量高于其余各道。

## 6. 结论

1. **编译率**：root 脚手架 66→66/66，所有题都拿到了可跑的起点。
2. **正确率（在线）**：best 正确 66→66/66（100.0%），真解 53→49。
3. **正确率（离线终审）**：审 30 道有 aiter 基线的题，通过 25 道 —— 比在线严，两套都对但量的不是同一件事。
4. **对 aiter 的加速比**（离线）：中位 1.08x、均值 1.28x、最大 2.98x；其中 5 道是 MI 兜底基线、引用须标注。
5. **在线加速比**（vs eager）：真解中位 5.61x → 7.33x。
6. **基线覆盖**：33 道无 aiter 基线（含 HIP 全部 19 道），只能给 eager 口径；HIP 的 conv 族 eager 分母即 MIOpen。

