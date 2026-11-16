# HIP / Triton 双轨对照

- 运行 runs/dualtrack_1027_0838（2026-10-27，132 次 = 66 题 x 2 轨，16 分片并发）；模型 gpt-6-astra、main profile，分母 torch_eager。
- 每道题按同一份语义在两条轨上各跑一次；题面只改「目标硬件 + 实现语言 + 锚点」那几句（`benchmark/tools/make_dualtrack.py` 生成），语义/IO 契约/边界/禁用列表逐字相同。
- 口径与 `metrics.md` 一致：候选排除 root；分母 `torch_eager`；best = 所有 correct 候选里的最大 speedup；尝试序号 root=0。**每格只跑了一次，逐题差异含采样噪声。**

## 两轨总表

| 指标 | HIP 轨 | Triton 轨 | 谁更好 |
|---|---|---|---|
| Compile Rate | **66/66** | **66/66** | 持平 |
| Correct Rate | **60/66** | **59/66** | HIP |
| Success@1 | **38/66** | **27/66** | HIP |
| Success@5 | **60/66** | **57/66** | HIP |
| Fast1@all | **50/66** | **56/66** | Triton |
| Geomean Speedup | **7.74x** | **15.61x** | Triton |
| Median Speedup | **5.75x** | **10.51x** | Triton |
| Mean Speedup | **57.63x** | **99.46x** | Triton |
| Best Speedup | **598.46x** | **1300.29x** | Triton |
| First Correct Median | **1.00** | **2.00** | HIP |
| Refine Gain Median | **1.54x** | **1.05x** | HIP |

两轨都写对 **59** 道；只 Triton 对 0 道；只 HIP 对 1 道（`1016_lean_atten_paged`）；都不对 6 道（`1026_sage_attention_qk_int8_per_block`、`1027_sage_attention_qk_int8_per_block_causal`、`2013_groupnorm_bwd`、`4008_gemm_a16w4`、`4009_gemm_a8w8`、`4017_group_quant_int8`）。

两轨都写对的 59 道里，**HIP 更快 11 道、Triton 更快 48 道**。

## 逐题对照

| 题 | HIP 正确 | HIP best | Triton 正确 | Triton best | Triton/HIP |
|---|---|---|---|---|---|
| `1032_deform_attention_bwd` | ✓ | 598.46x | ✓ | 392.52x | 0.66 |
| `7001_ctc_loss` | ✓ | 497.42x | ✓ | 1191.92x | 2.40 |
| `1031_deform_attention_fwd` | ✓ | 451.37x | ✓ | 1300.29x | 2.88 |
| `5004_deform_conv2d_bwd` | ✓ | 426.29x | ✓ | 585.17x | 1.37 |
| `5007_spconv_indice_pairs` | ✓ | 379.77x | ✓ | 117.70x | 0.31 |
| `5006_spconv_bwd` | ✓ | 206.63x | ✓ | 224.14x | 1.08 |
| `2001_activation` | ✓ | 95.41x | ✓ | 175.21x | 1.84 |
| `4005_fused_mxfp4_quant` | ✓ | 94.07x | ✓ | 125.12x | 1.33 |
| `1030_unified_attention` | ✓ | 61.24x | ✓ | 162.39x | 2.65 |
| `4018_quant` | ✓ | 59.67x | ✓ | 157.58x | 2.64 |
| `1016_lean_atten_paged` | ✓ | 59.55x | ✗ | - | - |
| `3003_moe_op` | ✓ | 58.22x | ✓ | 306.14x | 5.26 |
| `3006_moe_op_mxfp4` | ✓ | 56.77x | ✓ | 68.64x | 1.21 |
| `1022_pa_prefill` | ✓ | 55.68x | ✓ | 90.57x | 1.63 |
| `3007_moe_op_silu_fused` | ✓ | 52.34x | ✓ | 85.43x | 1.63 |
| `5003_deform_conv2d_fwd` | ✓ | 42.64x | ✓ | 56.94x | 1.34 |
| `3002_moe_align_block_size` | ✓ | 35.19x | ✓ | 18.03x | 0.51 |
| `1019_mha_onekernel_bwd` | ✓ | 28.07x | ✓ | 64.30x | 2.29 |
| `2004_fused_qk_concat` | ✓ | 25.81x | ✓ | 240.63x | 9.32 |
| `1010_chunked_pa_prefill` | ✓ | 25.63x | ✓ | 40.56x | 1.58 |
| `5005_spconv_fprop` | ✓ | 12.21x | ✓ | 7.94x | 0.65 |
| `1002_paged_attention` | ✓ | 12.03x | ✓ | 54.86x | 4.56 |
| `3004_moe_op_e2e` | ✓ | 11.16x | ✓ | 24.89x | 2.23 |
| `1014_hstu_attention` | ✓ | 9.85x | ✓ | 51.06x | 5.19 |
| `2012_groupnorm_fwd` | ✓ | 9.26x | ✓ | 3.69x | 0.40 |
| `6001_fused_adamw` | ✓ | 6.56x | ✓ | 2.60x | 0.40 |
| `2011_batchnorm_bwd` | ✓ | 6.47x | ✓ | 5.87x | 0.91 |
| `2008_rope` | ✓ | 6.31x | ✓ | 32.18x | 5.10 |
| `2002_add_swiglu` | ✓ | 6.13x | ✓ | 6.38x | 1.04 |
| `2006_norm` | ✓ | 5.79x | ✓ | 9.76x | 1.69 |
| `1017_mha` | ✓ | 5.71x | ✓ | 8.01x | 1.40 |
| `2014_instancenorm_fwd` | ✓ | 5.24x | ✓ | 5.53x | 1.06 |
| `1011_extend_attention` | ✓ | 4.53x | ✓ | 11.15x | 2.46 |
| `2005_moe_activation` | ✓ | 4.29x | ✓ | 4.65x | 1.09 |
| `1021_pa_decode` | ✓ | 4.27x | ✓ | 10.51x | 2.46 |
| `3010_topk` | ✓ | 3.10x | ✓ | 3.69x | 1.19 |
| `1024_prefill_attention` | ✓ | 3.01x | ✓ | 32.31x | 10.73 |
| `4001_batched_gemm_a8w8` | ✓ | 2.46x | ✓ | 10.00x | 4.07 |
| `1012_flash_attention_forward` | ✓ | 2.46x | ✓ | 16.23x | 6.61 |
| `1015_lean_atten` | ✓ | 2.44x | ✓ | 0.84x | 0.34 |
| `2007_rmsnorm` | ✓ | 2.41x | ✓ | 3.85x | 1.60 |
| `5009_conv_depthtospace` | ✓ | 2.02x | ✓ | 1.52x | 0.75 |
| `8002_gather` | ✓ | 1.89x | ✓ | 1.90x | 1.01 |
| `2009_softmax` | ✓ | 1.86x | ✓ | 4.27x | 2.29 |
| `8003_scatter_nd` | ✓ | 1.84x | ✓ | 1.88x | 1.02 |
| `1029_triton_decode_attention` | ✓ | 1.82x | ✓ | 39.96x | 22.00 |
| `2010_batchnorm_fwd` | ✓ | 1.80x | ✓ | 63.56x | 35.26 |
| `8001_cumsum` | ✓ | 1.77x | ✓ | 1.69x | 0.95 |
| `2003_fused_mul_add` | ✓ | 1.69x | ✓ | 1.82x | 1.08 |
| `1020_mla_decode_rope` | ✓ | 1.39x | ✓ | 0.94x | 0.67 |
| `3009_routing` | ✓ | 0.97x | ✓ | 1.79x | 1.85 |
| `4010_gemm_a8w8_blockscale` | ✓ | 0.88x | ✓ | 2.97x | 3.39 |
| `4004_batched_gemm_bf16` | ✓ | 0.75x | ✓ | 2.11x | 2.81 |
| `5008_conv_bias_act` | ✓ | 0.73x | ✓ | 1.30x | 1.78 |
| `5002_conv2d_bwd` | ✓ | 0.64x | ✓ | 2.94x | 4.60 |
| `4016_gemm_w8a8` | ✓ | 0.63x | ✓ | 17.50x | 27.67 |
| `5001_conv2d_fwd` | ✓ | 0.46x | ✓ | 1.00x | 2.19 |
| `1018_mha_fused_bwd` | ✓ | 0.30x | ✓ | 10.13x | 33.30 |
| `4006_gemm_a16w16` | ✓ | 0.25x | ✓ | 1.22x | 4.93 |
| `4007_gemm_a16w16_atomic` | ✓ | 0.07x | ✓ | 0.31x | 4.56 |
| `1026_sage_attention_qk_int8_per_block` | ✗ | - | ✗ | - | - |
| `1027_sage_attention_qk_int8_per_block_causal` | ✗ | - | ✗ | - | - |
| `2013_groupnorm_bwd` | ✗ | - | ✗ | - | - |
| `4008_gemm_a16w4` | ✗ | - | ✗ | - | - |
| `4009_gemm_a8w8` | ✗ | - | ✗ | - | - |
| `4017_group_quant_int8` | ✗ | - | ✗ | - | - |

