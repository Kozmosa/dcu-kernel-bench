# 66 道生成评测性能统计（2026-10-17）

五个 harness 缺陷修复前后的全量对照。数据源：

- 修复前：`runs/dcukb_final`（66 道，含此前 HIP20 重跑结果）
- 修复后：`runs/dcukb_full_fixed`（66 道，16 片并行，10:08–11:53）

## 0. 口径（先读这段，避免误读数字）

| 概念 | 定义 | 数据来源 |
|---|---|---|
| **编译率** | 脚手架/候选解能通过编译的比例 | run.json 的 `compile_ok` |
| **正确率** | 通过在线正确性比对（allclose 1e-4/trial）的比例 | run.json 的 `correct` |
| **真解** | best 节点**不是 root**（即 Agent 真的改写了代码）且 correct | `best_node_id` + `correct` |
| **加速比** | 生成实现 runtime 的倒数 ÷ 参考实现 runtime 的倒数 | run.json 的 `speedup` |

**关键口径**：run.json 里 `speedup` 的分母是 **`torch_eager`**（在线评测器拿参考 `Model`
当分母），**不是 aiter**。本轮 66 份 run.json 的 `primary_reference` 全部为 `torch_eager`。

aiter 基线在 `benchmark/private/<task>/baseline.json`，按 perf case 分 impl 存实测 us。
**「生成实现 vs aiter」这个数本报告不提供** —— 它需要离线 `audit_batch.py` 把生成产物
送进终审、在真实 perf case 上计时；在线 runtime 对应的是 `get_inputs()` 的固定 shape，
与 perf case 不一定同形状，直接相除不成立。第 4 节给出 aiter 基线本身的数据与缺口。

## 1. 编译率

| 指标 | 修复前 | 修复后 |
|---|---|---|
| root 脚手架编译通过（= 框架给出的起点可用） | **59/66** | **66/66** |
| best 节点编译通过 | 60/66 | 66/66 |
| compile 阶段失败次数（合计，含 Agent 自己写坏） | 319 | 210 |
| runtime 阶段失败次数（合计） | 71 | 46 |

root 脚手架编译率是**衡量 harness 质量**的指标：修复前 7 道题因脚手架多行签名截断
直接 SyntaxError，Agent 连起点都没有。修复后 66/66。
compile 阶段失败次数不归零是正常的 —— 那是 Agent 尝试过程里自己写坏的补丁，
属于生成侧信号，不是 harness 缺陷。

## 2. 正确率

| 指标 | 修复前 | 修复后 |
|---|---|---|
| best 正确（在线判据） | 57/66（86.4%） | **66/66（100.0%）** |
| └ 非 root 真解 | 47（71.2%） | **53（80.3%）** |
| └ 仅 root（Agent 未改写） | 10 | 13 |
| 有 speedup 可比 | 57/66 | 66/66 |

### 按 backend

| backend | 题数 | best 正确（前→后） | 真解（前→后） |
|---|---|---|---|
| HIP（backend=cuda） | 20 | 11→20 | 10→15 |
| Triton | 46 | 46→46 | 37→38 |

## 3. 加速比（分母 = `torch_eager`）

统计口径：只看**真解**（best 非 root 且 correct）的 speedup。

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 样本数 n | 47 | 53 |
| 中位数 | 6.50x | 5.61x |
| 均值 | 68.35x | 72.26x |
| P25 | 2.66x | 3.37x |
| P75 | 28.65x | 37.16x |
| 最大 | 1212.39x | 858.02x |
| 最小 | 0.81x | 0.82x |

### speedup 分布（真解）

| 区间 | 修复前 | 修复后 |
|---|---|---|
| 0–1x | 1 | 2 |
| 1–2x | 7 | 9 |
| 2–5x | 11 | 12 |
| 5–10x | 9 | 9 |
| 10–100x | 14 | 14 |
| ≥100x | 5 | 7 |

### 按题号段（修复后，真解）

| 号段 | 真解数 | speedup 中位 | 最大 |
|---|---|---|---|
| 1xxx | 17 | 12.06x | 513.08x |
| 2xxx | 13 | 4.87x | 57.31x |
| 3xxx | 6 | 40.08x | 858.02x |
| 4xxx | 7 | 1.78x | 37.16x |
| 5xxx | 6 | 77.98x | 416.07x |
| 6xxx | 1 | 5.13x | 5.13x |
| 7xxx | 1 | 1.00x | 1.00x |
| 8xxx | 2 | 2.67x | 3.42x |

### Top 10 / Bottom 5（修复后）

| # | 题 | speedup |
|---|---|---|
| 1 | `3003_moe_op` | **858.02x** |
| 2 | `1031_deform_attention_fwd` | **513.08x** |
| 3 | `1016_lean_atten_paged` | **436.91x** |
| 4 | `1032_deform_attention_bwd` | **435.53x** |
| 5 | `5004_deform_conv2d_bwd` | **416.07x** |
| 6 | `5007_spconv_indice_pairs` | **356.89x** |
| 7 | `5006_spconv_bwd` | **144.60x** |
| 8 | `1019_mha_onekernel_bwd` | **85.29x** |
| 9 | `1030_unified_attention` | **83.06x** |
| 10 | `3007_moe_op_silu_fused` | **64.75x** |
| 49 | `4006_gemm_a16w16` | 1.16x |
| 50 | `1020_mla_decode_rope` | 1.01x |
| 51 | `7001_ctc_loss` | 1.00x |
| 52 | `4004_batched_gemm_bf16` | 1.00x |
| 53 | `1018_mha_fused_bwd` | 0.82x |

## 4. aiter 基线

`benchmark/private` 共 68 个 `baseline.json`：**33 个含 aiter 基线**，33 个仅 `torch_eager`，2 个 placeholder。

### aiter 本身相对 torch_eager 的加速比（同机实测，按 perf case 取中位）

| 题 | aiter 相对 eager | (min–max) | perf case 数 |
|---|---|---|---|
| `1014_hstu_attention` | **42.09x** | 28.16–56.02x | 2 |
| `1002_paged_attention` | **38.67x** | 16.57–54.65x | 3 |
| `1016_lean_atten_paged` | **28.25x** | 7.37–54.33x | 3 |
| `4005_fused_mxfp4_quant` | **20.83x** | 18.98–21.35x | 3 |
| `1021_pa_decode` | **20.22x** | 13.59–21.98x | 3 |
| `3003_moe_op` | **9.80x** | 9.60–32.52x | 3 |
| `1024_prefill_attention` | **8.04x** | 1.33–13.20x | 3 |
| `2005_moe_activation` | **7.68x** | 1.38–10.44x | 3 |
| `2006_norm` | **4.69x** | 3.74–4.74x | 3 |
| `2002_add_swiglu` | **3.96x** | 3.54–3.97x | 3 |
| `2009_softmax` | **3.88x** | 2.72–4.24x | 3 |
| `2008_rope` | **3.62x** | 3.02–8.90x | 3 |
| `1017_mha` | **3.18x** | 1.32–3.59x | 3 |
| `2003_fused_mul_add` | **2.67x** | 0.83–2.74x | 3 |
| `4017_group_quant_int8` | **2.46x** | 2.38–2.84x | 3 |
| `4008_gemm_a16w4` | **2.42x** | 1.10–29.93x | 3 |
| `2007_rmsnorm` | **2.21x** | 1.26–2.25x | 3 |
| `1010_chunked_pa_prefill` | **2.04x** | 1.66–2.26x | 3 |
| `1022_pa_prefill` | **1.90x** | 1.64–2.48x | 3 |
| `4004_batched_gemm_bf16` | **1.70x** | 0.20–1.88x | 3 |
| `4018_quant` | **1.69x** | 1.16–1.70x | 3 |
| `1020_mla_decode_rope` | **1.68x** | 0.24–3.19x | 3 |
| `1026_sage_attention_qk_int8_per_block` | **1.46x** | 1.37–1.52x | 3 |
| `1012_flash_attention_forward` | **1.40x** | 1.25–5.29x | 3 |
| `3010_topk` | **1.24x** | 0.91–1.76x | 3 |
| `4016_gemm_w8a8` | **1.23x** | 1.17–2.45x | 3 |
| `2004_fused_qk_concat` | **1.22x** | 0.89–1.55x | 2 |
| `3004_moe_op_e2e` | **1.21x** | 0.73–2.05x | 3 |
| `1015_lean_atten` | **1.16x** | 1.11–2.39x | 3 |
| `3009_routing` | **1.07x** | 0.87–2.20x | 3 |
| `4001_batched_gemm_a8w8` | **0.37x** | 0.29–2.20x | 3 |
| `3002_moe_align_block_size` | **0.19x** | 0.05–0.75x | 3 |
| `4006_gemm_a16w16` | **0.11x** | 0.11–0.18x | 3 |

> 这张表说明的是**官方 aiter 实现比 PyTorch eager 快多少**，可以作为「生成实现要达到
> 什么水平才算追上官方」的参照系。它不是「生成实现 vs aiter」。
>
> **已知偏差（必须一起读）**：`4001` / `4004` / `4006` 与 `4008` 的部分 case，其 aiter
> 基线落在 **MI 系列 fallback tuner 配置**上（未针对 DCU 调优，见 AGENTS.md），aiter
> 数值本身偏慢，所以「aiter 相对 eager」会偏低甚至 <1x（`4006` 0.11x、`3002` 0.19x、
> `4001` 0.37x）。这时**拿 aiter 当分母会把生成实现的加速比算得偏高**，引用时须标注。

### 缺口：生成实现 vs aiter

要得到「生成实现相对 aiter 的加速比」，需要跑离线终审：

```bash
python3 benchmark/evaluator/audit_batch.py --help    # 把生成产物批量送离线终审
```

它做的事：静态审计 → 隐藏 case 按 task.yaml 容差判定 → **在真实 perf case 上计时**，
分母用 `private/<task>/baseline.json` 里记录的 aiter 用时。
这一步本次**没有执行**，所以本报告不含该数字。

## 5. 结论

1. **编译率**：root 脚手架 59→66/66，即所有题都拿到了可跑的起点。
2. **正确率**：best 正确 57→66/66（100.0%），其中真解 47→53。
3. **加速比**（vs eager）：真解中位 6.50x → 5.61x，最大 1212.39x → 858.02x。
4. **aiter 基线**：33 道题有官方 aiter 实测基线（第 4 节表），但生成实现 vs aiter 需要
   离线 `audit_batch.py`，本次未跑。

