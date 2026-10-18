# DCU 算子生成评测结果

**生成模型**：`gpt-6-astra`（经 hejuapi 的 OpenAI 兼容端点，`chat_completions` 协议）
**环境**：海光 DCU（BW / gfx936），DTK 26.04，torch 2.9.0+das.opt1.dtk2604
**生成框架**：PyramidKernel（profile = `main`）
**评测规模**：66 道算子题（Triton 46 道 / HIP 20 道）
**完成情况**：66 / 66 全部跑完，无中断、无崩溃

---

## 1. 编译率

| 指标 | 结果 |
|---|---|
| 题面脚手架编译通过（框架给出的起点可用） | **66 / 66（100%）** |
| 最优解节点编译通过 | **66 / 66（100%）** |

## 2. 正确率

### 在线判定（KernelBench 口径：单一 shape、`allclose(atol=rtol=1e-4)`）

| 指标 | 结果 |
|---|---|
| 产出正确实现 | **66 / 66（100%）** |
| 其中「真实解」（模型自行改写、非直接回退参考实现） | **53 / 66（80.3%）** |
| 其中「仅跑通参考实现」（模型未改写） | 13 / 66 |

| 算子轨 | 题数 | 正确 | 真实解 |
|---|---|---|---|
| HIP（`backend=cuda`，hipcc 编译） | 20 | **20（100%）** | **15（75%）** |
| Triton | 46 | **46（100%）** | **38（83%）** |

### 离线终审（静态审计 + 隐藏 case + 每题分档容差）

对 31 道有官方库基线的题做了离线终审：**通过 21 / 31（68%）**。

未通过的 10 道**不是同一类原因**，分两种，性质完全不同：

| 未通过原因 | 题数 | 说明 |
|---|---|---|
| **静态审计判负**（未进入正确性环节） | **6** | 实现里直接调用被禁的 ATen 算子完成核心计算，违反「禁止绕过」铁律 |
| **正确性不达标**（隐藏 case 对不上） | **4** | 静态审计通过，但边界 shape 处理不到位 |

#### A. 静态审计判负的 6 道

| 题 | 命中的禁用算子 |
|---|---|
| `1020_mla_decode_rope` | `torch.matmul`、`torch.softmax` |
| `1022_pa_prefill` | `torch.einsum`、`torch.softmax` |
| `1026_sage_attention_qk_int8_per_block` | `torch.matmul`、`torch.softmax` |
| `3009_routing` | `torch.matmul`、`torch.sigmoid` |
| `4004_batched_gemm_bf16` | `torch.matmul` |
| `4008_gemm_a16w4` | `torch.matmul` |

这 6 道在**在线**被判为「正确」——在线评测不执行禁用列表（题面的 forbidden 在线上只是软约束，
硬审计在离线）。所以它们不是「性能不好」，而是**实现方式不合规**。

#### B. 正确性不达标的 4 道

| 题 | 失败 case | 失败特征 |
|---|---|---|
| `2007_rmsnorm` | 2 / 6 | 非 2 次幂（`m873_n1245_f16`）、超长行（`m4_n65536_bf16`） |
| `2008_rope` | 1 / 6 | 非 2 次幂 + 部分旋转 + NEOX 布局（`nonpow2_partial_neox_bf16`） |
| `3002_moe_align_block_size` | 1 / 6 | 大规模专家数（`official_large_e224`） |
| `4018_quant` | 3 / 6 | fp8 动态量化：大宽表、零 token 行、bf16 输入 |

失败 case 全部是 `hidden_` 前缀的隐藏 case，形态为**非 2 次幂、超长/超宽、极大规模、
fp8 动态量化、零行**等边界。在线只跑 `get_inputs()` 的**单一固定 shape**，这些边界不会出现 ——
所以在线全对、离线翻车。这 4 道才是真正能归因于「离线判据更严」的。

> 两套判据不互相否定：在线回答「能不能跑通并对上参考」，离线回答「换一批 shape/边界还对不对、
> 以及实现是否合规」。引用时必须说明用的是哪一套。

## 3. 加速比

### 3.1 对 PyTorch 参考实现（`torch_eager`）

样本：53 道真实解。

| 中位数 | 均值 | P25 | P75 | 最大 | 最小 |
|---|---|---|---|---|---|
| **5.61x** | 72.26x | 3.37x | 37.16x | **858.02x** | 0.82x |

分布：

| <1x | 1–2x | 2–5x | 5–10x | 10–100x | ≥100x |
|---|---|---|---|---|---|
| 2 | 9 | 12 | 9 | 14 | **7** |

最快的 5 道：

| 题 | 加速比 |
|---|---|
| `3003_moe_op` | 858.02x |
| `1031_deform_attention_fwd` | 513.08x |
| `1016_lean_atten_paged` | 436.91x |
| `1032_deform_attention_bwd` | 435.53x |
| `5004_deform_conv2d_bwd` | 416.07x |

### 3.2 对官方库基线（双基线，离线终审）

31 道有官方 `aiter` 基线的题中，21 道跑出结果。同一批题、同一口径下同时与两种基线对比：

| 分母 | 中位数 | 均值 | 最大 | 达到或超过基线 |
|---|---|---|---|---|
| **官方 aiter 实现** | **1.21x** | 1.58x | 10.49x | **11 / 21** |
| PyTorch eager | 3.68x | 4.82x | 21.73x | — |

> 对 aiter 的中位数（1.21x）明显低于对 eager 的中位数（3.68x），因为 aiter 本身就是
> 高度调优的官方实现 —— 生成代码要超过它，比超过 PyTorch 难得多。

## 4. 基线覆盖情况

| 类别 | 题数 | 说明 |
|---|---|---|
| 有官方 aiter 基线 | 33 | 可与官方实现对比 |
| 仅有 `torch_eager` 基线 | 33 | 含 **HIP 全部 20 道** |

HIP 那 20 道的基线状况与 hipDNN 基线探针的结论一致：hipDNN 在当前环境的 vendor 引擎插件
（`libmiopen_legacy_plugin.so`）上不可用、官方样例要求 DTK ≥ 26.10，aiter 侧也无对等算子，
因此只有 `torch_eager` 一个分母。其中 **conv 族 4 道（5001/5002/5008/5009）的
`torch_eager` 实际就走 MIOpen**，即已经是 vendor 库分母，其加速比含金量高于其余各道。

## 5. 引用这些数字时的注意事项

1. **加速比的分母必须说明**：在线数字的分母是 `torch_eager`，不是 aiter。
2. **「vs aiter」必须逐题看**：6 道题的 aiter 基线是 MI 系列 fallback tuner 配置
   （未针对 DCU 调优），另有 3 道 aiter 配置本身比 PyTorch 还慢 —— 详见
   **附录附 1**，不能只看聚合中位数。
3. **HIP 那 20 道的分母含义分三档**（conv 族即 MIOpen / 纯 ATen / 未验证）——
   详见 **附录附 2**。
4. **2 道题没有离线终审结果**（`1016_lean_atten_paged`、`2004_fused_qk_concat`，
   离线工具侧崩溃），它们的在线结果是 436.91x 与 26.68x。

---

## 附：基线补充说明（读「vs aiter」这些数字前请先看这节）

### 附 1. 为什么有些 aiter 基线「不可靠」

aiter 是官方实现，但**它的实测数值不一定等于 aiter 在 DCU 上的最好成绩**。本轮实际遇到三种情况：

**(a) 掉进 MI 系列 fallback tuner 配置 —— 6 道**

`1014_hstu_attention`、`1017_mha`、`4001_batched_gemm_a8w8`、`4004_batched_gemm_bf16`、
`4006_gemm_a16w16`、`4008_gemm_a16w4`（`4008` 仅部分 case）。

aiter 的调优表里没有这些算子在 DCU 上的专用配置，回落到了 MI 系列 GPU 的 tuner 配置。
后果：**aiter 数值本身偏离其应有水平，方向因题而异** —— 有的题因此把生成实现的
「vs aiter」抬得虚高，有的题则相反。所以这 6 道在离线终审里被单独打了
`aiter_is_mi_fallback` 标记，引用其「vs aiter」必须同时标注。

**(b) aiter 配置本身比 PyTorch 还慢 —— 3 道（实测，aiter 相对 eager 的加速比）**

| 题 | aiter 相对 eager | 含义 |
|---|---|---|
| `4006_gemm_a16w16` | **0.11x**（0.11–0.18） | aiter 比 PyTorch 慢约 9 倍 |
| `3002_moe_align_block_size` | **0.19x**（0.05–0.75） | 慢约 5 倍 |
| `4001_batched_gemm_a8w8` | **0.37x**（0.29–2.20） | 慢约 3 倍 |

这类分母上算出的「vs aiter」会异常高（如 `4006` 的 10.49x），**不能当作「超过官方实现」的证据**。

作为对照，aiter 强的一侧确实很强：`1014` 42.09x、`1002` 38.67x、`1016` 28.25x、
`4005` 20.83x（aiter 相对 eager，同机实测）。**同是"aiter 分母"，强弱可差三个数量级**，
所以「vs aiter」这个指标必须逐题看，不能只看聚合中位数。

**(c) 根本没有 aiter 基线 —— 33 道**

原因有两类：aiter 里没有可对位的实现（主要是 hipDNN 样例覆盖的算子），
或采集不到（如 `1019_mha_onekernel_bwd`，DTK 版 aiter 的 onekernel bwd 报
`PassManager::run failed`，shared memory 81920 超出 DCU 上限 65536，非配置问题）。
这 33 道只能给 `torch_eager` 口径。

### 附 2. HIP 那 20 道的 `torch_eager` 就是 MIOpen（不是"弱基线"）

这 20 道源自 hipDNN 样例，本应补 hipDNN 分母，但 hipDNN 在当前环境不可用（见第 4 节）。
不过**它们的 `torch_eager` 分母并不弱** —— 因为 torch 在 DCU 上对 conv 族就是走 MIOpen：

```
ldd libtorch_hip.so                          -> libMIOpen.so.1            ✓
torch.backends.cudnn.is_available()          = True（DCU 上即 MIOpen 后端） ✓
profile F.conv2d(16,16,16,16 / 16,16,3,3)    -> winograd_fwd_2d_fp32_3x3_s1_16x16x4
                                                 （MIOpen 的 Winograd 内核）  ✓
```

所以对 conv 族，**`torch_eager` 就是 vendor 库（MIOpen）实现**，其加速比含金量等同于
"对官方库的加速比"。20 道按分母含义分三档：

| 档 | 题 | `torch_eager` 分母的含义 |
|---|---|---|
| **conv 族，即 vendor 库分母** | 4 道：`5001_conv2d_fwd`、`5002_conv2d_bwd`、`5008_conv_bias_act`、`5009_conv_depthtospace` | 走 MIOpen → **就是 vendor 实现，不弱** |
| **纯 ATen，偏弱** | 9 道：`5003`、`5004`（deform conv）、`5005`、`5006`、`5007`（spconv）、`6001`（adamw）、`8001`（cumsum）、`8002`（gather）、`8003`（scatter_nd） | torch 无 MIOpen 路径 → 纯 ATen，偏弱 |
| **未验证，不作断言** | 7 道：`1031`、`1032`（deform attention）、`2010`–`2014`（batchnorm/groupnorm/instancenorm） | 未探针确认是否走 MIOpen |

**引用这 20 道的加速比时必须注明属于哪一档**：conv 族那 4 道的数字含金量最高
（分母已是 vendor 库），纯 ATen 那 9 道会系统性偏高。

> 依据：`benchmark/tools/hipdnn_baseline_probe.md`（2026-10-15 的实测探针）。

---

**数据与复现**：详细统计见 `perf_report.md`；统计数据由 `stats_full66.py` 从
66 份 run.json 与 `offline_audit.json` 生成，可一条命令重算。
