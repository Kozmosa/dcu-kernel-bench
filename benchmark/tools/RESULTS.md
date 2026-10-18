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
2. **「vs aiter」必须逐题看，不能只看聚合中位数**：6 道题的 aiter 基线是 MI 系列
   fallback tuner 配置（未针对 DCU 调优），另有 3 道（`4006` / `3002` / `4001`）的
   aiter 配置本身比 PyTorch 还慢；这些题的「vs aiter」会虚高或失真。逐题对照表见
   `perf_report.md` 第 4 节。
3. **HIP 那 20 道的分母含义分三档**（conv 族即 MIOpen / 纯 ATen / 未验证）——
   见 `perf_report.md` 第 5.2 节。
4. **2 道题没有离线终审结果**（`1016_lean_atten_paged`、`2004_fused_qk_concat`，
   离线工具侧崩溃），它们的在线结果是 436.91x 与 26.68x。

---

## 附 1：aiter 基线为什么没覆盖剩下的题

66 道里 33 道有 aiter 基线。剩下 33 道中 **20 道是 hipDNN 来源题**（其基线口径见第 4 节），
按要求不计入本节；因此这里只说明**其余 13 道**：

| 原因 | 题数 | 题 |
|---|---|---|
| aiter 本来就没有对位实现 | 2 | `4007_gemm_a16w16_atomic`（缺 tuner 配置 `BW200-GEMM-A16W16-ATOMIC.json`）、`7001_ctc_loss`（aiter 无此算子） |
| 适配器已写，采集时 Triton 编译失败（`PassManager::run failed`） | 5 | `1011`、`1030`、`3007`、`4009`、`4010` |
| 超出 DCU 共享内存上限（81920 > 65536） | 1 | `1019_mha_onekernel_bwd` |
| DCU 版 triton 3.3.0 缺 aiter 依赖的 API | 3 | `1027`（`triton.utils`）、`1029`（`tl.matrix_load`）、`3006`（`tl.constexpr_function`） |
| 采集失败、当时未记录原因 | 2 | `1018_mha_fused_bwd`、`2001_activation` |

**结论：这 13 道不是"没测"，而是 aiter 侧跑不起来。** 11 道的适配器都写好了，卡在 Triton
编译失败 / 共享内存超限 / triton API 缺失；另 2 道 aiter 本身没有对应算子。

> 后记：后来加入的 `benchmark/tools/triton_compat.py` 补齐了 triton 3.3.0 缺的 4 个 API，
> 并把 `1010` / `1012` / `1022` / `1026` 等一批题的 aiter 基线采了出来；**但这 13 道没有重采**，
> 所以它们目前仍然只有 `torch_eager` 口径。

（HIP 那 20 道的分母说明 —— 尤其 conv 族的 `torch_eager` 即 MIOpen —— 见第 4 节，
详细三档表在 `perf_report.md` 第 5.2 节。）

## 附 2：对原框架的改动

本轮结果是在**改动过的**框架上跑出来的：5 处框架代码修复 + 一批 DCU 适配。
两类都**不改题目数学语义、不改评分判据**。

### A. 框架代码改动（5 处）

**PyramidKernel**（3 个提交，仅 1 个文件 `pyramidkernel/core/loader.py`，+201 / −18）

| 提交 | 改动 | 不改的后果 |
|---|---|---|
| `ea649d6` | CUDA 脚手架的 `extra_cuda_cflags` 去掉 `--use_fast_math`，改为按平台条件化 | DCU 的 `dcc` 不认该 nvcc 专有开关，全部 HIP 题编译必失败 |
| `77b7453` | 修多行签名截断与语句块缩进错乱（`ast.get_source_segment` 只把首行切到 `col_offset`） | 7 道题的脚手架从根节点起就是非法 Python |
| `9f9549f` | 脚手架补齐 `Model` 的辅助成员方法与模块级定义（求引用闭包、保留装饰器） | 3 道 root 报 `AttributeError: 'ModelNew' object has no attribute ...`、2 道报 `NameError` |

**KernelBench**（无独立 git 仓库，以补丁形式保存在 `benchmark/tools/kernelbench_*.patch`）

| 文件 | 改动 | 不改的后果 |
|---|---|---|
| `src/kernelbench/eval.py` | 正确性检查的 `no_grad` → `enable_grad` | 3 道反向题的参考实现用 `torch.autograd.grad`，被 `no_grad` 冲掉后 root 直接报错 |
| `src/kernelbench/timing.py` | 参考计时路径的 `no_grad` → `enable_grad` | 4 道反向题测不出参考时间 → `speedup` 恒为 None |

**影响**：修复前 66 道里 **13 道的 root 基线不合法**（按各自最初的失败签名统计：7 脚手架语法错
+ 3 autograd + 1 缺辅助方法 + 2 缺模块级定义；归因有重叠 —— `5004`/`5009` 是在修好脚手架语法错
之后才暴露出缺辅助方法/模块级定义），4 道反向题拿不到加速比。修复后 **root 基线 53/66 → 66/66**、
`best 正确` 57/66 → **66/66**、`有 speedup` 57/66 → **66/66**。

> 两个 `enable_grad` 改动的安全性：无 `requires_grad` 张量时 PyTorch 不建图，对 62 道前向题
> 零开销；反绕过仍由离线 `static_audit` 负责（线上 forbidden 本就是软约束）。
> 根因与验证细节见 `harness_fixes.md`。

### B. DCU 适配（题目与环境侧）

原框架是按 CUDA 生态写的，迁到海光 DCU 需要在题目与环境层面做这些适配：

| 项 | 改动内容 | 为什么 DCU 必须改 |
|---|---|---|
| **20 道题从 Triton track 迁到 HIP track** | `impl_lang: hip`、manifest 改 `backend: cuda`、题面语言整体改写（106 处 Triton 提法 → HIP）。迁移后 **Triton 46 / HIP 20** | 题面必须与实际编译路径一致，否则 Agent 按 Triton 写而框架走 hipcc |
| **题面注入 DCU 编译约束**（改提示） | 20 道 HIP 题补上：`目标硬件：海光 DCU（gfx936/gfx938），实现语言 HIP（经 hipcc 编译；在 CUDA_CU_SRC / CUDA_CPP_SRC 锚点区编写，设备侧必须 include <hip/hip_runtime.h>，本机无 <cuda_runtime.h>）` | 本机只有 `<hip/hip_runtime.h>`；框架默认模板 include 的恰是 `<cuda_runtime.h>`，题面不说清 Agent 必然写错头文件 |
| **task.yaml 锚点说明补 HIP 路径** | 20 道补全 `user_helpers / cuda_cpp / cuda_cu / init_body / forward_stmt_N` | HIP 路径比 Triton 路径多 `user_helpers` 锚点，不说明则 Agent 找不到落笔位置 |
| **修 DCU device bug** | `5003` / `7001` reference 的 forward 内张量构造漏 `device=`（另有 `2001_activation` 同类 1 行） | 在 DCU 上 `torch.zeros` 默认建在 CPU，与另一侧 GPU 张量运算直接 device mismatch |
| **`static_audit` 补 HIP 设备原语库** | `GLOBAL_FORBIDDEN` 增加 `rocprim` / `hipcub` | 迁到 HIP 后原禁用清单没有 ROCm 侧原语库，用 `hipcub::DeviceScan` 整段完成 scan 即绕过（实测改前对 8001 报 `passed=true`） |
| **`triton_compat.py` 垫片** | 在 DCU 版 triton 3.3.0 上补齐 aiter 依赖的 4 个缺失 API：`tl.matrix_load`、`tl.constexpr_function`、`autotune(perf_debug=)`、`triton.utils.hcutuner` | aiter 按更高版本 triton 写，缺这些 API 时 import 即失败，aiter 基线无从采集 |
| **`aiter_shim` 垫片** | 部署侧提供 `aiter/__init__.py` + `_triton_compat.py` + `aiter/jit/` | 真 aiter 含未编译的扩展，直接 import 会崩；基线与离线审计需要能 import 它 |

> 其中「改提示」的三项（track 迁移、DCU 约束句、锚点说明）是**题目数据**改动，
> 会同步到 `benchmark/kernelbench_compat/` 镜像树（两者逐字节一致，由测试强制）。

---

**数据与复现**：详细统计见 `perf_report.md`；统计数据由 `stats_full66.py` 从
66 份 run.json 与 `offline_audit.json` 生成，可一条命令重算。
