# DCU 算子生成：框架臂 vs 纯 API 臂（2026-11-22，gfx928 全量）

> 66 题 × 2 轨（HIP / Triton），同一模型 `gpt-6-astra`、同一台 DCU（**K500SM_AI / gfx928**，DTK 26.04）、
> 同一在线评测器（`KernelBenchPaperEvaluator` + DCUKB 桥）与同一套离线审计（`audit_batch.py`）。
> 两条臂各单次运行，数据采集于 2026-11-21 18:23 ~ 11-22 15:21。
>
> **口径（与上一轮不同，见 §0）**：在线 = 1116 契约（**原生 dtype + `task.yaml` 容差 + 执行前静态审计 + oracle root 不进候选池**）；
> 离线 = 静态审计（禁用算子/平台符号）+ 隐藏 case（分档容差）；**真解 = 在线正确 ∩ 离线通过**。
> 加速比分母一律为同一轮现测的 `torch_eager`。

---

## 0. 与上一次（2026-11-03/04，BW / gfx936）的差异

### 0.1 机器

| 项 | 上次（旧机） | 这次（2026-11-21 起） |
|---|---|---|
| 设备 | BW，`gfx936` | **K500SM_AI，`gfx928:sramecc+:xnack-`** |
| 计算单元 | —（容器内 15 核可见） | **120 CU**，4 SIMD/CU，40 wave/CU |
| 可见 CPU / 内存 | 15 核 / 59 GB | **32 核 / 503 GiB**（无 cgroup 上限） |
| 显存 | 59 GB | **65520 MiB** |
| wavefront | 旧机按 `gfx936` 写、未按新卡重测 | **64**（`warp_size=64`，32 车道假设会空转一半） |
| 静态共享内存/block | 同上（旧值未复测） | **65536 B**（96 KiB 编译期报 `local memory exceeds limit`） |
| L2 / 主频 | — | 8 MiB / 1400 MHz |
| 可见卡数 | 多卡 | **1 张**（`device_count()==1`，单 `/dev/kfd`） |
| 软件栈 | DTK 26.04 + torch 2.9.0+das | 同（torch `2.9.0+das.opt1.dtk2604`）+ **triton `3.3.0+das.opt1.dtk2604.torch290`（已验 JIT）** |
| 外网 | 有 | **无**（pypi/公网 443 全不通），LLM 走反向隧道 |

**影响**：题面硬件句、DCU Wiki 全部 36 页、消融配置的 `operator_memory_target_arch` 都按 gfx928 重写；
加速比的分母虽然都是"同轮现测 `torch_eager`"，但**机器换了，旧数字不能与本轮拼接**（同一算子在不同卡上的 eager/自定义实现比例不同）。

### 0.2 框架

| 项 | 上次（框架 v2） | 这次 |
|---|---|---|
| 基座 | KernelWeaver `c11ab7f` 时代快照 | KernelWeaver **`cd5d85c`** + 两个补丁序列 |
| 补丁 | 合规页 v2（docstring 软约束 + 离线审计） | `patches/kernelweaver`（**1116 在线契约**）+ `patches/kernelweaver-dcu-wiki`（Wiki 装载 + 检索槽位修复） |
| 在线判据 | KernelBench 语义：输入统一 cast **fp32**、`allclose(1e-4)`、单一 shape | **原生 dtype**（含 int32 索引）+ `task.yaml` 分档容差 + **执行前静态审计** + oracle root 不进排行榜/不导出 |
| 知识注入 | 无 DCU Wiki（合规页靠题面 docstring） | **DCU 公开 Wiki 36 页**：gfx928 实测硬件页（wavefront 64 / 65536 B / 120 CU）+ 合规页常驻 + 算子页（20 family） |
| 检索 | — | family 命中（`operator_api_structure_only`），每题上下文 4 页；**68/68 题都拿到硬件页**（此前 31/68） |
| root 处理 | 聚合时手工排除 | 框架层排除（`_evaluate_generated`：原样返回 scaffold 判 `oracle_not_candidate`） |
| 题面契约摘要 | 无 | `prepare_task()` 把公开 `io/tolerance/forbidden` 注入 description 与 scaffold 顶部注释，**且强制纯 ASCII**（修掉了中文散文泄漏） |
| 记忆 | 共享经验池（JSON） | 同机制，本次共享池 249 条 / 63 题 |

### 0.3 运行配置与标定

- 12 分片（6/轨）、`max_attempts=10`、`dcu_d_combined`（epsilon 0.3、`root_child_limit` 5、`leaderboard` 2、温度全 0）
- 纯 API 臂：一次调用生成完整 `ModelNew`，无 Plan/Debug、无记忆、**无 Wiki**、`DCUKB_ROOT` 同一套判据
- 分母：两臂、两轨一律 `torch_eager` 同轮现测（`run.json` 的 `reference_runtimes`）
- 与上次的**方向性**对照（不可直接拼接，判据都换了）：

| 指标（真解口径不同 → 仅参考方向） | 上次 框架 HIP | 这次 框架 HIP | 上次 框架 Triton | 这次 框架 Triton |
|---|---|---|---|---|
| Correct Rate | 57/66（86%） | **61/66（92.4%）** | 56–59/66 | **62/66（93.9%）** |
| Fast1@all | 44/66 | **54/66（在线）/ 53（真解）** | 53/66 | **58/66（在线）/ 51（真解）** |

---

## 1. 在线指标（HIP 轨）

| 指标 | **框架臂** | 纯 API 臂 |
|---|---|---|
| Compile Rate（模块级） | **65/66（98.5%）** | 62/66（93.9%） |
| Compile Rate（kernel 级，跨轨口径） | **65/66** | 62/66 |
| Correct Rate | **61/66（92.4%）** | 50/66（75.8%） |
| Success@1 | **55/66** | 50/66 |
| Success@5 | **61/66** | 50/66 |
| Success@10 | **61/66** | 50/66 |
| Fast1@1 | 41/66 | **42/66** |
| Fast1@5 | **51/66** | 42/66 |
| Fast1@10（= Fast1@all） | **54/66（81.8%）** | 42/66（63.6%） |
| Geomean Speedup | **6.91×** | 4.91× |
| Median Speedup | **5.72×** | 3.98× |
| Mean Speedup | 59.73× | **81.67×** |
| Best Speedup | **1817.85×** | 1548.14× |
| First Correct Median | 1.0 | 1.0 |
| Refine Gain Median | **1.53×** | 1.00× |

First Correct 分布（尝试序号→题数）：框架 `1→55、2→4、3→2`；纯 API `1→50`（只有 1 个候选，定义上恒为 1.0）。

## 2. 在线指标（Triton 轨）

| 指标 | **框架臂** | 纯 API 臂 |
|---|---|---|
| Compile Rate（模块级） | **66/66（100%）** | 65/66（98.5%） |
| Compile Rate（kernel 级，跨轨口径） | **66/66** | 65/66 |
| Correct Rate | **62/66（93.9%）** | 51/66（77.3%） |
| Success@1 | 43/66 | **51/66** |
| Success@5 | **61/66** | 51/66 |
| Success@10 | **62/66** | 51/66 |
| Fast1@1 | 40/66 | **44/66** |
| Fast1@5 | **56/66** | 44/66 |
| Fast1@10（= Fast1@all） | **58/66（87.9%）** | 44/66（66.7%） |
| Geomean Speedup | 6.56× | **7.24×** |
| Median Speedup | 3.95× | **5.57×** |
| Mean Speedup | 86.17× | **160.64×** |
| Best Speedup | 2051.80× | **5694.40×** |
| First Correct Median | 1.0 | 1.0 |
| Refine Gain Median | **1.19×** | 1.00× |

First Correct 分布：框架 `1→43、2→15、3→3、9→1`；纯 API `1→51`。

> **Success@1 低于纯 API 是设计的必然**：框架第 1 个候选通常还是探索性的，价值体现在 @5/@10（HIP 50→61、Triton 43→61）与 Refine Gain（1.19–1.53×）。

## 3. 离线与真解

| | **框架 HIP** | 纯 API HIP | **框架 Triton** | 纯 API Triton |
|---|---|---|---|---|
| 在线 Correct | 61 | 50 | 62 | 51 |
| 离线通过（静态审计 + 隐藏 case） | **60** | 45 | **55** | 50 |
| **真解（在线 ∩ 离线）** | **60/66** | 45/66 | **55/66** | 50/66 |
| 真解 Fast1@all | **53** | 39 | **51** | 43 |
| 真解 Geomean / Median / Best | 6.33× / 5.53× / 544.77× | 4.32× / 3.90× / 1548.14× | 4.45× / 3.15× / 441.90× | 7.03× / 5.43× / 5694.40× |
| 在线对但离线不过 | 1（`4018_quant`） | 5 | 7 | 1 |

离线的判定构成（框架臂 Triton 为例）：4 题在线就没解出来、7 题隐藏 case 不过（其中 6 题是候选在**隐藏 shape 上 Triton 编译失败**，1 题数值不达标）。

## 4. 逐题差异（真解集合）

**HIP 轨**：都通过 45 题；**只有框架通过 15 题**；只有纯 API 通过 **0 题**；都没过 6 题。
- 只有框架通过：`1002_paged_attention`、`1011_extend_attention`、`1014_hstu_attention`、`1015_lean_atten`、`1016_lean_atten_paged`、`1017_mha`、`1019_mha_onekernel_bwd`、`1022_pa_prefill`、`1031_deform_attention_fwd`、`2010_batchnorm_fwd`、`2014_instancenorm_fwd`、`3003_moe_op`、`3006_moe_op_mxfp4`、`3007_moe_op_silu_fused`、`5001_conv2d_fwd`
- 都没过：`1027_sage_attention_qk_int8_per_block_causal`、`1032_deform_attention_bwd`、`4018_quant`、`5002_conv2d_bwd`、`5004_deform_conv2d_bwd`、`5006_spconv_bwd`

**Triton 轨**：都通过 47 题；**只有框架通过 8 题**；只有纯 API 通过 3 题；都没过 8 题。
- 只有框架通过：`1012_flash_attention_forward`、`1015_lean_atten`、`1017_mha`、`1027_sage_attention_qk_int8_per_block_causal`、`2010_batchnorm_fwd`、`2014_instancenorm_fwd`、`4009_gemm_a8w8`、`4010_gemm_a8w8_blockscale`
- 只有纯 API 通过：`1014_hstu_attention`、`3003_moe_op`、`8001_cumsum`
- 都没过：`1016_lean_atten_paged`、`1032_deform_attention_bwd`、`3002_moe_align_block_size`、`3007_moe_op_silu_fused`、`4018_quant`、`5002_conv2d_bwd`、`5004_deform_conv2d_bwd`、`5006_spconv_bwd`

**跨四轨都没过的 4 题：`1032_deform_attention_bwd`、`5002_conv2d_bwd`、`5004_deform_conv2d_bwd`、`5006_spconv_bwd`**
—— ⚠️ **这 4 题不是"题目难"，是评测桥接的 bug**（2026-11-23 复查结论）：

- 它们的参考实现用 `torch.autograd.grad` 反传求梯度，而 `dcukb_online._evaluate()` 把 trials 包在
  `with torch.no_grad():` 里 → 参考实现自己就抛
  `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`；
- 参考实现对不上，**任何候选**都判负 → 这 4 题在两条臂、两条轨上恒为失败；
- 同一坑 2026-10-16 已在 KernelBench 的 `eval.py` 上修过（`benchmark/tools/harness_fixes.md` 缺陷 1/2），
  但 1116 契约绕开那个 eval、走我们自己的桥，于是在这里又复现了一次；
- 已修：`dcukb_online.py` 与 `audit_model_class.py` 的三处 `torch.no_grad()` 改成
  `torch.enable_grad()`（前向题不建图、零开销；反向题才能跑）。真机 A/B 验证：4 题的
  public / hidden 两条路径从 `False` 全部变为 `True`。

**因此本报告的 Correct Rate 对两条臂都系统性少算了最多 4 题**（两组同减，对比结论不受影响，
但绝对数字偏低）。补测这 4 题后需要重算 §1–§3 的数字。

## 5. 口径与偏差（引用数字时必须一起给）

1. **两次的在线判据不同**：上一轮是 `fp32 + allclose(1e-4)`，本轮是**原生 dtype + `task.yaml` 容差 + 执行前静态审计**。因此本轮的 Compile Rate 语义是"编译通过**且**静态合规"，与旧数据不可拼接。
2. **HIP 轨中途换过条件**（同一批数字内部不完全同条件，如实标注）：
   - 评测超时 `240s → 900s`（240s 对 hipcc 编译太紧，曾让 HIP 轨 94 次超时判负）；
   - 分片 `6 → 12`（重切片，**记忆暴露时序前后不一致**；逐题召回条数 `task_experience_count` 已在明细里，可做协变量控制）；
   - `PYTORCH_ROCM_ARCH=gfx928`（此前每个候选按 5 个架构编译，编译量 5×）。
3. **一次 8 小时中断**：隧道抖动导致 provider 预检失败，框架对预检失败会 abort 整批（并照写 `summary.json`），中断发生在 HIP 轨早期，后续已由续跑看护补完；中断期的失败不计入任何指标。
4. **Accelerator 侧数字越小越可疑**：`Mean Speedup` 与 `Best Speedup` 由个别 GEMM 题主导（纯 API Triton Best 5694×），不要用它下能力结论；**覆盖率（真解数）与 Fast1 才是可比的**。
5. **Geomean 反向的解释**：Triton 轨纯 API Geomean（7.03×）高于框架（4.45×），是因为两边真解集合不同——框架多解出的 8 题是 attention/GEMM 块量化类（加速比天然低），纯 API 的真解集合里高加速比算子占比更高。HIP 轨两边真解集合差异更小，框架反而领先（6.33× vs 4.32×）。
6. 两条臂都是**单次运行**，逐题数字含采样噪声（温度 0 但不能消除服务端非确定性）；聚合指标才可比。
7. **评测桥接的 `no_grad` bug**（2026-11-23 发现并修复）：反向题（`1032`/`5002`/`5004`/`5006`）的参考实现
   需要 autograd，被 `torch.no_grad()` 包住时直接抛错，导致这 4 题对谁都判负。修复后它们的在线正确性
   与离线终审都恢复可判。**本报告的数字未补测**，引用时请注明"4 道反向题因桥接 bug 未计入有效结果"。
8. **Compile Rate 的口径**：1116 契约把**静态审计失败**标成 `failure_stage=compile`，所以这一栏实际是
   "编译通过 **且** 静态合规"。本轮 HIP 轨 65/66 就是被 `1032_deform_attention_bwd` 拉下的——它的 10 个候选
   全部调用了 `torch.softmax`（参考实现自己也用，但 `task.yaml` 禁止提交侧调用），并非编译失败。

## 6. 复现

```bash
# 生成（框架臂）：12 分片并行，环境见 docs/machine-2026-11-21-k500sm-gfx928.md §5.4
bash benchmark/tools/launch_framework_arm.sh        # RUN_ROOT=/root/runs2 SEARCH_CONFIG=dcu_d_combined

# 生成（纯 API 臂）
DCUKB_ROOT=<repo> KW_CODE=<KernelWeaver> python3 benchmark/tools/pure_bench.py \
  --manifest <manifest> --backend {triton,cuda} \
  --kernelbench-root <repo>/benchmark/kernelbench_compat --out <dir> --workers 5

# 指标（在线）
python3 benchmark/tools/collect_metrics.py --run-root <dir>
# 离线终审（隐藏 case + 静态审计）
python3 benchmark/evaluator/audit_batch.py --run-root <dir> --manifest <manifest> --skip-perf
# 真解（在线 ∩ 离线）：README 里给的 gated 脚本 / 见本报告 §3
```

**本轮原始数据**：`run.json` ×264、`best_code.py` ×224、逐题离线结果 ×264、`gated_summary.json`（逐题 online/offline/gated/加速比/召回条数）。
