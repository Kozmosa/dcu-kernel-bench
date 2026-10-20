# A/B：详版题面 vs 精简题面（66 道，同模型同 profile）

**问题**：Agent 能直接读到参考实现（root scaffold 就是 `Model` 本体），题面里再复述数学定义与实现约束是否多余？
**做法**：只改 docstring、不改任何代码（66/66 文件剥掉 docstring 后 AST 完全相同），两臂各跑一次全量 66 道，`main` profile，16 分片，同一模型 `gpt-6-astra`。

| 臂 | 题面 | 字符数 | 数据 |
|---|---|---|---|
| **A** | 详版：数学定义 + I/O 规格 + 边界行为 + 实现约束 + 禁用列表 + 硬件 | 246041 | `runs/dcukb_en_1019_1027` → `.dcu_runs/en66` |
| **B** | 精简：摘要 + I/O 契约（含隐藏 case 域）+ 禁用列表 + 硬件 + 一句锚点规则 | 118626（**-51.8%**） | `runs/dcukb_lean_1020_1030`（61 道）+ 补跑 5 道 → `.dcu_runs/lean66` |

B 里删掉的段落：`Mathematical definition`、`Boundary behaviour`、打包协议、`Task scope`、`Implementation constraints` 细节，以及 `online softmax / two-pass / split-K` 这类做法词。

## 指标对比（同口径：排除 root 候选）

| 指标 | A（详版） | B（精简） | Δ |
|---|---|---|---|
| Compile Rate | **66/66** | 63/66 | **-3** |
| Correct Rate | **60/66（90.9%）** | 60/66（90.9%） | 0 |
| Success@1 | 26/66 | 26/66 | 0 |
| Success@5 | 56/66 | 54/66 | -2 |
| Fast1@all（预算内快于 eager） | 50/66（75.8%） | **54/66（81.8%）** | **+4** |
| Geomean Speedup | 7.82x | **10.02x** | +2.2x |
| Median Speedup | 6.39x | **7.45x** | +1.06x |
| Mean Speedup | **64.61x** | 52.17x | -12.4x |
| Best Speedup | **900.66x** | 872.61x | -28.1x |
| First Correct Median | 2.0 | 2.0 | 0 |
| Refine Gain Median | 1.13x | 1.13x | 0 |

## 逐题看：差异基本是采样噪声

- 两臂都 correct 的 58 道里，**B 更快 30 道、A 更快 28 道**——接近抛硬币；单臂单次运行，任务级差异被 LLM 采样主导。
- 状态翻转 5 道：B 新增 correct `1027`、`4017`；B 丢掉 correct `2014`、`5004`，另有 `2013` 由"能编译"退化为"没有可编译候选"。
- **B 唯一明确的退步是 Compile Rate**：`2013_groupnorm_bwd`、`2014_instancenorm_fwd`、`5004_deform_conv2d_bwd` 三道在 B 臂里**一个可编译候选都没产出**。这三道的详版题面里恰好有"中间累加用 fp32 / 输出 cast 回输入 dtype / 签名不可改"这类硬约束——删掉后 agent 更容易写出编译不过的补丁。
- 变化最大的几道（`2012` 0.21x→11.64x、`5007` 5.18x→148x、`1022` 1.94x→30x、`3003` 44x→267x）幅度远超题面影响，属采样/搜索路径差异，不能归因给题面。

## 点名看：题面里原本写了题解的 4 道

| 题 | 原题面里的泄露 | A（详版） | B（精简） | 结论 |
|---|---|---|---|---|
| `2009_softmax` | "Target implementation shape"整段：单 kernel、扫两趟、rescale 递推公式 | 2.53x，首次正确第 3 次 | 2.60x，首次正确第 **1** 次 | 删掉反而更好，说明这段题解没帮上忙 |
| `1021_pa_decode` | "flash-decoding style split-KV + 1024 splits + LSE 合并公式" | 7.49x | **3.56x** | **删了确实掉一半**——这是唯一能被题面解释的退化 |
| `1029_triton_decode_attention` | "Reference structure: two-stage split-KV, stage1/stage2" | 21.20x | **29.31x** | 删了反而更快 |
| `1020_mla_decode_rope` | "num_kv_splits … sliced online softmax with log-sum-exp merging" | 0.77x | 0.71x | 基本无变化（都慢于 eager） |

## 结论

1. **精简题面在聚合指标上不吃亏**：Correct Rate 持平（60/66），First Correct Median、Refine Gain 完全一致，Fast1@all 与 Geomean/Median 反而更高。
2. **代价集中在 Compile Rate**（66→63）：被删掉的"实现约束"段确实在帮 3 道数值类反问题（groupnorm/batchnorm/deform-conv bwd）少写坏补丁。
3. **题解级提示只有 `1021` 一道真被用到**（7.49x → 3.56x），其余 3 道删掉后没有变差。也就是说"把做法写进题面"对分数的贡献远小于预期，但**确实会污染那一道题的分数**。
4. **必须声明的局限**：每臂只跑了一次，任务级 delta 属噪声（30/58 vs 28/58）；要给出可信的题面效应，需要 3 次以上重复取均值，或用配对 bootstrap。

## 复现

```bash
# A 臂指标
python benchmark/tools/collect_metrics.py --run-root .dcu_runs/en66  --out benchmark/tools/metrics_en.md
# B 臂指标
python benchmark/tools/collect_metrics.py --run-root .dcu_runs/lean66 --out benchmark/tools/metrics_lean.md
# B 臂题面生成
python benchmark/tools/make_lean_prompts.py --out .dcu_runs/min/benchmark --mode minimal --write
```
