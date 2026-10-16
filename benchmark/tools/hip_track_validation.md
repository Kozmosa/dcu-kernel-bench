# HIP track 端到端验证（2026-10-16，DCU BW / gfx936 / DTK 26.04）

## 结论

20 道 hipDNN 来源题走 `backend: cuda`（= `torch.utils.cpp_extension.load_inline` +
hipcc）的 HIP 生成路径**已端到端打通**：题面声明 HIP → 脚手架给 HIP 锚点 →
Agent 写 HIP 设备代码 → hipcc 编出 gfx936 产物 → 结果正确。

## 验证

单题冒烟（`runs/hip_smoke/8001_cumsum_l2_p8001/`，profile/search 均为 quick）：

| 项 | 值 |
|---|---|
| best_node_id | `n6`（真实生成节点，非 root） |
| compile_ok | **True** |
| correct | **True** |
| runtime / reference | 0.000123 / 0.000216 |
| speedup | **1.756x** |
| 节点日志 | `applied_region_patch:cuda_cu:replace` |

生成代码（`best_code.py`）里：`__global__` ×4、`hip/hip_runtime` ×2、`load_inline` ×2、
**`triton` ×0**；构建日志出现 `Successfully preprocessed all matching files.`
（即 `load_inline` 编译成功）。

## 打通需要的两处修复

### 1. PyramidKernel：`--use_fast_math` 用 DCU 不认的 nvcc 开关

`core/loader.py` 生成的 `_pyramidkernel_get_extension()` 硬编码
`extra_cuda_cflags=['-O3', '--use_fast_math']`，而 `--use_fast_math` 是 nvcc 专有，
海光 `dcc` 直接拒绝：

```
dcc: error: unsupported option '--use_fast_math'
[1/3] /opt/dtk/bin/hipcc ... cuda.hip -o cuda.cuda.o
FAILED: cuda.cuda.o
```

这段 helper 在锚点区**之外**（Agent 改不到），所以 `backend=cuda` 的所有题都会在
「实现完 kernel、首次调用扩展」时必然编译失败。

修法（PyramidKernel commit `ea649d6`）：按平台条件化，ROCm 下不加该 flag

```python
extra_cuda_cflags=['-O3'] + ([] if getattr(torch.version, 'hip', None)
                             else ['--use_fast_math'])
```

顺带在 `_ensure_cuda_imports` 的 required 里补 `import torch`（原只保证 hashlib 与
load_inline，而条件化需要 torch 可用）。

> 为什么不用 `-ffast-math` 替代：它会放松 IEEE 语义，而本评测集以容差判正确性，
> 部分题 `atol=rtol=0` 或要求严格 fp32 累加，fast-math 有制造假失败的风险。

### 2. 题面必须写明用 `<hip/hip_runtime.h>`

实测本机 hipcc **找不到 `<cuda_runtime.h>` / `<cuda.h>`**（只有 `<hip/hip_runtime.h>`
能编过），而 PyramidKernel 的默认 `CUDA_CU_SRC` 模板 include 的正是那两个头。
20 道题的题面已补：

> 目标硬件：海光 DCU（gfx936/gfx938），实现语言 HIP（经 hipcc 编译；在
> CUDA_CU_SRC / CUDA_CPP_SRC 锚点区编写，设备侧必须 include `<hip/hip_runtime.h>`，
> 本机无 `<cuda_runtime.h>`）。

## 复现（单题冒烟）

```bash
bash /root/dcudeploy/hip_smoke.sh                    # 默认 8001_cumsum
TASK=5001_conv2d_fwd LEVEL=3 PID_=5001 bash /root/dcudeploy/hip_smoke.sh
```

清单形态：

```yaml
name: hip_smoke
backend: cuda
tasks:
  - alias: 8001_cumsum
    level: 2
    problem_id: 8001
    backend: cuda
```

## 20 道题全量重跑（2026-10-16 20:42–21:30，20 片并行）

`manifest_hip20.yaml`（= `manifest_dcukb.yaml` 里 `backend: cuda` 的 20 道），
`N=20` 每题一片，输出 `runs/dcukb_hip20`，此后合并回 `runs/dcukb_final`。

**结果：20/20 全部以 `backend=cuda` 跑通，correct 11（真解 10 + 仅根节点 1）。**
全部 correct 的非根节点生成代码都含 `<hip/hip_runtime.h>`、`__global__` 且
**`triton` 残留为 0** —— 不是"换了壳还在跑 Triton"。

真解 speedup 中位数 **1.952x（Triton）→ 2.466x（HIP）**：

| 题 | Triton | HIP | | 题 | Triton | HIP |
|---|---|---|---|---|---|---|
| 2012_groupnorm_fwd | 1.368 | **6.868** | | 2014_instancenorm_fwd | 8.848 | 6.502 |
| 8003_scatter_nd | 1.842 | **2.567** | | 1031_deform_attention_fwd | 891.14 | 558.42 |
| 8002_gather | 1.858 | **1.950** | | 5001_conv2d_fwd | 2.046 | 1.026 |
| 2011_batchnorm_bwd | 4.814 | 4.444 | | 5008_conv_bias_act | 1.274 | 0.813 |
| 2010_batchnorm_fwd | 4.842 | 2.365 | | 8001_cumsum | 1.701 | 1.662 |

两轮的**未通过集合完全相同**（9 道）且 best 全部停在 `root` —— 说明失败与
HIP 迁移无关，是 harness 缺陷（见下）。`2013_groupnorm_bwd` 两轮都只有
root 正确（speedup ≈1.0），是 PyTorch 基线本身，不算 Agent 解。

合并后 `runs/dcukb_final` 共 **66 道**，backend 分布 `{triton: 46, cuda: 20}`；
全量 66 道 correct 57（**真解 47** + 仅根节点 10），真解 speedup 中位 6.50x。

### 9 道未通过的归因（全部非 Agent 能力问题）

| 类别 | 题 | 根因 |
|---|---|---|
| 脚手架非法（6 道） | 5003, 5004, 5005, 5007, 5009, 6001 | `_extract_signature` 只取 `get_source_segment` 的**首行**，多行签名被截断成括号未闭合的非法语法 |
| 同上但被 Agent 自行修好（1 道） | 5008 | 同一签名缺陷（另有 `IndentationError`），Agent 改对后 correct 0.81x |
| autograd 冲突（3 道） | 1032, 5002, 5006 | 该 3 题脚手架本就合法；`KernelBench/src/kernelbench/eval.py` 用 `torch.no_grad()` 包住执行，与需要求导的参考实现对冲突 |

签名缺陷同一根因还导致 `IndentationError`：`ast.get_source_segment` 只把**首行**
切到 `col_offset`、其余行保留绝对缩进，于是 `textwrap.dedent` 公共前缀恒为 0、
退化成空操作，补缩进后 `if`(8) / 首行体(20) / `elif`(16) 层级错乱。

复现与修法验证：`.dcu_runs/deploy/repro_scaffold_bug.py`（修前 7/10 非法）、
`verify_scaffold_fix.py`（修后 0/10，40 份脚手架全合法，测试 297 pass / 0 fail）。
补丁待应用：`.dcu_runs/deploy/apply_loader_fix.py`（默认 dry-run）。

## 复现（全量重跑）

```bash
bash /root/dcudeploy/rerun_hip20.sh     # 清旧 Triton 产物 -> 生成 hip20 清单 -> 起 20 片
bash /root/dcudeploy/finish_hip20.sh    # 对照 -> 合并回 dcukb_final -> 全量汇总
```

`finish_hip20.sh` 的顺序不能颠倒：对照读 `<RUN_ROOT>/par_s*/`，而
`collect_results.py --relocate` 会把题目录**移走**。

## 后续待办

- **应用脚手架补丁**（`.dcu_runs/deploy/apply_loader_fix.py --apply`）后重跑那 6 道
  非法脚手架的题 —— 它们现在的失败完全不能反映 Agent 能力。
- 3 道 autograd 题（1032/5002/5006）需要处理 `KernelBench/eval.py` 的
  `torch.no_grad()` 与求导参考的冲突。
- PyramidKernel 的默认 `CUDA_CU_SRC` 模板仍 include `<cuda_runtime.h>`（属该仓库的
  可改进项，本次未动）。
