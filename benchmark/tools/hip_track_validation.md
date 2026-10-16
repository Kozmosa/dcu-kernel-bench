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

## 复现

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

## 后续待办

- 20 道 hip 题需要用 HIP backend **重跑全量**（此前那次全量对它们是用 Triton 跑的，
  结果不作数）
- PyramidKernel 的默认 `CUDA_CU_SRC` 模板仍 include `<cuda_runtime.h>`（属该仓库的
  可改进项，本次未动）
