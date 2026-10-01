# AGENTS.md

## 项目方向

**dcu-kernel-bench**：面向海光 DCU（gfx936/gfx938）的算子生成评测集，方法论参照寒武纪 BANG C 算子评测集构建指南（分层隔离、统一 Starter、隐藏 Case、静态审计防绕过）。

主算子来源：`third_party/aiter`（OpenDAS/aiter，HCU 适配的开源算子库）。补充题源：FlagGems（Triton）、composable_kernel-das / HYTLASS（GEMM/Conv 类）、HYGON-AI 组织。闭源库（hipDNN、LightOP）只作性能基线，不作题源。

## 铁律

1. **源码隔离**：`third_party/aiter/` 和 `benchmark/private/` 的内容**绝不**进入给生成 Agent 的上下文。Agent 只看 `benchmark/tasks/<id>/`。
2. **禁止绕过**：生成实现的核心计算必须在提交文件中完成，禁止调用 MIOpen / rocBLAS / hipBLASLt / hipDNN / ATen 完成核心计算（`static_audit.py` 强制检查）。
3. **统一接口**：所有任务共用同一 Starter 结构，Agent 只补全 kernel 文件与必要的 launch 逻辑。
4. **环境一致**：官方基线与生成实现必须在同一机器、同一 `environment.yaml` 锁定环境下比较。

## 目录结构

| 目录 | 跟踪 | 用途 |
|---|---|---|
| `benchmark/tasks/` | ✅ | Agent 可见的任务定义（task.yaml / reference.py / public_cases.json / starter/） |
| `benchmark/private/` | ✅ | 隐藏案例与基线，**仅评测端使用** |
| `benchmark/sources/` | ✅ | 准入审核记录 |
| `benchmark/evaluator/` | ✅ | 评测管线脚本 |
| `third_party/` | submodule | 外部仓库（aiter 等）以 git submodule 注册，**不 vendor 源码**；勿 `git add -f` 子仓内容 |
| `operator_catalog.yaml` | ✅ | 全量算子登记与准入状态 |
| `environment.yaml` | ✅ | 硬件/软件版本锁定 |

## 新增一个任务的流程

1. 在 `operator_catalog.yaml` 登记候选算子（来源路径、commit、impl_lang、核心计算位置）。
2. 完成准入审核，记录写入 `benchmark/sources/<id>.yaml`（确认核心计算在可审查源码中、非闭源库包装）。
3. 创建 `benchmark/tasks/<id>/`：编写 task.yaml（语义、dtype/shape 范围、容限、禁调用项）、reference.py（纯 PyTorch）、public_cases.json、starter。
4. 在 `benchmark/private/<id>/` 准备隐藏案例（正确性/边界/极值/性能）与基线占位。
5. 用 `benchmark/evaluator/run_eval.py` 在 DCU 真机上验证 reference 与基线可复现。

## 当前进度：生成框架接入（2026-10-01）

采用 PyramidKernel（`../PyramidKernel`，与本仓库同级）驱动 Agent 生成 kernel，评测其在 DCU 上编写算子的能力。**当前走其原生 model_class 路径**：Agent 在 `ModelNew` scaffold 锚点内写**纯 Triton 代码**（`helpers` 区放 `@triton.jit` kernel，`forward_stmt_N` 区放 launch 逻辑），在 DCU 上由 DTK Triton JIT 编译为 gfx936，全程无 hipcc 参与；在线正确性/计时由框架自带的 KernelBenchPaperEvaluator 判定。

已落地：

- **1002_paged_attention**：1001 的 model_class 变体，题目文件 `benchmark/tasks/1002_paged_attention/reference.py`（自包含 Model/get_inputs/get_init_inputs，计算与 1001 逐行一致），镜像到 `benchmark/kernelbench_compat/level3/`（字节一致性由 `tests/test_1002_model_class.py` 保证）。运行：`--kernelbench-root benchmark/kernelbench_compat --level 3 --problem-id 1002 --backend triton`。
- 本机（WSL，CPU）验证到"缺 GPU"为止：26/26 测试绿；quick profile（mock provider）全链路冒烟通过——静态守卫 0 拦截、全部节点到达评测器、唯一失败原因为 `CUDA is not available`。
- 语义边界（有意为之）：在线评测为 KernelBench 语义（输入统一 cast fp32、allclose 1e-4、全局 RNG 输入、forbidden 仅 docstring 软约束）；本评测集的容差/隐藏 case/static_audit 硬审计/aiter 基线属**离线终审**，对 `runs/<task>/best_code.py` 执行。

待办：真机（DTK）跑通官方评测器 + 真 LLM provider 的小预算端到端；实现离线终审评测器（复用 task.yaml 容差、private/ 隐藏 case、static_audit.py）。

## 环境注意

- 真机评测依赖 DTK ≥ 25.04（Triton track 的硬要求）、hipcc、DTK 版 PyTorch。
- 开发 venv 在 WSL 侧 `~/.venvs/dcu-kernel-bench`（Python 3.13，torch CPU + pytest + pyramidkernel editable + kernelbench 依赖）。**不要把 venv 放回 `/mnt/e`**：NTFS 挂载上的 `.venv` 已两次被清空 `bin/` 与 `site-packages/`（原因未查明）。跑测试：`wsl -e bash -lc "cd /mnt/e/Code/agent-kernel-dcu/dcu-kernel-bench && $HOME/.venvs/dcu-kernel-bench/bin/python -m pytest -q"`。
- `third_party/aiter` 是独立 git 仓库。Windows 下它有两个文件名带冒号、无法 checkout 的 Triton autotune JSON，已用 sparse-checkout 排除并设 `core.protectNTFS=false`——**不要在该仓库内执行 `git reset --hard`**。
- 网络：GitHub 直连不通，境外资源走代理 127.0.0.1:16888 或 ghfast.top 镜像；developer.sourcefind.cn / download.sourcefind.cn 直连。
