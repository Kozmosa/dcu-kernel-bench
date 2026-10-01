# AGENTS.md

## 项目方向

**dcu-kernel-bench**：面向海光 DCU（gfx936/gfx938）的算子生成评测集，方法论参照寒武纪 BANG C 算子评测集构建指南（分层隔离、统一 Starter、隐藏 Case、静态审计防绕过）。

主算子来源：`third_party/aiter`（OpenDAS/aiter，HCU 适配的开源算子库）。补充题源：FlagGems（Triton）、composable_kernel-das / HYTLASS（GEMM/Conv 类）、HYGON-AI 组织。闭源库（hipDNN、LightOP）只作性能基线，不作题源。

## 铁律

1. **源码隔离**：`third_party/aiter/` 和 `benchmark/private/` 的内容**绝不**进入给生成 Agent 的上下文。Agent 只看 `benchmark/tasks/<id>/`。
2. **禁止绕过**：生成实现的核心计算必须在提交文件中完成，禁止调用 MIOpen / rocBLAS / hipBLASLt / hipDNN / ATen 完成核心计算（`static_audit.py` 强制检查）。
3. **统一接口**：同一入口形态的任务共用同一结构——callable 任务共用 Starter（Agent 只补全 kernel 与 launch）；model_class 任务共用框架生成的 `ModelNew` scaffold（Agent 只在锚点区域内编辑）。
4. **环境一致**：官方基线与生成实现必须在同一机器、同一 `environment.yaml` 锁定环境下比较。

## 目录结构

| 目录 | 跟踪 | 用途 |
|---|---|---|
| `benchmark/tasks/` | ✅ | Agent 可见的任务定义（task.yaml / reference.py / public_cases.json；callable 形态另有 starter/） |
| `benchmark/private/` | ✅ | 隐藏案例与基线，**仅评测端使用**（支持 `inherit` 复用同语义任务） |
| `benchmark/sources/` | ✅ | 准入审核记录 |
| `benchmark/evaluator/` | ✅ | 评测管线脚本（static_audit.py / audit_model_class.py / run_eval.py 骨架） |
| `benchmark/kernelbench_compat/` | ✅ | model_class 形态的框架寻题镜像树（`level{N}/<id>_*.py`，与 tasks 侧 reference.py 逐字节一致，测试强制） |
| `third_party/` | submodule | 外部仓库（aiter 等）以 git submodule 注册，**不 vendor 源码**；勿 `git add -f` 子仓内容 |
| `operator_catalog.yaml` | ✅ | 全量算子登记与准入状态 |
| `environment.yaml` | ✅ | 硬件/软件版本锁定 |

## 新增一个任务的流程

**第 0 步（定形态）**：在两种入口形态里选一，并在 catalog 的 `entry` 字段登记——

- `callable`（函数式，样板 1001）：题面框架无关，reference.py 提供 `reference()` + `make_inputs()`，starter 为待补全骨架。生成侧需自建 harness（loader/evaluator，尚未实现）。
- `model_class`（KernelBench 兼容，样板 1002，**当前唯一全链路打通的形态，新任务默认**）：reference.py 为自包含题目文件（`Model` + `get_init_inputs` + `get_inputs`），无 starter——`ModelNew` scaffold 由 PyramidKernel loader 从 Model 自动生成；**Model 的 docstring 就是 Agent 可见题面全文**（语义 + 约束 + forbidden + DCU 目标），不得含 aiter 溯源与 private 路径。

1. `operator_catalog.yaml` 登记（来源路径、commit、impl_lang、core_compute、entry、difficulty）。
2. 准入审核 → `benchmark/sources/<id>.yaml`：核心计算在可审查源码中、非闭源库包装，记录文件 sha256 证据；同语义变体用 `derived_from` 引用母题（样板：1002 引用 1001）。
3. 创建 `benchmark/tasks/<id>/`：task.yaml（语义、dtype/shape 域、**容差**、forbidden）、reference.py（按形态）、public_cases.json（callable：公开 case 列表；model_class：get_inputs 固定 shape 族契约）。
4. model_class 形态镜像到 `benchmark/kernelbench_compat/level{N}/<id>_<name>.py`（level 映射 difficulty：basic=1 / medium=2 / hard=3），与 tasks 侧逐字节一致。
5. `benchmark/private/<id>/`：hidden_cases.json（边界/极值/正确性）、perf_cases.json、baseline.json 占位；与已有任务同语义时用 `{"inherit": "<task_id>"}` 复用，生成器为母题的 `make_inputs`，不复制内容防漂移。
6. 写测试进 `tests/`：产物一致性（case/catalog/sources 交叉核对）、reference 语义（独立 oracle）、model_class 另需——镜像字节一致、PyramidKernel loader 构建 scaffold 且静态守卫通过、与母题输出逐位相等、fp32 cast 生存、隔离自检（TaskSpec 全字段无 `pa_decode`/`OpenDAS`/commit/`private` 等溯源词）。
7. 本地验证（WSL venv，见环境注意）：pytest 全绿。真机验证（部署方式见 `../notes/曙光环境访问.md`）：pytest 全绿 + quick profile mock 冒烟（全部候选过静态守卫、到达评测器、唯一失败原因为环境缺失）。
8. 离线终审实裁：model_class 用 `benchmark/evaluator/audit_model_class.py`（静态审计 → 隐藏 case 按 task.yaml 容差 → perf 计时）在真机跑 reference 自检与首个生成产物，确认题目包与私有资产可复现。`run_eval.py` 为 callable 形态的终审骨架，其 TODO(dcu) 在该形态接入时补全。

## 当前进度：生成框架接入（2026-10-01）

采用 PyramidKernel（`../PyramidKernel`，与本仓库同级）驱动 Agent 生成 kernel，评测其在 DCU 上编写算子的能力。**当前走其原生 model_class 路径**：Agent 在 `ModelNew` scaffold 锚点内写**纯 Triton 代码**（`helpers` 区放 `@triton.jit` kernel，`forward_stmt_N` 区放 launch 逻辑），在 DCU 上由 DTK Triton JIT 编译为 gfx936，全程无 hipcc 参与；在线正确性/计时由框架自带的 KernelBenchPaperEvaluator 判定。

已落地：

- **1002_paged_attention**：1001 的 model_class 变体，题目文件 `benchmark/tasks/1002_paged_attention/reference.py`（自包含 Model/get_inputs/get_init_inputs，计算与 1001 逐行一致），镜像到 `benchmark/kernelbench_compat/level3/`（字节一致性由 `tests/test_1002_model_class.py` 保证）。运行：`--kernelbench-root benchmark/kernelbench_compat --level 3 --problem-id 1002 --backend triton`。
- 本机（WSL，CPU）验证到"缺 GPU"为止：26/26 测试绿；quick profile（mock provider）全链路冒烟通过——静态守卫 0 拦截、全部节点到达评测器、唯一失败原因为 `CUDA is not available`。
- 语义边界（有意为之）：在线评测为 KernelBench 语义（输入统一 cast fp32、allclose 1e-4、全局 RNG 输入、forbidden 仅 docstring 软约束）；本评测集的容差/隐藏 case/static_audit 硬审计/aiter 基线属**离线终审**，工具为 `benchmark/evaluator/audit_model_class.py`（静态审计 → 隐藏 case 正确性 → perf 计时；case 资产经 private/1002 的 `inherit` 字段复用 1001；static_audit 已支持剥除 docstring，避免题面声明禁用项被误杀）。GLM-5.3-flash 首个产物已通过全部终审（7 隐藏 case 含非 2 次幂 head/bf16/GQA，在线从未见过的 shape 族）。

待办：构建 aiter 性能基线（真机装 aiter 后测，填 baseline.json）；扩大真 LLM 实验规模（多任务、多预算、与 aiter 基线对比）。

真机（BW / gfx936，DTK 26.04）已验证（2026-10-01）：测试 26/26 绿；原生路径 quick 冒烟全部节点 correct（mock provider + 真评测器 + cuda_event 计时）；**Triton 3.3.0+das.opt1.dtk2604.torch290 已安装并验证 JIT**（vecadd + tl.dot fp16 矩阵乘）；**首个真 LLM 端到端完成**（GLM-5.3-flash 经 CC Switch 反向隧道，1002 任务 speedup 17.35x，含 debug 修复环路），访问与部署细节见 `../notes/曙光环境访问.md`。

## 环境注意

- 真机评测依赖 DTK ≥ 25.04（Triton track 的硬要求）、hipcc、DTK 版 PyTorch。
- 开发 venv 在 WSL 侧 `~/.venvs/dcu-kernel-bench`（Python 3.13，torch CPU + pytest + pyramidkernel editable + kernelbench 依赖）。**不要把 venv 放回 `/mnt/e`**：NTFS 挂载上的 `.venv` 已两次被清空 `bin/` 与 `site-packages/`（原因未查明）。跑测试：`wsl -e bash -lc "cd /mnt/e/Code/agent-kernel-dcu/dcu-kernel-bench && $HOME/.venvs/dcu-kernel-bench/bin/python -m pytest -q"`。
- `third_party/aiter` 是独立 git 仓库。Windows 下它有两个文件名带冒号、无法 checkout 的 Triton autotune JSON，已用 sparse-checkout 排除并设 `core.protectNTFS=false`——**不要在该仓库内执行 `git reset --hard`**。
- 网络：GitHub 直连不通，境外资源走代理 127.0.0.1:16888 或 ghfast.top 镜像；developer.sourcefind.cn / download.sourcefind.cn 直连。
