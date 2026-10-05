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
  - **硬约束：forward 必须返回单个张量**。KernelBench 评测器的 correctness check 直接调 `output.shape` 与 `torch.allclose(output, output_new)`，只支持单 tensor 输出——多值 tuple 输出的题无论产物对错都会在评测器内炸 `'tuple' object has no attribute 'shape'` 并被判负（1017/1019 首批实测踩坑，产物逐位正确仍被误杀）。算子天然多输出时用**一维拼接打包**：各输出按既有 dtype 约定 cast 后 `reshape(-1)`、`torch.cat` 成单个 float32 一维张量，docstring 与 task.yaml 写明分段协议（样板：1018 的 dq/dk/dv 拼接、1019 修正版、1020、3009 的"单张量打包协议"）。

1. `operator_catalog.yaml` 登记（来源路径、commit、impl_lang、core_compute、entry、difficulty）。
2. 准入审核 → `benchmark/sources/<id>.yaml`：核心计算在可审查源码中、非闭源库包装，记录文件 sha256 证据；同语义变体用 `derived_from` 引用母题（样板：1002 引用 1001）。
3. 创建 `benchmark/tasks/<id>/`：task.yaml（语义、dtype/shape 域、**容差**、forbidden、多输出题的打包分段协议）、reference.py（按形态，model_class 的 forward 输出遵守单张量硬约束）、public_cases.json（callable：公开 case 列表；model_class：get_inputs 固定 shape 族契约）。
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

- **本地批量生成实测（2026-10-09，RTX 4050 Laptop + triton-windows 3.8 + CC Switch 网关 GLM-5.3-flash，quick 档 6 attempts/题）**：10xx 段前 10 道题全量跑通，8 道 correct（speedup：1016 57.6x、1014 23.1x、1011 11.2x、1002 8.7x、1010 7.7x、1015 3.0x、1012 1.02x、1018 1.01x），1017/1019 因多值 tuple 输出被评测器误杀（产物数值逐位正确，已改为单张量打包协议修正）。运行环境要点：`PYTHONUTF8=1`（中文题面临时文件编码）、provider 配 `timeout_seconds: 1200` + `stream: false`、境外端点走 `NO_PROXY` 直连。产物在 `../PyramidKernel/runs/grokbatch/`。

待办：构建 aiter 性能基线（真机装 aiter 后测，填 baseline.json）；扩大真 LLM 实验规模（多任务、多预算、与 aiter 基线对比，1017/1019 打包修正后重跑）。

真机（BW / gfx936，DTK 26.04）已验证（2026-10-01）：测试 26/26 绿；原生路径 quick 冒烟全部节点 correct（mock provider + 真评测器 + cuda_event 计时）；**Triton 3.3.0+das.opt1.dtk2604.torch290 已安装并验证 JIT**（vecadd + tl.dot fp16 矩阵乘）；**首个真 LLM 端到端完成**（GLM-5.3-flash 经 CC Switch 反向隧道，1002 任务 speedup 17.35x，含 debug 修复环路），访问与部署细节见 `../notes/曙光环境访问.md`。

## 本分支相对上游 main 的增量（`main-merge`）

基座 = `origin/main` @ `5094481`（47 道题，四段号段 1xxx/2xxx/3xxx/4xxx）。本分支在其上**以加工具、测试与基线适配器为主**；对上游任务定义的改动只有一处必要的可跑性修复（`1017_mha` / `1019_mha_onekernel_bwd` 的单张量打包，见下文"上游缺陷"）：

| 增量 | 作用 |
|---|---|
| `benchmark/tools/dcukb.py` + `scan_candidates.json` | 算子收集流水线：`scan`（AST 扫 aiter 找候选并裁决）→ `admit`（sha256 证据）→ `new`（任务骨架）→ `mirror`（镜像一致性，`--check` 供 CI）。零第三方依赖 |
| `benchmark/evaluator/record_baseline.py` | aiter 官方基线采集器：逐 perf case 计时，且**先与 reference 按 task.yaml 容差比对通过才记录**（避免记下错误基线） |
| `benchmark/evaluator/audit_model_class.py`（替换上游版本） | 上游版本的**功能超集**：保留 `io.init_inputs` 按名取参 + 容差按输出 dtype，另加 ①逐参数用 `get_init_inputs()` 补齐 ②`io.init_inputs: []` 时无参构造 ③关键字绑定（`case_init_kwargs`），杜绝稀疏构造参数的位置错位 |
| `tests/test_compat_mirrors.py` | 通用镜像一致性：按 difficulty→level 映射覆盖**全部** model_class 题，并检查"错 level 的残留镜像" |
| `tests/test_reference_case_contract.py` | 通用 case 契约：逐 case 生成输入→构造 Model→跑 forward→校验单个 Tensor/数值有限/容差可解析；另含 `io.init_inputs` 声明与 Model 签名前缀一致、case 字段必须生效 |
| `benchmark/private/<id>/aiter_impl.py`（44 个） | aiter 官方实现适配器，使基线可采集。契约 `run(inputs, init_kwargs: dict, device)`；按号段 1xxx 18 / 2xxx 9 / 3xxx 7 / 4xxx 10。覆盖 44/45（唯一缺口 `4007_gemm_a16w16_atomic`，缺 `BW200-GEMM-A16W16-ATOMIC.json` tuner 配置） |

- `record_baseline.py` 的适配器契约是 `run(inputs, init_kwargs: dict, device)`（按名取参）。**上游一个 `aiter_impl.py` 都没有**，本分支已为 44 道题补齐（见上表）；**基线数值仍待真机采集**——本分支只抢救回 `1002` 的 3 条，其余 43 道仍是 placeholder。
- 本地跑全部测试（本机无 pytest，驱动自带垫片）：
  `PYTHONPATH=../PyramidKernel <venv-python> .dcu_runs/run_all_tests.py` → **213 项断言全绿**。

### 迁移时发现的上游缺陷（3 类，均已修）

1. **`1017_mha` / `1019_mha_onekernel_bwd`**：声明 `entry: model_class`，但 reference 的 `forward` **返回 tuple**（`out,lse` / `dq,dk,dv`）。KernelBench 的 `output.shape != output_new.shape` 与终审的 `actual.shape` 都会 AttributeError → **这两题在真机上跑不了**。**已修成"打包成单张量"**（本仓库已有先例：`2002_add_swiglu` 就是单张量拼接）：1017 按最后一维加 1 列（`[B,Sq,Hq,D]` + lse → `[B,Sq,Hq,D+1]`，dtype 不变）；1019 三段梯度形状可不同（GQA、`seqlen_q != seqlen_k`），各自展平后沿第 0 维拼接 → `[numel(dq)+numel(dk)+numel(dv)]`。两者的 `io.outputs`／Model docstring／compat 镜像已同步。测试白名单 `KNOWN_SINGLE_TENSOR_VIOLATIONS` 已清空（这是**唯一**与上游分叉的语义改动，建议回馈上游）。
2. **`4006_gemm_a16w16` / `4007_gemm_a16w16_atomic`**：case 只给 m/n/k，而构造参数 `in_features`/`out_features` 只在 `get_init_inputs()` 里；上游 `resolve_init_args` 在"case 与声明无公共字段"时直接抛 KeyError → 构造不出 Model。本分支的"逐参数补齐"已修。
3. **`4008_gemm_a16w4`**：同类缺口，`io.init_inputs` 只声明了 `group_size`，拿到部分 kwargs 后 `Model()` 缺 `in_features`/`out_features` 报 TypeError。同样已修。

### 关于 1001/1002 的重复题（保留上游形态）

在 dev 线上曾按"同一算子不重复登记"把 1001 并入 1002。**换到上游基座后保留两题**，因为代价已反超收益：上游把这对写成 `derived_from` + `inherit` 的**规范样板**（"第 0 步定形态"直接引用它），且 1001 同时是 `test_1002_model_class.py` 的独立 oracle（逐位相等校验）与 `test_reference_1001_*.py` 的被测对象——删它要连带改 3 个上游测试模块、private 资产与文档，而收益只是少一条登记（case 内容本就 inherit 复用，没有双份）。
**判据：重复的是"登记"还是"内容"——登记重复可接受，内容重复才必须消除。**

## 环境注意

- 真机评测依赖 DTK ≥ 25.04（Triton track 的硬要求）、hipcc、DTK 版 PyTorch。
- 开发 venv 在 WSL 侧 `~/.venvs/dcu-kernel-bench`（Python 3.13，torch CPU + pytest + pyramidkernel editable + kernelbench 依赖）。**不要把 venv 放回 `/mnt/e`**：NTFS 挂载上的 `.venv` 已两次被清空 `bin/` 与 `site-packages/`（原因未查明）。跑测试：`wsl -e bash -lc "cd /mnt/e/Code/agent-kernel-dcu/dcu-kernel-bench && $HOME/.venvs/dcu-kernel-bench/bin/python -m pytest -q"`。
- `third_party/aiter` 是独立 git 仓库。Windows 下它有两个文件名带冒号、无法 checkout 的 Triton autotune JSON，已用 sparse-checkout 排除并设 `core.protectNTFS=false`——**不要在该仓库内执行 `git reset --hard`**。
- 网络：GitHub 直连不通，境外资源走代理 127.0.0.1:16888 或 ghfast.top 镜像；developer.sourcefind.cn / download.sourcefind.cn 直连。
