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
| `third_party/` | ❌ | 外部仓库快照（aiter 等），不进 git |
| `operator_catalog.yaml` | ✅ | 全量算子登记与准入状态 |
| `environment.yaml` | ✅ | 硬件/软件版本锁定 |

## 新增一个任务的流程

1. 在 `operator_catalog.yaml` 登记候选算子（来源路径、commit、impl_lang、核心计算位置）。
2. 完成准入审核，记录写入 `benchmark/sources/<id>.yaml`（确认核心计算在可审查源码中、非闭源库包装）。
3. 创建 `benchmark/tasks/<id>/`：编写 task.yaml（语义、dtype/shape 范围、容限、禁调用项）、reference.py（纯 PyTorch）、public_cases.json、starter。
4. 在 `benchmark/private/<id>/` 准备隐藏案例（正确性/边界/极值/性能）与基线占位。
5. 用 `benchmark/evaluator/run_eval.py` 在 DCU 真机上验证 reference 与基线可复现。

## 环境注意

- 真机评测依赖 DTK ≥ 25.04（Triton track 的硬要求）、hipcc、DTK 版 PyTorch。
- `third_party/aiter` 是独立 git 仓库。Windows 下它有两个文件名带冒号、无法 checkout 的 Triton autotune JSON，已用 sparse-checkout 排除并设 `core.protectNTFS=false`——**不要在该仓库内执行 `git reset --hard`**。
- 网络：GitHub 直连不通，境外资源走代理 127.0.0.1:16888 或 ghfast.top 镜像；developer.sourcefind.cn / download.sourcefind.cn 直连。
