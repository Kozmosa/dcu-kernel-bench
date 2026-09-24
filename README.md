# dcu-kernel-bench

面向曙光/海光 DCU（HCU，gfx936/gfx938）的算子生成与评测集。参照寒武纪 BANG C 算子评测集的方法论，以 [OpenDAS/aiter](https://github.com/ROCm/aiter)（HCU 适配的开源算子库）为主要算子来源，构建「任务定义 → Agent 生成 HIP/Triton kernel → 静态审计 → 编译 → 正确性 → 性能」的完整评测闭环。

## 分层结构

```
原始算子层      third_party/aiter/        # git submodule，仅维护者审阅，不给 Agent
       │
来源审核层      operator_catalog.yaml      # 全量算子登记
               benchmark/sources/         # 准入审核记录：commit、LICENSE、依赖、证据哈希
       │
任务定义层      benchmark/tasks/<id>/      # Agent 可见：task.yaml / reference.py /
               │                          # public_cases.json / starter/
       │       ┌────────────────┐
参考验证层     reference.py + 公开案例      Agent 生成层   evaluator/generate.py
               │                          静态审计        evaluator/static_audit.py
       └───────┴───────────────┘
统一 Runtime 评测层  benchmark/evaluator/   # hipcc 编译、正确性、稳定性、性能计时
```

## 目录说明

| 目录 | 内容 | 可见范围 |
|---|---|---|
| `third_party/aiter/` | 原始算子源码快照（git submodule 注册，不 vendor 源码） | 仅维护者 |
| `operator_catalog.yaml` | 算子登记：来源、语言、核心计算位置、准入状态 | 维护者 |
| `benchmark/sources/` | 准入审核记录（commit / LICENSE / 依赖 / 证据哈希） | 维护者 |
| `benchmark/tasks/<id>/` | 任务规格、参考实现、公开案例、Starter 工程 | **Agent 可见** |
| `benchmark/private/<id>/` | 隐藏正确性/边界/性能案例、官方基线 | 仅评测端 |
| `benchmark/evaluator/` | 静态审计、编译、正确性、性能计时、Agent 调用 | 评测服务 |
| `environment.yaml` | 硬件型号与软件版本锁定 | 公开 |

**关键原则**：aiter 官方实现只作语义依据和性能上界，不作为 Agent 输入；Agent 仅根据 `tasks/<id>/` 中的公开材料重新实现 kernel。

## 任务赛道

- **HIP C++ track**（`impl_lang: hip`）：Agent 补全 `starter/kernel.hip`，hipcc 编译，目标 gfx936/gfx938。
- **Triton track**（`impl_lang: triton`）：Agent 补全 `starter/kernel.py`，要求 DTK ≥ 25.04。

## 评测管线

静态审计 → 编译 → 公开 Case → 隐藏 Case → 边界 Case → 性能计时（预热 + 重复取中位数）。任一隐藏正确性 Case 失败则不进入性能评分。核心指标：Compile Rate / Correctness / Speedup（对 aiter 官方实现）。

详见各目录下的 README 与 [AGENTS.md](AGENTS.md)。

## 本地开发环境（uv）

本仓库用 [uv](https://docs.astral.sh/uv/) 管理 Python 开发环境（`pyproject.toml` + `uv.lock`）：

```bash
uv sync        # 创建 .venv 并按 lockfile 精确复现依赖
uv run pytest  # 运行本仓库测试（reference 语义 / case 一致性 / 静态审计回归）
```

- `torch` 固定来自 PyTorch 官方 **CPU 轮子源**（见 `pyproject.toml` 的
  `[[tool.uv.index]]`，`explicit = true` 只对 torch 生效）——开发机上只做
  reference 与评测逻辑验证，不拉 CUDA/DCU 巨型依赖。
- DCU 真机（gfx936/gfx938）运行环境仍以 `environment.yaml` 锁定为准
  （DTK ≥ 25.04、DTK 版 PyTorch/Triton）；`uv` 环境不用于性能评测。
- `tests/` 只覆盖本仓库自身资产；`third_party/aiter` 子模块的测试需要
  ROCm/DCU 环境，已在 pytest 配置中显式排除，不会被收集。
