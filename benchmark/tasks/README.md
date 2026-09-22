# benchmark/tasks/ — 任务定义层（Agent 可见）

每个子目录是一道独立任务，**只放允许 Agent 看到的内容**：

| 文件 | 作用 |
|---|---|
| `task.yaml` | 任务规格：语义、dtype/shape/layout 范围、容限、禁调用项 |
| `reference.py` | 独立 PyTorch（或 CPU）参考实现，语义的唯一依据 |
| `public_cases.json` | 少量公开测试案例（shape/dtype/生成方式，不含隐藏数据） |
| `starter/` | 统一 Starter 工程，Agent 只补全 kernel 文件与 launch 逻辑 |

规则：

- 任务描述只引用 `reference.py` 的语义，不得粘贴 aiter/官方实现代码。
- `task.yaml` 的 `forbidden` 列表会被 `evaluator/static_audit.py` 强制执行。
- 隐藏正确性/边界/性能案例在 `benchmark/private/<id>/`，不进本目录。
