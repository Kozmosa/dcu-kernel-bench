# benchmark/private/ — 隐藏评测资产（仅评测端）

**严禁进入生成 Agent 的上下文。**

每个任务一个子目录：

| 文件 | 内容 |
|---|---|
| `hidden_cases.json` | 隐藏正确性案例（含边界、非对齐、极值输入描述与种子） |
| `perf_cases.json` | 性能计时输入（预热/重复次数可覆盖 environment.yaml 默认值） |
| `baseline.json` | 官方实现（aiter / DTK 版 PyTorch）在同机同环境下的性能基线 |
| `gold_source/` | 官方实现快照（可选，仅作性能上界溯源） |

规则：任一隐藏正确性 Case 失败 → 不进入性能评分；基线必须与生成实现同机同环境测量。
