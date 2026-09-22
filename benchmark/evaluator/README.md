# benchmark/evaluator/ — 统一 Runtime 评测层

评测管线（固定顺序，任一阶段失败即终止）：

```
static_audit.py   静态审计：禁调用项、外部库符号、文件白名单
       │
compile           hipcc 编译（hip track，flags 见 environment.yaml）/ Triton JIT 预热
       │
correctness       公开 Case → 隐藏 Case → 边界 Case（对 reference.py 输出，容限见 task.yaml）
       │
performance       预热 + 重复取中位数；与 private/<id>/baseline.json 同机比较
       │
report            输出 Compile / Correctness / Speedup，写入 results/
```

脚本：

| 脚本 | 状态 | 作用 |
|---|---|---|
| `static_audit.py` | 可用 | 扫描提交文件中的禁调用符号（CPU 即可运行） |
| `run_eval.py` | 骨架 | 串起完整管线；compile/correctness/performance 需在 DCU 真机补全 |

核心指标：Compile Rate（成功编译任务比例）、Correctness（通过全部隐藏 Case 比例）、Speedup（基线延迟 / 生成实现延迟）。
