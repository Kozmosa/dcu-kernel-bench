# 两处 harness 缺陷与修复（2026-10-16）

这两处缺陷让 66 道题里的 **9 道**（1032/5002/5003/5004/5005/5006/5007/5009/6001）
无论 Agent 写什么都必然失败 —— 它们不是 Agent 能力问题。首次全量跑与 HIP 重跑
两轮的未通过集合完全相同、best 全部停在 `root`，即为此故。

**产物位置**：本文件与 `kernelbench_no_grad.patch` 是本仓库里的耐久记录；
修复 1 落在 `../PyramidKernel`（已提交 `77b7453`），修复 2 落在
`../KernelBench-main`（该目录**没有自己的 git 仓库** —— `git rev-parse
--show-toplevel` 会走到主目录，所以以补丁文件为准）。
文中的 `repro_*.py` / `verify_*.py` / `apply_loader_fix.py` 是本地工作脚本，
在 `.dcu_runs/deploy/` 下而该目录被 `.gitignore` 忽略（第 24 行），不入库。

---

## 缺陷 1：脚手架多行签名截断 / 语句块缩进错乱

**位置**：PyramidKernel `pyramidkernel/core/loader.py`（`_extract_signature` /
`_extract_body` / `_extract_forward_steps` / `_extract_class_preamble`）

**根因**：`ast.get_source_segment(node)` 返回的是**整个函数体**的源码，且只把
**首行**切到节点的 `col_offset`（首行缩进被吃掉），其余行保留绝对缩进。两个消费者
因此各错一处：

1. `_extract_signature` 只取 `segment.splitlines()[0]` —— 多行签名只剩头一行：

   ```
   def __init__(self, variant: str = "modulated", stride=(1, 1), padding=(1, 1),:
   ```

   括号未闭合 → 整份脚手架 `SyntaxError`，**从根节点起就编译不过**。

2. `_extract_body` / `_extract_forward_steps` 先 `textwrap.dedent(segment)` 再统一补
   缩进。因为首行缩进为 0，dedent 的公共前缀恒为 0、退化成空操作，补缩进后块内
   层级错乱：

   ```
   91|         if self.variant == "relu":
   92|                     z = torch.relu(z)          <- 20 列
   93|                 elif self.variant == "swish":  <- 16 列  ← IndentationError
   ```

**影响面**：7 道脚手架非法 —— `5003 5004 5005 5007 5008 5009 6001`。
其中 `5008` 的签名本身合法、只有缩进问题，Agent 自行把它改对了（correct，0.81x），
所以最终未通过的是另外 6 道。

**修复**：新增 `_normalized_segment()` —— 把首行被切掉的缩进补回去再 dedent；
`_extract_signature` 改为按括号配平一直读到以冒号结尾的那一行（支持多行签名）。

**验证**：
- `.dcu_runs/deploy/repro_scaffold_bug.py`：修前 **7/10** 脚手架非法 → 修后 **0/10**
- `.dcu_runs/deploy/verify_scaffold_fix.py`：20 道 HIP 题 × 2 backend = **40 份脚手架全部合法**
- 本地全套测试：**297 pass / 0 fail**

**复现/应用**：
```bash
python .dcu_runs/deploy/repro_scaffold_bug.py            # 用真实 loader 复现
python .dcu_runs/deploy/apply_loader_fix.py              # dry-run，报告能否精确匹配
python .dcu_runs/deploy/apply_loader_fix.py --apply      # 写入（自动备份 .bak-<时间戳>）
```

---

## 缺陷 2：评测器 `torch.no_grad()` 与 autograd 型参考实现冲突

**位置**：KernelBench `src/kernelbench/eval.py` 的 `run_and_check_correctness`，
原第 756 行

**根因**：3 道「反向」题的**参考实现本身**就是用 `torch.autograd.grad` 反传求梯度的
（`1032_deform_attention_bwd/reference.py:193-199`，注释明确「reference 不受禁用列表
约束」，禁用列表是写给 Agent 的）。而 `run_and_check_correctness` 把整个正确性循环
——包括 `output = model(*inputs)`（参考实现的 forward）——包在 `torch.no_grad()` 里，
参考实现必炸：

```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

PyramidKernel 的 **root 节点就是这份参考的直通**，所以 root 自己就跑不过 →
整道题无论 Agent 写什么都判负。真机 run.json 里记录的正是
`node=root, stage=runtime, type=builtins.RuntimeError` 且错误字符串逐字相同。

**影响面**：3 道 —— `1032` `5002` `5006`。

**修复**：`with torch.no_grad():` → `with torch.enable_grad():`（只此一行）。

为什么是 `enable_grad()` 而不是直接删掉 `with`：
- 对无 `requires_grad` 张量的前向题**零开销** —— PyTorch 根本不会建图
  （KernelBench 的输入与模型参数都是普通张量）；
- 显式覆盖调用方可能存在的 `no_grad` 上下文；
- 保住了 diff 最小。

为什么不必担心"放开 grad 会纵容 autograd 绕过"：**线上评测本来就不执行禁用列表**
（AGENTS.md：forbidden 在线上仅为 docstring 软约束，硬审计是离线 `static_audit`）。
`no_grad` 提供不了任何反绕过保证，却把 3 道题变成了不可能完成。

**验证**（`.dcu_runs/deploy/repro_nograd_bug.py`，CPU 即可，验的是 autograd 语义）：

| 题 | `torch.no_grad()` | `torch.enable_grad()` |
|---|---|---|
| 1032_deform_attention_bwd | RuntimeError（与真机 root 日志逐字相同） | 通过 |
| 5002_conv2d_bwd | 同上 | 通过 |
| 5006_spconv_bwd | 同上 | 通过 |
| 2012_groupnorm_fwd（前向对照） | 通过 | 通过 |

---

## 部署与复验（**尚未执行**）

两处修复都在本地仓库，未推 git、未部署到 DCU。

```bash
# 1) 修复 1 已落到 ../PyramidKernel/pyramidkernel/core/loader.py
#    修复 2 已落到 ../KernelBench-main/src/kernelbench/eval.py
# 2) 重建 kernelbench.tar —— 该 tar 是 push.ps1 上传的**制品**，不是现场构建的。
#    已核对：tar 内的 eval.py 与 KernelBench-main 版除本改动外逐字节相同，
#    顶层 13 项也完全一致，故从 KernelBench-main 重建不会夹带别的改动。
# 3) 上传（push.ps1 会用 `git archive HEAD` 打包 PyramidKernel，故需先 commit）
# 4) 重跑那 9 道，确认 6 道脚手架题 + 3 道反向题都能建起合法基线
```
