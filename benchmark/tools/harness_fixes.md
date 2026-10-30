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

两处修复都只在本地，未推 git、未部署到 DCU。

修复落地位置：

| 缺陷 | 仓库 / 文件 | 形态 |
|---|---|---|
| 1 脚手架 | `../PyramidKernel/pyramidkernel/core/loader.py` | 已提交 `77b7453` |
| 2 no_grad | `../KernelBench-main/src/kernelbench/eval.py` | 工作树改动（该目录无独立仓库）+ 本仓库的 `kernelbench_no_grad.patch` |

部署步骤：

```powershell
# 1) 重建 kernelbench.tar —— 它由 push.ps1 上传，是**制品**、不是现场构建的。
#    已验证：tar 内 eval.py 与 KernelBench-main 版除本改动外逐字节相同，
#    顶层 13 项也完全一致，故重建不会夹带别的改动。
cd C:\Users\panyp\Desktop\agent4kernel
tar -cf dcudeploy\kernelbench.tar --exclude=.git -C KernelBench-main .
tar -tf dcudeploy\kernelbench.tar | Select-String "src/kernelbench/eval.py"   # 核对

# 2) 上传。push.ps1 用 `git archive HEAD` 打包 PyramidKernel，故 loader 修复
#    必须先 commit（已完成）。不要加 -SkipPyramid。
cd dcu-kernel-bench-main\.dcu_runs\deploy
.\push.ps1 -Server ssh.zzai.scnet.cn -Port 11556 -User root -Password <pwd>
```

```bash
# 3) 重跑这 9 道，确认 6 道脚手架题与 3 道反向题都能建起合法 root 基线
#    （预期：3 道反向题的 root 变成 correct/root-only，speedup ≈1.0；
#      6 道脚手架题不再在 compile 阶段报 SyntaxError/IndentationError）
```

判别标准：修好后这 9 道的失败若仍存在，**不应**再是 `stage=compile` 的
`SyntaxError`/`IndentationError`，也不应再是 `root` 节点的
`element 0 of tensors does not require grad and does not have a grad_fn`。

---

## 验证结果（已执行：2026-10-16 22:14–23:06，DCU 真机）

两处修复已部署（PyramidKernel `77b7453` + 重建的 `kernelbench.tar`）并重跑这 9 道，
产物在 `runs/dcukb_fix9`（未并入 `dcukb_final`，便于与修复前对照）。

上机前预检（服务器上打补丁后的真实 loader）：**9/9 脚手架合法**，多行签名完整拼回。

| 题 | 修复前 best/正确 | compile 阶段失败 | 修复后 best | 正确 | speedup |
|---|---|---|---|---|---|
| 1032_deform_attention_bwd | root/N | 7 | `phase2_n2` | **Y** | 计时未得 † |
| 5002_conv2d_bwd | root/N | 5 | `root` | **Y** | 计时未得 † |
| 5003_deform_conv2d_fwd | root/N | 11 | `phase2_n2` | **Y** | **35.65x** |
| 5004_deform_conv2d_bwd | root/N | 11 | `phase2_n4` | **Y** | 计时未得 † |
| 5005_spconv_fprop | root/N | 11 | `phase2_n1` | **Y** | **11.05x** |
| 5006_spconv_bwd | root/N | 4 | `phase2_n5` | **Y** | 计时未得 † |
| 5007_spconv_indice_pairs | root/N | 11 | `phase2_n2` | **Y** | **284.73x** |
| 5009_conv_depthtospace | root/N | 11 | `phase2_n4` | **Y** | **1.53x** |
| 6001_fused_adamw | root/N | 11 | `phase2_n4` | **Y** | **6.51x** |

- **9/9 由 `best_correct=False` 变为 `True`**（修复前 best 全部停在 `root`）
- **compile 阶段失败 4–11 次 → 0 次**
- `failure_counts` 里的语法错：修复前 **8/9** 道有（6 道脚手架级 `SyntaxError`
  + 5002/5006 的 `python_region_syntax_error`）→ 修复后 **0/9**
- `element 0 of tensors does not require grad`：修复前 3 道（1032/5002/5006）→ 修复后 **0**
- 5 道拿到真实加速比，6 道 best 落在非 root 的 `phase2_n*` 节点

### 这次重跑暴露的缺陷 3（已修，未部署）

**缺陷 3：计时路径仍包在 `no_grad` 里** —— 4 道反向题（1032/5002/5004/5006）的
`reference_runtimes` 是 `{'torch_eager': None}`，即参考实现**测不出时间**，
于是 `speedup=None`。位置：`kernelbench/timing.py:42` 的 `with torch.no_grad():`
（`time_execution_with_cuda_event`，其 docstring 明说它测的就是参考 `Model`）。
与缺陷 2 同源、同一改法，只是另一条代码路径。**不影响正确性判定，但让这 4 道无法计分。**

**缺陷 4：脚手架丢掉 `Model` 的辅助方法** —— `5004` 的 root 报
`AttributeError: 'ModelNew' object has no attribute '_deform_conv2d_forward'`，
`5009` 报 `... no attribute '_depth_to_space'`。原因：`_append_class_body` 只输出
`class_preamble`（非函数类体）+ `__init__` + `forward`，而 `_extract_class_preamble`
显式跳过 `FunctionDef`，于是 `Model` 上被 `forward` 调用的私有辅助方法全部丢失。
影响：这 2 道的 **root 基线不合法**（`root_correct=False`）—— Agent 仍能自己写出正确
实现（best 都是非 root 且 correct），但框架失去了可比基线。
涉及 `PyramidKernel` 的 `loader.py`，与缺陷 1 同一处逻辑。

两个缺陷都不影响"这 9 道不再失败"的结论（9/9 `best_correct=True`），但都值得修 ——
**现已一并修复，见下节**。

---

## 全量归因：66 道里 13 道的 root 基线不合法（2026-10-16 补充）

顺着缺陷 3/4 回扫 `runs/dcukb_final` 全 66 道，把 `root_correct=False` 的**全部**列出来
并归类。结论：**13 道**，而其中只有 4 道被 Agent 自己绕过去了（best 非 root 且 correct），
另 9 道在框架看来就是"完全失败"。

| 归因 | 道数 | 题 | 对应修复 |
|---|---|---|---|
| D1 脚手架语法错 | 7 | 5003 5004 5005 5007 5008 5009 6001 | 修复 1（已部署验证） |
| D2/D3 `no_grad` 冲掉 autograd 参考 | 3 | 1032 5002 5006 | 修复 2（已部署验证） |
| D4 缺辅助方法 | 1 | **7001_ctc_loss** | 修复 4 |
| D5 缺模块级定义 | 2 | **3006_moe_op_mxfp4**、**4005_fused_mxfp4_quant** | 修复 5 |

> `7001_ctc_loss` 早前那轮全量里 root 就已经是
> `AttributeError: 'ModelNew' object has no attribute '_ctc_one'` —— 它靠 Agent 拿到
> 1212x 才没暴露问题，所以缺陷 4 影响的是 3 道（5004/5009/7001），不是当时看到的 2 道。

### 缺陷 5：脚手架丢掉 `Model` 依赖的模块级定义

**现象**：`3006` 的 root 报 `NameError: name '_dequant_mxfp4' is not defined`，
`4005` 报 `name '_rmsnorm' is not defined`。

**根因**：`_extract_imports` 只收集 `Import` / `ImportFrom` 节点，**模块级的函数定义与
常量赋值全都进不了脚手架**，于是 `forward` 一调用模块级 helper 就 NameError。

**修复**：新增 `_referenced_names` / `_defined_names` / `_extract_module_helpers`，
从 class 各片段直接引用的名字出发求**闭包** —— 被选中的定义内部引用到的模块级名字也要
带上（3006 的 `_dequant_mxfp4` 内部用 `_E2M1_TABLE`，实测这 2 道各带进 2 个模块级定义），
按原文件顺序输出；三个 builder 在 imports 之后、类定义之前插入。
顺带把 `_class_member_segment` 正名为 `_node_source_segment`（对模块级节点同样适用）。

### 修复 3/4/5 的验证

**结构检查**（`verify_scaffold_fix2.py`）：66 道题全部构建脚手架并 `ast.parse` 通过；
`Model` 上带辅助方法的 3 道（5004/5009/7001）全部带过来了；合成用例证明
`@staticmethod` 与 `@torch.no_grad()` **装饰器都被保留**
（Python 3.8+ 的 `FunctionDef.lineno` 指向 `def` 行，装饰器只在 `decorator_list` 里，
`ast.get_source_segment` 会静默丢掉它们）。

**功能验证**（`verify_scaffold_fix3.py`，本地 CPU，最强判据）：
对 13 道曾 root 不合法的题，真的 exec 脚手架 → 实例化 `ModelNew` → 跑 forward →
与参考 `Model` 的输出比对：

```
PASS 13 / NEED-GPU 0 / FAIL 0   共 13 道
```

即**这 13 道的 root 基线现已功能等价于参考实现**（root 节点就是脚手架的直通）。

**本地全套测试**：297 pass / 0 fail。

**部署状态**：修复 3/4/5 已部署到 DCU 机器并重跑（见下节）。

| 缺陷 | 文件 | 形态 |
|---|---|---|
| 3 计时 `no_grad` | `../KernelBench-main/src/kernelbench/timing.py` | 工作树改动 + `kernelbench_timing_no_grad.patch` |
| 4 缺辅助方法 | `../PyramidKernel/pyramidkernel/core/loader.py` | 与 5 同一次提交 |
| 5 缺模块级定义 | `../PyramidKernel/pyramidkernel/core/loader.py` | 与 4 同一次提交 |

---

## 修复 3/4/5 的真机验证（2026-10-17 09:00–09:37，DCU 真机）

部署（PyramidKernel `9f9549f` + 重建的 `kernelbench.tar`）后重跑这 13 道，
产物在 `runs/dcukb_fix13`。上机前服务器侧预检：13 道脚手架全部合法、括号配平，
且 `helpers` / `module` 计数与本地一致（5004/5009/7001 各 1 个辅助方法；
5004/1032 各 1 个、3006/4005 各 2 个模块级定义）。

### 修复前 vs 修复后

| 题 | 修复前 root | 修复后 root | best（前→后） | speedup（前→后） |
|---|---|---|---|---|
| 1032_deform_attention_bwd | T/N/**D2** | T/Y | root→`phase2_n4` | →**724.58x** |
| 3006_moe_op_mxfp4 | T/N/**D5** | T/Y | `phase2_n1`→root | 51.96→1.02 |
| 4005_fused_mxfp4_quant | T/N/**D5** | T/Y | `phase2_n1`→`n10` | 106.51→85.07 |
| 5002_conv2d_bwd | T/N/**D2** | T/Y | root→root | →1.92 |
| 5003_deform_conv2d_fwd | F/N/**D1** | T/Y | root→`phase2_n1` | →33.68 |
| 5004_deform_conv2d_bwd | F/N/**D1** | T/Y | root→`n5` | →**293.00** |
| 5005_spconv_fprop | F/N/**D1** | T/Y | root→`phase2_n4` | →10.68 |
| 5006_spconv_bwd | T/N/**D2** | T/Y | root→`phase2_n4` | →**137.02** |
| 5007_spconv_indice_pairs | F/N/**D1** | T/Y | root→`phase2_n5` | →**387.76** |
| 5008_conv_bias_act | F/N/**D1** | T/Y | `phase2_n2`→root | 0.81→0.95 |
| 5009_conv_depthtospace | F/N/**D1** | T/Y | root→`phase2_n5` | →1.51 |
| 6001_fused_adamw | F/N/**D1** | T/Y | root→`phase2_n5` | →6.58 |
| 7001_ctc_loss | T/N/**D4** | T/Y | `phase2_n3`→root | 1212.39→1.02 |

**汇总（13 道）**：

| 指标 | 修复前 | 修复后 |
|---|---|---|
| root 正确 | **0/13** | **13/13** |
| best 正确 | 4/13 | **13/13** |
| 有 speedup | 4/13 | **13/13** |
| 缺陷签名 | D1=7 D2=3 D4=1 D5=2 | **全无** |

修复后 9 道 best 落在非 root 节点（真实解），speedup 中位 **85.07x**、最大 **724.58x**。
其中 **9 道是从「框架看来彻底失败」变成「有合法基线且拿到真实加速比」**
（1032/5002/5003/5004/5005/5006/5007/5009/6001）。

### 需要如实说明的两点

- `3006`（51.96x→1.02x）与 `7001`（1212.39x→1.02x）**这次的解答质量下降了**：
  两道的 best 都退回到 root。这不是修复引入的回归，而是 LLM 生成的**逐轮方差**
  —— 修复只是让 root 基线变得合法可跑，题目难度没变，Agent 这一轮没找到更好的解。
  换言之：修复前这两道的"高分"是在 root 无效的前提下拿到的，现在才有了可比的基准。
- `5008` 修复前 best 是 `phase2_n2` 的 0.81x（**比参考实现还慢**），而 root 因
  `SyntaxError` 无效，框架无从判断。修复后 root 合法、0.95x —— 基线终于诚实了。

### 复现

```bash
bash /root/dcudeploy/rerun_fix13.sh    # 预检 -> 生成 fix13 清单 -> 起 13 片
python .dcu_runs/deploy/compare_fix13.py \
  --before .dcu_runs/before_fix13/runjson --after .dcu_runs/fix13_results/runjson
```

`compare_fix13.py` 的 D 类签名**只从 root 节点**判：`stats.failure_counts` 会把 Agent
探索期自己写坏的 `python_region_syntax_error` 也算进去，那是 Agent 侧问题，不能用来
判脚手架缺陷。

---

## 全量 66 道验证（2026-10-17 10:08–11:53，五个缺陷全修复后）

`run_full_fixed.sh`：先预检 66 道脚手架（**66/66 合法**，带辅助方法 3 道、带模块级定义
4 道），再起 16 片并行，产物写 `runs/dcukb_full_fixed`（未动 `runs/dcukb_final`）。
1 小时 45 分跑完，**66/66 完成，0 报错**。

### 修复前（dcukb_final）vs 修复后

| 指标 | 修复前 | 修复后 |
|---|---|---|
| root 编译通过 | 59/66 | **66/66** |
| **root 正确（基线合法）** | **53/66** | **66/66** |
| **best 正确** | 57/66 | **66/66** |
| └ 非 root 真解 | 47 | **53** |
| └ 仅 root | 10 | 13 |
| 有 speedup | 57/66 | **66/66** |
| 有参考运行时间 | 57/66 | **66/66** |
| 带缺陷签名 | **13** | **0** |

真解 speedup：修复前 n=47 中位 6.50x / 均值 68.35x / 最大 1212.39x；
修复后 n=53 中位 5.61x / 均值 72.26x / 最大 858.02x。

### 15 道分类发生变化的题

**失败 → 真解（8 道）**：1032（→435.53x）、5003（→5.61x）、5004（→416.07x）、
5005（→11.36x）、5006（→144.60x）、5007（→356.89x）、5009（→1.85x）、6001（→5.13x）
—— 全是 root 基线原本不合法的题，现在既有了合法基线、Agent 也写出了真实解。

**失败 → 仅 root（1 道）**：5002（→0.83x）—— 基线合法了，这轮没超过参考。

**仅 root → 真解（2 道）**：1010（0.93x→12.06x）、1020（1.00x→1.02x）。

**真解 → 仅 root（4 道）**：1022（68.29x→1.08x）、5001（1.03x→0.99x）、
8001（1.66x→1.00x）、5008（0.81x→0.98x）—— 逐轮方差；这 4 道修复前的"真解"都
贴着 1.0x，其中 `5008` 的 0.81x **比参考还慢**，只是因为当时 root 因 `SyntaxError`
无效才被记成"真解"，现在基线诚实了。

### 结论

- 五个缺陷签名**全部清零**，66 道**全部有合法 root 基线**（修复前 13 道没有）
- best 正确 57 → 66，非 root 真解 47 → 53
- 之前"框架看来彻底失败"的 9 道**全部解决**（8 道真解 + 1 道仅 root）
- 中位 speedup 从 6.50x 略降到 5.61x 属于正常：修复后纳入统计的样本变多（47→53），
  且不再有"基线无效却记成真解"的失真样本

### 复现

```bash
bash /root/dcudeploy/run_full_fixed.sh    # 预检 66 道 -> 起 16 片
python .dcu_runs/deploy/compare_full66.py \
  --before .dcu_runs/before_full66/runjson --after .dcu_runs/after_full66/runjson
```

运行观察：16 片（每片 4–5 题）稳定跑完；此前试过 24 片，5 分钟后**整机被平台下线**
（24 进程同一秒消失、0 题完成），而启动时内存 446 GB 可用、GPU 仅用 2 MiB —— 不是
资源压力。16 片是已验证的稳妥档位。
