# 上游 KernelBench L1/L2 在 DCU 上跑（A 方案：上游语义）

> 目的：把**上游 KernelBench 官方 L1/L2（各 100 题）**接到 KernelWeaver 上在 gfx928 上跑，
> 用于和 KernelBench 官方榜单口径对齐。
> **本页是 A 方案**：不启用 DCUKB 的 1116 在线契约，回到**上游 KernelBench 语义**。
> 若要和本仓库 66 题的表并列（含静态审计与真解门槛），走 B 方案（onboard 成 dcukb 任务，另文）。
> L1 结果见 §9，L2 结果见 §10，两级并排见 §11。

---

## 1. 与 DCUKB 66 题的口径差异（不要混表）

| 维度 | 本页（A，上游语义） | 本仓库 66 题（1116 契约） |
|---|---|---|
| 输入 dtype | 统一 cast **fp32** | 原生 dtype（含 int32 索引） |
| 正确性 | `allclose(atol=rtol=1e-4)`，单一 `get_inputs()` shape | `task.yaml` 分档容差 + 分段容差 |
| 执行前静态审计 | **无** | 有（禁用算子 + 平台符号） |
| oracle root | 可以当冠军（`collect_metrics` 默认排除 root 后再统计） | 框架层排除，不导出 |
| 离线终审 / 真解门槛 | **没有**（上游题没有 task.yaml / 隐藏 case） | 有 |
| 题面硬件提示 | 上游 docstring（不含 gfx928 提示） | 题面写明 DCU 目标 + 注入公开契约 |
| DCU Wiki | **仍然生效**（按 `reference_code` 的 API 符号检索，与 `DCUKB_ROOT` 无关） | 生效 |

> 因此：`--max-attempts 1` 时最接近 KernelBench 官方 pass@1；`10` 时是"给框架 10 次机会"的口径，
> 两者不可互换引用。

## 2. 前置条件（机器上）

| 项 | 要求 |
|---|---|
| KernelBench 源码 | `/root/KernelBench-main`（含 `KernelBench/level1/` 100 个 `.py`） |
| kernelbench 包可导入 | `site-packages/kernelbench_src.pth` → `/root/KernelBench-main/src`（否则 `paper_evaluator_import_error`） |
| 框架 | `/root/KernelWeaver`（`cd5d85c` + 两个补丁序列，见 `docs/machine-2026-11-21-k500sm-gfx928.md`） |
| LLM 出口 | `/root/llm.env`（含 `OPENAI_*` 与反代 `HTTPS_PROXY`），反向隧道 `-R 17890` |
| HIP 编译 | `PYTORCH_ROCM_ARCH=gfx928`（否则每个候选按 5 个架构编译，慢 5×） |
| 评测超时 | `STARK_PAPER_EVAL_TIMEOUT_SECONDS=900`（240s 对 hipcc 太紧） |
| int-cast 补丁 | `eval.py::_process_input_tensor` 与 `timing.py::measure_ref_program_time` 两处改成"只 cast 浮点张量"，否则带整数索引的题（如 `95_CrossEntropyLoss`）恒判负、且拿不到 speedup（详见 §9.2） |

## 3. 一条命令开跑

```bash
cd /root/dcukb
bash benchmark/tools/launch_kernelbench.sh
```

默认：**L1 + Triton 单轨 + 8 分片 + 10 attempts**，输出 `/root/runs/kb_l1/`，清单 `/root/manifests/kb_l1/`。

常见变体：

```bash
TRACKS=triton,cuda bash benchmark/tools/launch_kernelbench.sh   # 双轨（HIP 编译慢，建议先 Triton）
ATTEMPTS=1             bash benchmark/tools/launch_kernelbench.sh   # 对齐官方 pass@1
LEVEL=2                bash benchmark/tools/launch_kernelbench.sh   # 换 level
SHARDS=10              bash benchmark/tools/launch_kernelbench.sh   # 加并发
```

脚本做四件事：① `unset DCUKB_ROOT`（**A 方案的关键**，设了会 `dcukb_contract_error`）；
② 生成清单（缺了才生成）；③ 后台起 `benchmark/tools/supervise_shards.py`；
④ 打印 `supervisor_status.json` 与在跑进程。

## 4. 清单与产物

```
/root/manifests/kb_l1/
├── l1_triton_all.json / l1_triton_s0..s7.json      # 100 题，按 problem_id 轮转切 8 片
└── run_kn_l1_triton_s0.json ...                    # 看护器每轮写的"剩余题"清单

/root/runs/kb_l1/
├── kn_l1_triton_s0/<alias>_l1_p<pid>/{run.json,best_code.py}
├── logs/kn_l1_triton_s0.log
├── pools/memory_pool.json                          # 全新目录 = 全新记忆
└── supervisor_status.json                          # 每 90s 刷新：running/done/total/restarts
```

## 5. 看指标

```bash
# 在线指标（默认排除 root 候选；与 66 题同一套定义）
python3 benchmark/tools/collect_metrics.py --run-root /root/runs/kb_l1_flat

# 零候选 / 有候选但不正确 / provider 错误分开列名
python3 benchmark/tools/audit_zero_candidates.py --root /root/runs/kb_l1
```

注意：`collect_metrics.py` 需要"一题目录"这一层，而分片产物是"分片→题目"两层，
所以统计前先拍平（硬链接，不动原数据）：

```bash
python3 benchmark/tools/flatten_run_root.py \
  --run-root /root/runs/kb_l1 --out /root/runs/kb_l1_flat --shards "kn_l1_triton_s*"
# 拍平 100 题 -> /root/runs/kb_l1_flat
```

## 6. 看护器的行为（为什么能直接放着跑）

`benchmark/tools/supervise_shards.py`（本轮重写）：

- **完成判据 = 已完成 `run.json` 数 == 清单题数**。框架在 provider 预检失败时会 abort 整批并**照常写
  `summary.json`**，所以不能用 `summary.json` 判完成（上一轮因此白等 8 小时）。
- **进程存活 = pid 文件 + `/proc/<pid>/cmdline` 双重校验**：只看 `os.kill(pid,0)` 会被 pid 复用骗到。
- 每轮把**剩余题**写进 `run_<name>.json` 再起，已完成的题不重跑。
- 日志超过 `STALL_MIN`（默认 50 分钟）没更新且进程还在 → 判卡死，kill 后重拉（`MAX_RESTARTS` 次上限）。

本机已做的离线自检（`WIRING PASS`）：① loader 的 `glob "<pid>_*.py"` 在 100 题上不串题（1 / 10 / 11 / 100 互不冲突）；
② `summary.json` 不被当完成；③ 死 pid 与 pid 复用都判"没在跑"。

## 7. 已知限制（引用结果时必须说明）

1. **没有离线终审**：上游题没有 `task.yaml` 与隐藏 case，所以拿不到"真解（在线 ∩ 离线）"，
   也抓不到"在线侥幸通过、离线编不过"的情况（本仓库 66 题里后者有 7–15 题）。
2. **没有执行前静态审计**：候选可以内部调 `torch.matmul`/`F.conv2d` 而不被判负——这属于上游语义的原貌。
3. **root 可以当冠军**：`collect_metrics.py` 默认把 root 排除在候选池外，但 `run.json` 里
   `best_node_id="root"` 的题要在报告里单独标注（这些题等于"框架没产出超过脚手架的候选"）。
4. **题面没有 gfx928 硬件提示**：L1 docstring 是上游原文；DCU Wiki 仍在上下文里。
5. 纯 API 臂（`pure_bench.py`）在 A 方案下**不能直接跑**——它硬性要求 `DCUKB_ROOT`，
   设了就会走 1116 桥并要求 `task.yaml`。要出纯 API 基线就得上 B 方案，或放宽该检查。

## 8. 预计耗时（按本轮实测速率）

- 准备：0（脚本已就绪，清单自动生成）
- Triton 单轨、8 分片、10 attempts：单体调用 ~2–4 分钟/次、每题最多 10 次 → **约 4–7 小时**
- 若 `ATTEMPTS=1`：**约 40–60 分钟**
- 双轨会把 HIP 编译开销叠上来（本轮 HIP 轨修复后约 21 题/小时），建议先只跑 Triton。
- **实测（2026-11-26 首轮）**：L1 双轨 200 个 task-run、8 分片、10 attempts → **约 10.5 小时**（≈19 个 task-run/小时），
  另有 16 个 task-run 的串行补跑约 2 小时。

---

## 9. 首轮实测结果（2026-11-26，L1 双轨，200 个 task-run）

分母 100 题/轨；`collect_metrics.py` 口径（**排除 root 候选**，分母 `torch_eager`，10 attempts）。

| 指标 | Triton 轨 | HIP（cuda）轨 |
|---|---|---|
| Compile Rate（模块级） | 100/100（100%） | 99/100（99%） |
| Compile Rate（kernel 级） | 100/100（100%） | 99/100（99%） |
| Correct Rate | **100/100（100%）** | **99/100（99%）** |
| Success@1 | 52/100（52%） | 71/100（71%） |
| Success@5 | 93/100（93%） | 96/100（96%） |
| Success@10 | 100/100（100%） | 99/100（99%） |
| Fast1@1 | 16/100（16%） | 22/100（22%） |
| Fast1@5 | 64/100（64%） | 41/100（41%） |
| Fast1@10 | 84/100（84%） | 51/100（51%） |
| Fast1@all | **84/100（84%）** | **51/100（51%）** |
| Geomean Speedup | **1.82×** | **1.17×** |
| Median Speedup | 1.40× | 1.08× |
| Mean Speedup | 3.96× | 2.29× |
| Best Speedup | 79.78× | 18.42× |
| First Correct Median | 1.0 | 1.0 |
| Refine Gain Median | 1.29× | 1.99× |

### 9.1 结论

1. **正确性上 L1 被做穿了**（Triton 100/100、HIP 99/100），唯一的真实失败是
   HIP 轨 `97_ScaledDotProductAttention`：root 正确，但 10 个候选全部 `cuda_compile_error`
   （`(32,32,512,1024)` 的三输入 attention 用 HIP C++ 手写，工程难度确实高）。
2. **速度很差，且这是 L1 的固有属性**：Triton 中位 1.40×、HIP 中位 1.08×，Fast1@all 84% vs 51%。
   与 dcukb 66 题（HIP geomean 7.41×、Triton 6.66×）差一个数量级——L1 全是 launch-bound 小算子，
   eager 本来就贴着下限，能抢的空间很小。引用时不要把这两套数字放在同一张表里比"强弱"。
3. **HIP 轨在 L1 明显吃亏**：同一批题 Triton 84% 能快过 eager，HIP 只有 51%；geomean 1.82× vs 1.17×。
   原因是多一层 pybind + `load_inline` 调用开销，而 L1 的算子里程碑极短，开销占比被放大。
4. **`Success@1` 只有 52%/71%**，但 `Success@10` 接近 100%——框架的价值在迭代补救，不在第一枪。

### 9.2 上游 KernelBench 的两个缺陷（本轮踩到并修了，与 DCU 无关）

| # | 位置 | 症状 | 修法 |
|---|---|---|---|
| 1 | `src/kernelbench/eval.py` 的 `_process_input_tensor` | 无条件 `input.to(dtype=precision)`，把**整数张量**（类别索引 / mask）也转成 fp32 → `95_CrossEntropyLoss` 的 root 自身抛 `NotImplementedError: "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Float'`，整题两轨对任何候选恒判负 | `input.to(dtype=precision) if input.is_floating_point() else input` |
| 2 | `src/kernelbench/timing.py` 的 `measure_ref_program_time` | 同一处错误在**参考实现计时**里，`inputs`/`init_inputs` 全量 cast → 参考耗时抛异常，`speedup` 记成 `None`（`95` 类题即使算对也没有加速比） | 同上：只 cast 浮点张量 |

两条在 NVIDIA 上同样会触发（不是 ROCm 特有），MLU 那份移植也没修。补丁只改 `if` 条件，不改变浮点张量的行为。

### 9.3 显存：OOM 是资源撞车，不是能力

L1 有一批题输入极大（`batch_size=4096, dim=393216` → 单张量 fp32 就 **6.4 GB**，参考+输出+中间量要 20~25 GB）。
8 分片并发时一旦两三个这样的题撞在一起就 OOM，报错形如
`HIP out of memory … 0 bytes free … 12.00 GiB is allocated by PyTorch`。

- 首轮 200 个 task-run 里 **192 个候选**被 OOM 打掉过；其中 6 道题因此 0 正确。
- **串行（1~2 分片）补跑后，这 6 道全部 correct**（0.25×~3.76×）——证明是环境问题。
- 另一个独立原因是 ROCm 的 caching allocator **不归还显存**：同一进程连跑多题，VRAM 一路涨到打满
  （实测 3 个分片占 61 GiB，而四个 python 进程 RSS 各只 2.4 GiB）。

因此本轮给看护器加了两个开关，跑 L1/L2 都应当打开：

```bash
MAX_CONCURRENT_SHARDS=8      # 同时活着的分片上限（避免把卡打满）
TASKS_PER_LAUNCH=1           # 每题换一个进程，显存复位
```

### 9.4 产物与复现

```
/root/runs/kb_l1b/                  # 主跑（200 个 task-run）
/root/runs/kb_l1_fix/               # 补跑①：环境失败 8 个 task-run（OOM + int-cast）
/root/runs/kb_l1_fix2/              # 补跑②：参考计时失败、重测速度 8 个 task-run
/root/runs/kb_l1_final_{triton,cuda}/   # 合并后的干净产物（后两轮覆盖前一轮）
/root/runs/kb_l1_final_{triton,cuda}.md # collect_metrics 报告
```

合并 + 出表的完整命令：

```bash
# 合并：以 kb_l1_fix2 > kb_l1_fix > kb_l1b 的优先级覆盖，按 track 落成 <alias>_l1_p<pid>/run.json
python3 /root/dcukb/benchmark/tools/collect_metrics.py \
  --run-root /root/runs/kb_l1_final_triton --out /root/runs/kb_l1_final_triton.md
python3 /root/dcukb/benchmark/tools/collect_metrics.py \
  --run-root /root/runs/kb_l1_final_cuda --out /root/runs/kb_l1_final_cuda.md
```

本地留档（仓库外）：`dcukb-results-20261126-l1/`（合并产物 + 两份 md + 三轮看护器日志）。

---

## 10. L2 实测结果（2026-11-28，双轨，200 个 task-run）

L2 是**算子融合**（Gemm/Conv + Norm + 激活 + 池化 的串联）。分母 99/99：**剔除了
`66_Matmul_Dropout_Softmax`**——它的参考实现用 `nn.Dropout(0.2)`（训练模式），两次 forward 的
mask 不同，root 直通自己都对不上，属**不可判**（上游语义下的题级缺陷，见 §10.2）。

| 指标 | Triton 轨 | HIP（cuda）轨 |
|---|---|---|
| Compile Rate（模块级） | 99/99（100%） | 98/99（99%） |
| Compile Rate（kernel 级） | 99/99（100%） | 98/99（99%） |
| Correct Rate | **99/99（100%）** | **98/99（99%）** |
| Success@1 | 62/99（62.6%） | 69/99（69.7%） |
| Success@5 | 98/99（99.0%） | 95/99（96.0%） |
| Success@10 | 99/99（100%） | 98/99（99.0%） |
| Fast1@1 | 33/99（33.3%） | 12/99（12.1%） |
| Fast1@5 | 72/99（72.7%） | 26/99（26.3%） |
| Fast1@10 | 83/99（83.8%） | 40/99（40.4%） |
| Fast1@all | **83/99（83.8%）** | **40/99（40.4%）** |
| Geomean Speedup | **1.54×** | **0.66×** |
| Median Speedup | 1.25× | 0.83× |
| Mean Speedup | 12.29× | 7.06× |
| Best Speedup | 723.63× | 533.68× |
| First Correct Median | 1.0 | 1.0 |
| Refine Gain Median | 1.34× | **3.40×** |

若把 `66` 也计入分母（100/100），则是 Triton `correct 99/100`、HIP `98/100`、`compile 99/100`。

### 10.1 结论

1. **正确性同样做穿了**：Triton 99/99、HIP 98/99。融合题对框架不是难点——难点全在"融合得好不好"。
2. **HIP 轨在 L2 上整体慢于 eager**：geomean **0.66×**、中位 **0.83×**、只有 40% 的题跑赢 eager；
   同一批题 Triton 是 1.54× / 1.25× / 84%。这是 L2 最值得注意的一条：多算子链路里，HIP C++ 的
   每次 launch + pybind 开销被放大到"抵消甚至超过"融合收益。
3. **但 HIP 的 Refine Gain 是 3.40×**（Triton 1.34×）：说明 HIP 候选的第一枪普遍很差（`Fast1@1` 只有 12%），
   框架靠后续迭代把它拉起来——这条正是"框架臂 vs 单发 API"的价值所在。
4. **`Success@1` 62.6%/69.7% vs `Success@10` 100%/99%**：10 次尝试的预算是这类题的必要条件。

### 10.2 L2 的两个失败（性质不同，不要混）

| 题 | 现象 | 判定与处理 |
|---|---|---|
| cuda `30_Gemm_GroupNorm_Hardtanh` | root 正确（1.0×），10 个候选里 **7 个 `cuda_compile_error`**（扩展构建失败）；Triton 轨解得 `0.769×` | **真实失败**（HIP 侧写不出能编的 kernel），保留在分母 |
| `66_Matmul_Dropout_Softmax`（两轨） | **root 直通也判负**，`correctness_error`：`max_difference ≈ 3.1~3.8e-4`、`avg_difference ≈ 6e-6`（fp32 容差 `1e-4`） | **不可判**：参考实现含训练态 `nn.Dropout`，两次 forward 的 mask 不同；从分母剔除 |

`66` 的机制（逐条都有数据支撑）：

- 参考实现是 `Linear(16384→16384) → nn.Dropout(0.2) → softmax(dim=1)`，输入 `128×16384`，
  权重 16384² ≈ **1.07 GB**；`nn.Module` 默认 `training=True`，dropout 生效。
- 评测器只在每次 trial 前 `set_seed(trial_seed)`，随后"参考跑一次、候选跑一次"，**两次 forward
  之间不重置 RNG** → 两个 mask 不同。
- 差异刚好卡在容差线上：`max_difference` 3.1~3.8e-4（> 1e-4 判负），`avg_difference` 6e-6。
  原因是 softmax 在 16384 类上归一化，单个被丢掉的元素只挪动 1e-4 量级。
- **最硬的证据**：`root` 节点（脚手架直通＝与参考逐字节相同的代码）也判负，且它的三个 trial 的
  `max_difference`（0.000322 / 0.000310 / 0.000380）与多个候选**完全一致**——差异来自 mask，不来自实现。
- 反过来也说明它测不出东西：若容差是 KernelBench 早期默认的 `1e-2`，这道题"什么都不做"也能通过。
- 三种修法的取舍：① `model.eval()` → 变成推理态，丢掉训练态考点；② **每次 forward 前重置种子** →
  对手写 kernel 无效（kernel 不消费 PyTorch 的 RNG），仍然对不上；③ 把参考改成确定性等价版本
  （p=0 或固定 mask）→ 可判，但属"改题目"。故本轮剔除并记录，不擅自改题。

`30`（L2 HIP）与 `97_ScaledDotProductAttention`（L1 HIP）属同一类：**扩展构建阶段挂掉**，
且都出现在"通用 / 模板化 fallback kernel"的写法上（`97` 用带 `Metadata{shape[64], strides}` 的通用
kernel；`30` 用 `scalar_t`/`acc_t` 模板 + `ATen/AccumulateType.h`）。两者在 Triton 轨都能过
（`0.247×` / `0.769×`），所以不是题不可解，而是 C++ 侧的工程细节——但**存档里拿不到编译器原文**：
框架把 `runtime_error` 截断到约 200 字符（只剩 `Error building extension … [1/3] ninja …`）。
要定位到具体一行，需用 `run.json` 里保存的候选代码在机器上重放 `load_inline` 并保留 ninja/hipcc 输出。

### 10.3 产物

```
/root/runs/kb_l2/                    # 主跑 200 个 task-run
/root/runs/kb_l2_final_{triton,cuda}/   # 100/100，含不可判题
/root/runs/kb_l2_n66_{triton,cuda}/     # 99/99，剔除 66（上表口径）
```

本地留档（仓库外）：`dcukb-results-20261127-l2/`（189 题快照 + 产物 + 补丁文件）；
最终 200 题产物在机器 `/root/runs/kb_l2/`。

---

## 11. L1 + L2 并排（同一口径、同一机器）

| 指标 | L1 Triton | L1 HIP | L2 Triton | L2 HIP |
|---|---|---|---|---|
| 题数（分母） | 100 | 100 | 99 | 99 |
| Correct | 100（100%） | 99（99%） | 99（100%） | 98（99%） |
| Fast1@all | 84（84%） | 51（51%） | 83（84%） | 40（40%） |
| Geomean | 1.82× | 1.17× | 1.54× | **0.66×** |
| Median | 1.40× | 1.08× | 1.25× | 0.83× |
| Best | 79.78× | 18.42× | 723.63× | 533.68× |
| Refine Gain 中位 | 1.29× | 1.99× | 1.34× | 3.40× |

三条跨 level 的稳定规律：

1. **Triton 全面优于 HIP**（geomean 1.82/1.54 vs 1.17/0.66），且 level 越高差距越大——HIP 的
   launch + pybind 固定开销在"算子多、单算子小"的场景里最吃亏。
2. **正确性与优化是两件事**：两级都是 99~100% 正确，但 Fast1 从 84% 掉到 40%（HIP 侧）。
3. **层级越高，迭代收益越大**（RG：L1 1.3~2.0× → L2 1.3~3.4×）——融合题的搜索空间更大，
   框架的 Plan/Debug 循环才有发挥余地。
