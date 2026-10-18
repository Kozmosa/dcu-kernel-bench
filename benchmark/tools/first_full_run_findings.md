# 首次全量生成评测：结果与harness 缺陷（2026-10-16，DCU BW / gfx936 / DTK 26.04）

## 1. 运行概况

- 题集：`benchmark/kernelbench_compat/manifest_dcukb.yaml`，**66 题**（model_class 全量）
- 配置：`--experiment main`（`deliberation: main` + `search: main`，`max_attempts: 10`）
- 模型：`gpt-6-astra`（hejuapi，经反向隧道 + 本机直连 IPv4 代理）
- 执行方式：**12 片并行**（见第 4 节），2026-10-16 12:06 → 13:29，约 **83 分钟**
- 结果目录：`/root/dcudeploy/runs/dcukb_final/`（已合并 66 个题目录）

## 2. 结果

| 指标 | 数量 |
|---|---|
| 题目总数 | **66** |
| 跑通（有产物） | 66 |
| **correct** | **57** |
| 　其中**真实解（best 非 root）** | **47** |
| 　仅 root（=未改动的基线，不算解） | 10 |
| 未通过 | 9 |

**47 个真实解的 speedup**（这是**在线**口径，分母是评测器内的 torch 参考实现）：

| 统计量 | 值 |
|---|---|
| 中位数 | **6.48x** |
| 均值 | 75.44x |
| 最大 | **1212.39x**（`7001_ctc_loss`） |
| 最小 | 1.21x（`4006_gemm_a16w16`） |

> 报加速比必须说明分母：评测器内分母是 torch 参考实现（`REFERENCE_MODES` 里的
> torch_eager / torch_compile_default / torch_compile_max_autotune 取优）；
> `benchmark/private/*/baseline.json` 里的 aiter 分母要另跑离线终审
> （`benchmark/evaluator/audit_batch.py`）才有。两者不可混用。

「仅 root」的 10 题：`1010`、`1020`、`1026`、`1027`、`2013`、`3009`、`4007`、`4008`、
`4009`、`4017` —— Agent 探索了节点但没能产出比未改动基线更好的解。

## 3. 关键发现：9 道未通过**没有一道是 Agent 能力问题**，全是 harness/题目集成缺陷

### 缺陷 A：scaffold 生成器处理多行代码不完整（6 题）

`ModelNew` scaffold 由 PyramidKernel 的 loader 从 reference 生成。它把 reference 的
`__init__` / `forward` **签名压成一行**、把函数体切成 `forward_stmt_N` 块**按下标拼回**，
但这两步都只按「第一行」处理，多行就坏：

**(1) 多行签名被截断** —— `5005_spconv_fprop` 生成的 scaffold：

```python
    def forward(self, features, filters, indice_pairs, indice_num_per_loc,:
```

reference 的签名本来是多行的，生成器只取了第一行再补 `:` → `SyntaxError`。
同型的还有 `5003`(L61)、`5004`(L78)、`5007`(L85)、`5009`(L90)、`6001`(L65)。

**(2) 语句块只重缩进第一行** —— `5008_conv_bias_act` 的 scaffold：

```python
        if self.variant == "relu":
                    z = torch.relu(z)
                elif self.variant == "swish":      # ← IndentationError
```

`if` 体首行被改成 8 空格，后续行保留原缩进、`elif` 变成 16 空格 → `IndentationError`。

**证据**：每题的 `root` 节点（即**未经 Agent 改动的起点 scaffold**）就是
`node_status=compile_fail`：

```
root.compile_ok = false
failure_type   = builtins.SyntaxError / builtins.IndentationError
logs           = compilation_error=invalid syntax (tmpXXXX.py, line NN)
```

起点就编译不过 → 该题的所有候选必然全挂，与 Agent 无关。

> 反证：`5008_conv_bias_act` 的 scaffold 同样是坏的（IndentationError），但 Agent 把它
> 重写对了，最终 `correct=True, best 非 root, 1.27x`。所以缺陷**不是必然致命**，
> 但会显著压低通过率。

### 缺陷 B：评测器的 `no_grad` 与「用 autograd 定义语义」的 reference 冲突（3 题）

`5002_conv2d_bwd`、`5006_spconv_bwd`、`1032_deform_attention_bwd` 三题的 root 节点都是：

```
node_status  = runtime_fail
failure_type = builtins.RuntimeError
runtime_error= element 0 of tensors does not require grad and does not have a grad_fn
```

这三题的 reference 都靠 **autograd 反传**来定义 dgrad/wgrad（例如 5002 的注释：
「对前向 conv2d 构图后用 autograd 反传：保证 dgrad/wgrad 与前向语义严格一致」）。
而 `KernelBench/src/kernelbench/eval.py` 在 **497 / 560 / 756 行**用
`with torch.no_grad():` 包住参考实现与候选的执行 —— `no_grad` 下
`requires_grad_(True)` 不建图，`torch.autograd.grad` 随即抛错。

**评测器一旦抛错，该题正确性判定整体失败，任何候选都不可能通过。**

### 影响与修法方向

| 缺陷 | 影响题数 | 可能的修法 |
|---|---|---|
| A1 多行签名截断 | 5（5003/5004/5005/5007/5009/6001 共 6 处，其中 5006 是 runtime 另一因） | loader 用 `ast.get_source_segment` 取完整签名，而非按行截断 |
| A2 语句块多行缩进 | 1（5008） | 拼块时按 `ast` 的 `col_offset` 整体重缩进，而非只改首行 |
| B `no_grad` vs autograd | 3（5002/5006/1032） | 评测这两类题时不要包 `no_grad`（或在题面要求 reference 不用 autograd 定义语义） |

**这 9 题合计**：6 属缺陷 A、3 属缺陷 B。修完这两处，理论通过上限可到 66/66。

## 4. 并行执行方法（本轮把 26 小时压到 83 分钟）

`run-kernelbench-batch` 是单进程串行跑 manifest 的，而整条链路是**纯 LLM I/O 等待**
（实测单进程 24 分钟只用 **3 秒 CPU**）。机器有 128 核 / 503 GB 内存 / 64 GB 显存，
并行几乎线性加速。

- `split_manifest.py`：把全量 manifest 轮转切成 N 份（可 `--skip-roots` 跳过已完成的题）
- `run_parallel.sh`：起 N 个进程，各写 `<RUN_ROOT>/par_sNN/`
- `retry_tasks.sh`：对 `summary.json` 里 `status=error` 的题单独重试（本轮 1029/1015
  因 LLM 偶发返回非 JSON 而 error，重试后都成功了）
- `collect_results.py`：把分片产物合并到一个目录 + 汇总（含 baseline 分母标注）
- `shard_status.py`：逐片进度

**踩过的坑**：`seq -w 1 6` 在最大值为个位数时**不补零**（得到 1..6 而非 01..06），
而分片文件名是 `shard_01..06`，导致启动器一个进程都没起来 —— 改用 `printf '%02d'`。

## 5. 复现

```bash
# 全量并行（服务器上）
N=12 RUN_ROOT=/root/dcudeploy/runs/dcukb_par12 bash /root/dcudeploy/run_parallel.sh
# 汇总
python3 /root/dcudeploy/collect_results.py \
  --shard-root /root/dcudeploy/runs/dcukb_par12 --out /root/dcudeploy/runs/dcukb_final --relocate
```

本地留存：`.dcu_runs/run_results/summary_merged.json`、`final_summary.txt`。
