# aiter 官方实现适配器登记

离线终审与基线采集都依赖 `benchmark/private/<id>/aiter_impl.py`：它把题面的
（`make_inputs` 返回顺序 + `io` 规格）绑定到 aiter 官方算子的（入口签名 + layout 约定）。
**上游 47 道题一个适配器都没有**，本批一次性补齐。

## 覆盖

| 项 | 数 |
|---|---|
| model_class 题 | 45 |
| **已有适配器** | **44**（含原有 1002） |
| 缺 | 1（`4007_gemm_a16w16_atomic`，见下） |
| 号段分布 | 1xxx: 18 ｜ 2xxx: 9 ｜ 3xxx: 7 ｜ 4xxx: 10 |

## 真机采集结果（BW / gfx936，DTK 26.04）

45 道 model_class 题**全部采到至少一种基线**，`benchmark/private/<id>/baseline.json`
已从 placeholder 换成实测值：

| 覆盖面 | 题数 |
|---|---|
| `aiter` + `torch_eager` 双基线 | 27 |
| 仅 `torch_eager` | 18 |
| 仅 `aiter` | 0 |
| **无基线** | **0** |

- **分母必须说清**：`record_baseline.py --impl {aiter,eager}` 按 `impl` **合并**写入，
  同一条 case 可以同时有两条基线。两者量级差别很大（例：`1017_mha` aiter 57 ms vs
  `torch_eager` 180 ms，约 3.2x）。`audit_model_class.py` 的 `stage_perf` 会带回
  `impl` / `baseline_us` / `speedup`，**报 speedup 时必须说明用哪个 `impl` 当分母**，
  否则数字没有意义。
- **`1019_mha_onekernel_bwd` 的 aiter 基线不可得**：DTK 版 aiter 的 onekernel bwd 在本机
  `PassManager::run failed`（shared memory 81920 > 65536，超出 DCU 上限），**不是配置问题**，
  无法绕过。该题只有 `torch_eager` 基线。
- **6 道题的 aiter 基线用的是 MI 系列兜底配置**（找不到 `BW200-*` tuner 配置时入口回退），
  未针对 BW200 调优 → **分母偏大 → 这些题的 speedup 会被高估**，引用时必须标注：

  | 题 | 缺的 BW200 配置 | 程度 |
  |---|---|---|
  | `1014_hstu_attention` | `hstu_attn/BW200-HSTU_ATTN_FWD.json` | 全部 case |
  | `1017_mha` | `BW200-MHA-DEFAULT.json` | 全部 case |
  | `4001_batched_gemm_a8w8` | `gemm/BW200-BATCHED_GEMM-A8W8.json` | 全部 case |
  | `4004_batched_gemm_bf16` | `gemm/BW200-BATCHED_GEMM-A16W16.json` | 全部 case |
  | `4006_gemm_a16w16` | `gemm/BW200-GEMM-A16W16.json` | 全部 case |
  | `4008_gemm_a16w4` | awq_w4a16 的 `BW200...` 配置（3 case 缺 1） | 部分 case |

  量级旁证：`4006` 的 `perf_llm_mlp_4864_4096_8192_bf16` 录到 **63.7 ms**，对
  m4864·n4096·k8192 的 bf16 GEMM 而言明显异常慢。其余已采题不依赖 tuner 配置或配置齐备。
- `1017` 与 `1019` 的适配器改完打包协议后已在新协议下重新采集，`1017` 三条 aiter case
  真机判 `[OK]`（证明打包对齐）；`1019` 的 `torch_eager` 三条也补齐。

## 适配器契约

```python
def run(inputs, init_kwargs: dict, device) -> tuple[torch.Tensor, dict]:
    """inputs : reference.py::make_inputs(**case字段) 的返回列表，元素已在 device 上
       init_kwargs : 该 case 的构造参数（按名），即 case_init_kwargs 的产物
       返回 (out, ctx)：out 必须与任务 reference 输出**同形同 dtype 的单个张量**
    """
```

- 调用方 `record_baseline.py` 会拿 `out` 与 reference 输出按 `task.yaml` 容差比对，
  **不一致就拒绝记录基线**——所以适配器写错不会污染基线，只会失败。
- 输出是打包形式的题在适配器里做**逐句同构**的打包。上游 `bd991d0` 之后统一为
  **一维 float32 拼接**协议，适配器必须跟着改，否则官方基线会因形状/精度不对齐而判负：
  - `1017_mha`：`cat([out.to(q.dtype).reshape(-1).to(float32), lse.reshape(-1)])`
    —— `lse` 保持 fp32、**不**降精度到 `q.dtype`（早期适配器把 `lse` 塞进 4 维最后一列，
    已随上游协议废弃）。
  - `1019_mha_onekernel_bwd`：三段梯度各自先 cast 回输入 dtype（`dq`/`dk`→`k.dtype`、
    `dv`→`v.dtype`）再转 fp32，沿第 0 维拼接。
  - `1020_mla_decode_rope` 的 `o|k_pe` 同理。
- aiter 一律**函数内 import**（真机走最小导入垫片，aiter 顶层 import 很重）。
- 契约是**按名取参**（不是位置式 `init_args`）——位置式在稀疏构造参数下会错位。

## ⚠️ 需要在部署侧补的文件：15 题的 autotune config

这些题的 aiter 入口在 `config=None` 时读
`$AITER_TRITON_CONFIGS_PATH/<device>-<OP>.json`（`<device>` 由 `arch_info` 把 gfx936 映射成 `BW200`），
而 pinned commit 只带 `MI300X`/`MI350X` 的文件。缺文件时**不是失败**：入口回退到 MI 系列配置，
正确性不受影响，但**未针对 BW200 调优 → 用时偏大**。

| 题 | 影响 |
|---|---|
| `1010_chunked_pa_prefill` | 缺文件只 warning 后退兜底 config → 正确性不受影响，**性能次优** |
| `1014_hstu_attention` | 同上；适配器已内联兜底 config |
| `1017_mha` | 同上（缺 `BW200-MHA-DEFAULT.json`） |
| `1018_mha_fused_bwd` | 同上 |
| `1019_mha_onekernel_bwd` | 同上；该题 aiter 另有 shared memory 上限问题，基线最终不可得（见上） |
| `1020_mla_decode_rope` | 同上；适配器已内联兜底 config |
| `1027_sage_attention_qk_int8_per_block_causal` | 同上 |
| `1030_unified_attention` | 同上 |
| `4001_batched_gemm_a8w8` | 同上 |
| `4004_batched_gemm_bf16` | 同上 |
| `4006_gemm_a16w16` | 同上 |
| `4008_gemm_a16w4` | 同上 |
| `4009_gemm_a8w8` | 同上 |
| `4010_gemm_a8w8_blockscale` | 同上 |
| `4016_gemm_w8a8` | 同上 |

补 `BW200-<OP>.json` 即可让这些题走官方调优通道（适配器会优先探测该文件，无需改代码）。
其中 `1014` / `1017` / `4001` / `4004` / `4006`（全部 case）与 `4008`（部分 case）
**已逐个核对 309 个本地配置确认落到了 MI 系列兜底**（清单见上文"真机采集结果"），
补齐配置后这 6 道的基线应重采。

补 `BW200-<OP>.json` 即可让这些题走官方调优通道（适配器会优先探测该文件，无需改代码）。
**注意**：其中若含 `waves_per_eu` / `matrix_instr_nonkdim` / `kpack` 等 AMD 专有 launch 参数，
DTK Triton 是否接受未经验证。

## 未写适配器的题（1）

### `4007_gemm_a16w16_atomic` — 阻塞在设备调优 JSON

算子语义本身可对齐（`gemm_a16w16_atomic(x, w, dtype, y, config)` 单输出 `[M,N]`，
权重布局直接匹配），但入口在 `config=None` 时**无条件**读
`{AITER_TRITON_CONFIGS_PATH}/gemm/BW200-GEMM-A16W16-ATOMIC.json`，该文件不存在 →
真机直接 `FileNotFoundError`（不是退化，是失败）。同族的 `4006_gemm_a16w16` 同样缺
`BW200-GEMM-A16W16.json`，但 `4006` 的适配器已内联兜底 config，`4007` 未内联
（可用官方 `config=<dict>` 形参绕过，但那等于手工猜分块与 launch 参数，
得到的是"手工配置的 aiter"而非官方调优基线，会系统性污染 speedup —— 按"别猜"原则不采用）。

**部署侧补上 `BW200-GEMM-A16W16-ATOMIC.json`（且含 `"any"` 键）后，此题约 20 行即可适配。**

## 需人工复核的题（4）

workflow 里这 4 个 agent 未按约定回报结论 JSON，但**文件已落盘**；静态校验通过，
仍建议人工确认绑定是否正确：

`1017_mha`、`2001_activation`、`2004_fused_qk_concat`、`2005_moe_activation`

## 静态校验（本机无 GPU / 无 aiter，只能静态）

`.dcu_runs/validate_adapters.py` 逐文件检查：`py_compile`、`def run(inputs, init_kwargs, device)`
形参名、`return` 是否为 2 元组、aiter 是否只在函数内 import、顶层是否只 import torch/标准库、
`init_kwargs` 用到的键是否 ⊆（`io.init_inputs` ∪ `Model.__init__` 签名）、是否出现核心计算算子名。

结果：**44 个文件全部通过语法与契约检查**。两类提示已逐条复核：

- `init_kwargs` 用了未声明的键（12 题）：是 `scale` / `sm_scale`，**它们是 `Model.__init__`
  的签名参数但未被 `io.init_inputs` 声明**，适配器用 `.get(name, default)` 防御性读取 →
  解析出来的 kwargs 里本就不会有该键，取默认值，与 `Model` 缺省一致。**不是 bug。**
- `4008_gemm_a16w4` 的 `use_fused_kernel`：aiter 源码里该融合分支已被注释掉
  （inert 参数），适配器读到非 0 就 raise、否则传 0。**合理。**
- `4010_gemm_a8w8_blockscale` 的 `head_size`：从旧位置式约定抄来的**死分支**
  （我们只传 `group_k/group_n/out_dtype`，永不触发）。**无害，建议后续清理。**
- `2009_softmax` / `1026_*` 命中 `torch.softmax` / `scaled_dot_product_attention`：
  **只在注释里**（引用官方测试的比对口径），代码未调用。

## 顺带修掉的一个评测器真 bug

`1022_pa_prefill` 的 `alibi_slopes` 是可选输入，`make_inputs` 在 `use_alibi=False` 时返回
`None`，而 `audit_model_class.py` / `record_baseline.py` 里的
`[t.to(device) for t in case_inputs(...)]` 会在 `None` 上抛 `AttributeError` ——
**整题的基线采集与终审都会崩**。已改为 `move_inputs_to_device()`：非张量元素原样透传，
与上游 KernelBench 的 `_process_input_tensor` 口径一致。

## 真机前置条件（部署侧）

1. **最小导入垫片**：每个适配器头部注明了需要的 aiter 模块，垫片要带上它们及传递依赖
   （`aiter/ops/triton/utils/*`、`configs/` 等）。样例见 `.dcu_runs/aiter_shim/`。
2. **DTK Triton 扩展**：部分模块 `from triton.utils.hcutuner import get_gpu_label`
   （如 sage_attention 链），依赖 DTK Triton 的 hcutuner；缺失会导入即失败。
3. **aiter commit** 必须是 `c39fff8c77df4e80617649e92fa3c2615f2c43d1`（各题 `sources/<id>.yaml`
   记录了 sha256 证据）。

## 怎么用

```bash
# 采一道题的 aiter 官方基线（真机；需要 private/<id>/aiter_impl.py）
python benchmark/evaluator/record_baseline.py --task <id>              # --impl 默认 aiter
# 采同一道题的 torch_eager 基线（直接计时 reference.py，不需要适配器）
python benchmark/evaluator/record_baseline.py --task <id> --impl eager
# 只校验不写回
python benchmark/evaluator/record_baseline.py --task <id> --dry-run
# 终审一道题的 agent 产物
python benchmark/evaluator/audit_model_class.py --task <id> --submission <runs>/best_code.py
```

- `--impl` 决定**写哪一条**：`aiter`（默认，需适配器）或 `eager`/`torch_eager`（无需适配器）。
  同一条 case 可以同时有两条基线，采集器按 `impl` **合并**写入（不覆盖另一种）；批量采集
  用外部循环逐题调（`--task` 是必填的单题参数，没有 `--all`）。
- 采集前会先用 `task.yaml` 的容差比较实现与 `reference`，**不一致就拒绝记录**——
  所以适配器写错只会失败，不会污染基线。
- `--impl eager` 只依赖 `reference.py`，因此是**任何一道题都能拿到的兜底分母**
  （18 道题的 aiter 基线缺失时就是靠它）。

建议顺序：先 2xxx 号段（norm / rope / act，接口简单、无 config 依赖）建立信心，
再啃 1xxx attention（分页 / varlen / layout 转换多），最后 3xxx moe（接口最杂）。
