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
- 输出是打包形式的题（`1017_mha` 的 `out|lse` 拼列、`1019_mha_onekernel_bwd` 的三段展平、
  `1020_mla_decode_rope` 的 `o|k_pe` 拼接）在适配器里做同样的打包。
- aiter 一律**函数内 import**（真机走最小导入垫片，aiter 顶层 import 很重）。
- 契约是**按名取参**（不是位置式 `init_args`）——位置式在稀疏构造参数下会错位。

## ⚠️ 需要在部署侧补的文件：14 题的 autotune config

这些题的 aiter 入口在 `config=None` 时读
`$AITER_TRITON_CONFIGS_PATH/<device>-<OP>.json`（`<device>` 由 `arch_info` 把 gfx936 映射成 `BW200`），
而 pinned commit 只带 `MI300X`/`MI350X` 的文件：

| 题 | 影响 |
|---|---|
| `1010_chunked_pa_prefill` | 缺文件只 warning 后退默认 config → 正确性不受影响，**性能次优** |
| `1014_hstu_attention` | 同上；适配器已内联兜底 config |
| `1018_mha_fused_bwd` | 同上 |
| `1019_mha_onekernel_bwd` | 同上 |
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
# 采一道题的基线（真机）
python benchmark/evaluator/record_baseline.py --task <id>
# 终审一道题的 agent 产物
python benchmark/evaluator/audit_model_class.py --task <id> --submission <runs>/best_code.py
```

建议顺序：先 2xxx 号段（norm / rope / act，接口简单、无 config 依赖）建立信心，
再啃 1xxx attention（分页 / varlen / layout 转换多），最后 3xxx moe（接口最杂）。
