# aiter_impl.py — 1012_flash_attention_forward 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
#
# 来源（aiter pinned 检出 .dcu_runs/aiter_pinned，commit c39fff8c）：
#   aiter/ops/triton/flash_attention_forward.py
#     sha256 3ae6731a443ec04bdc925221feb671361a0a82997bb932a9f7987bdb46ce9da3
#     role   device_kernel（_attn_fwd_inner + attn_fwd；核心计算全在源文件内）
#   op_tests/triton_tests/test_flash_attention_forward.py
#     sha256 fb769ccb318783775f4e94230425bfd9b1792016ec69bb94c5e65bc075a0d312
#     role   official_test（语义权威；含 PyTorch 参考实现）
#
# 公开入口：本文件**没有**独立的 flash_attention_forward()/attn_fwd_fwd() 包装函数。
# 算子以 torch.autograd.Function 形式暴露，模块级别名 triton_attention =
# _attention.apply（源码 flash_attention_forward.py:1148），官方测试与官方 bench
# 均直接调用它（test_flash_attention_forward.py:209、bench_flash_attention_forward.py:206）：
#
#   _attention.apply(q, k, v, o, cu_seqlens_q, cu_seqlens_k,
#                    max_seqlens_q, max_seqlens_k,
#                    causal=False, sm_scale=1.0, bias=None,
#                    fp8_scales=None, fp8_out_scale=None) -> (o, encoded_softmax)
#
# 本适配器按官方测试的**位置式**实参顺序调用（与 :209 逐参对齐），取返回元组的
# 第 0 项作为 out。
#
# 语义对齐（题面 reference.py vs attn_fwd）：
#   - scale：kernel 内部 q = (q * sm_scale * log2(e)) 后走 exp2（源码 669/690/228），
#     与 reference 的 matmul 结果乘 sm_scale 再标准 softmax 数值等价（fp16 预乘
#     只引入输入精度级舍入，官方测试即以此对 fp32 参考断言 atol=1e-2）。
#   - causal：offs_n_causal = offs_n + (seqlen_q - seqlen_k)，
#     mask 为 OFFS_M >= causal_boundary（源码 769/211-214），即 query i 只 attend
#     key j <= i + (L_k - L_q)——与 reference 的 diagonal=L_k-L_q+1 右下对齐掩码同义。
#   - GQA/MQA：GROUP_SIZE = HQ // HK，off_h_k = off_h_q // GROUP_SIZE（源码 606-607），
#     与 reference 的 repeat_interleave(group) 的 h_q -> h_q // group 映射一致。
#   - 输出：仅 store O（源码 863-877），LSE 的写回在源码里被整段注释（849-861），
#     题面也声明不含 LSE 输出；返回元组第二项 encoded_softmax 恒为 None
#     （RETURN_ENCODED_SOFTMAX=False），故直接丢弃，产出单个张量。
#   - 题面约束每条序列 0 < L_q <= L_k，恰好避开 kernel 在 causal 且 L_q > L_k 时
#     「整块早退 + 不 store」与参考实现「softmax 全 -inf 行 -> NaN」的冲突区
#     （见 benchmark/sources/1012_flash_attention_forward.yaml 的说明）。本适配器
#     对该约束做显式校验，越界即 raise，绝不静默算出错结果。
#
# 布局：本评测集 make_inputs 产出的 q [total_q, HQ, D] / k,v [total_k, HK, D] 稠密
# 张量（按 cu_seqlens 分段拼接）与官方测试 build_inputs 完全同构，正是 aiter
# varlen 分支要求的形态，**无需 permute/reshape**；host 侧 stride 由
# _attention.forward 自行组装为 (0, stride(1), stride(0), stride(2))——varlen 下
# 第 0 维（"z"）stride 记 0，段起点靠 cu_seqlens 偏移，段与段之间互不覆写。
#
# autotune：attn_fwd 带源码内联的 @triton.autotune（configs 来自本文件的
# get_cdna_autotune_configs，源码 301-440），**不读取 AITER_TRITON_CONFIGS_PATH /
# 任何 config JSON**，故无需外部 autotune 配置文件。
#
# 注意：本机（无 GPU、无 aiter）只做过静态检查（py_compile），未运行验证。

import torch

# aiter 认为「无需 PADDED_HEAD」的 head_dim 集合（源码 981）；仅用于 ctx 记录，
# 不作为分派依据——真正的 padding 决策在 _attention.forward 内部。
_UNPADDED_HEAD_DIMS = (32, 64, 128, 256)


def _padded_dmodel(head_dim: int) -> int:
    """复刻源码 980-990 的 padded_d_model 选择（仅记录用，不参与调用）。"""
    if head_dim in _UNPADDED_HEAD_DIMS:
        return head_dim
    for dim in _UNPADDED_HEAD_DIMS:
        if dim > head_dim:
            return dim
    raise RuntimeError(f"head_dim={head_dim} 超过 aiter 支持的最大 head dim 256")


def _as_causal_flag(causal) -> bool:
    """题面 causal 是 0-dim int32 张量（io 声明 0/1），转成 Python bool。"""
    if torch.is_tensor(causal):
        value = int(causal.to(torch.int32).reshape(-1)[0].item())
    else:
        value = int(causal)
    if value not in (0, 1):
        raise ValueError(f"causal 只允许 0/1，收到 {value}")
    return bool(value)


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方实现（varlen FlashAttention v2 前向）。

    inputs    : [q, k, v, cu_seqlens_q, cu_seqlens_k, causal]，与
                reference.py::make_inputs 的返回顺序一致，已在 device 上；
                q [total_q, HQ, D]、k/v [total_k, HK, D]（fp16/bf16，contiguous）、
                cu_seqlens_q/k [num_seqs+1] int32、causal 0-dim int32。
    init_kwargs: {"head_dim": int, "scale": float | None}（Model.__init__ 的参数名）
    device    : 目标设备（输入已在设备上，此处不重复搬运）

    返回 (out, ctx)：out 为 [total_q, HQ, D]、dtype 同 q 的单个张量（与 reference
    输出同形同 dtype）；ctx 记录走了哪条 aiter 路径与关键 shape。
    """
    # aiter 顶层 import 很重，按契约在函数内最小导入
    from aiter.ops.triton.flash_attention_forward import _attention

    if len(inputs) != 6:
        raise ValueError(
            f"1012 期望 6 个输入 [q, k, v, cu_seqlens_q, cu_seqlens_k, causal]，"
            f"收到 {len(inputs)} 个"
        )
    q, k, v, cu_seqlens_q, cu_seqlens_k, causal = inputs

    # ---- 构造参数（题面 io.init_inputs 只声明 head_dim；scale 走默认 None）----
    head_dim = init_kwargs.get("head_dim")
    if head_dim is None:
        raise KeyError("init_kwargs 缺 head_dim：Model.__init__ 的必需参数，不能猜")
    head_dim = int(head_dim)
    scale = init_kwargs.get("scale", None)
    sm_scale = float(scale) if scale is not None else head_dim ** -0.5

    # ---- 形状 / dtype / 布局校验（越界 raise，绝不静默用错参数）----
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(f"q/k/v 必须是 3 维 [total, heads, head_dim]，收到 {q.dim()}")
    if k.shape != v.shape:
        raise ValueError(f"k/v 形状必须一致，收到 {tuple(k.shape)} / {tuple(v.shape)}")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q/k 的 head_dim 必须一致，收到 {q.shape[-1]} / {k.shape[-1]}")
    if q.shape[-1] != head_dim:
        raise ValueError(
            f"init_kwargs 的 head_dim={head_dim} 与 q.shape[-1]={q.shape[-1]} 不一致"
        )
    if not (1 <= head_dim <= 256):
        raise ValueError(f"题面约束 1 <= head_dim <= 256，收到 {head_dim}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"q/k/v dtype 必须一致，收到 {q.dtype}/{k.dtype}/{v.dtype}")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"本题 dtype 域为 fp16/bf16，收到 {q.dtype}")
    num_q_heads = int(q.shape[1])
    num_kv_heads = int(k.shape[1])
    if num_kv_heads < 1 or num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"题面约束 num_q_heads 是 num_kv_heads 的整数倍，收到 {num_q_heads}/{num_kv_heads}"
        )

    # cu_seqlens：单调非降、首元素 0、尾元素 total_*，且每条序列 1 <= L_q <= L_k
    cu_q = cu_seqlens_q.to(torch.int32).contiguous()
    cu_k = cu_seqlens_k.to(torch.int32).contiguous()
    if cu_q.dim() != 1 or cu_k.dim() != 1:
        raise ValueError("cu_seqlens_q / cu_seqlens_k 必须是一维张量")
    if cu_q.numel() != cu_k.numel() or cu_q.numel() < 2:
        raise ValueError(
            f"cu_seqlens_q/k 长度必须相等且 >= 2，收到 {cu_q.numel()} / {cu_k.numel()}"
        )
    bounds_q = [int(x) for x in cu_q.to(torch.int64).cpu().tolist()]
    bounds_k = [int(x) for x in cu_k.to(torch.int64).cpu().tolist()]
    if bounds_q[0] != 0 or bounds_k[0] != 0:
        raise ValueError(f"cu_seqlens 首元素必须为 0，收到 {bounds_q[0]} / {bounds_k[0]}")
    if bounds_q[-1] != int(q.shape[0]) or bounds_k[-1] != int(k.shape[0]):
        raise ValueError(
            f"cu_seqlens 尾元素必须等于 total_q/total_k，收到 "
            f"{bounds_q[-1]}/{bounds_k[-1]} vs {q.shape[0]}/{k.shape[0]}"
        )
    num_seqs = len(bounds_q) - 1
    seq_lens_q = [bounds_q[i + 1] - bounds_q[i] for i in range(num_seqs)]
    seq_lens_k = [bounds_k[i + 1] - bounds_k[i] for i in range(num_seqs)]
    for b, (len_q, len_k) in enumerate(zip(seq_lens_q, seq_lens_k)):
        if len_q < 1 or len_q > len_k:
            raise ValueError(
                f"题面约束每条序列 0 < L_q <= L_k，第 {b} 段收到 "
                f"L_q={len_q} / L_k={len_k}（aiter kernel 在 causal 且 L_q > L_k 时"
                "会整块早退输出 0，与 reference 的 NaN 语义冲突，不在准入范围内）"
            )

    max_seqlen_q = max(seq_lens_q)
    max_seqlen_k = max(seq_lens_k)
    is_causal = _as_causal_flag(causal)

    # ---- 调用 aiter 官方入口（位置式实参，与官方测试 :209 逐参对齐）----
    # stride 由 host 侧按 varlen 规则自行组装，这里只需保证 contiguous。
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = torch.empty_like(q)

    result = _attention.apply(
        q,
        k,
        v,
        out,
        cu_q,
        cu_k,
        max_seqlen_q,
        max_seqlen_k,
        is_causal,
        sm_scale,
        None,   # bias：题面无 bias/ALiBi，BIAS_TYPE=0
        None,   # fp8_scales：题面非量化，USE_FP8=False
        None,   # fp8_out_scale：无 fp8 输出
    )
    # _attention.forward 返回 (o, encoded_softmax)，后者恒为 None（题面不含 LSE）
    if isinstance(result, (tuple, list)):
        out = result[0]
    else:  # 理论上不会走到；防御性处理
        out = result

    torch.cuda.synchronize()

    ctx = {
        "impl": "aiter.ops.triton.flash_attention_forward._attention.apply",
        "path": (
            "varlen fwd kernel attn_fwd (VARLEN=True, BIAS_TYPE=0, "
            "ENABLE_DROPOUT=False, USE_FP8=False, USE_FP8_OUT=False, "
            "RETURN_ENCODED_SOFTMAX=False)"
        ),
        "varlen": True,
        "causal": is_causal,
        "is_mls_path": "运行时由 is_mls_avail() 决定（gfx938/gfx92a 为 True）",
        "num_seqs": num_seqs,
        "seq_lens_q": seq_lens_q,
        "seq_lens_k": seq_lens_k,
        "total_q": int(q.shape[0]),
        "total_k": int(k.shape[0]),
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "group_size": num_q_heads // num_kv_heads,
        "head_dim": head_dim,
        "padded_d_model": _padded_dmodel(head_dim),
        "dtype": str(q.dtype),
        "sm_scale": sm_scale,
        "scale_from_init": scale is not None,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
        "out_shape": list(out.shape),
        "out_dtype": str(out.dtype),
        "autotune": "in-source @triton.autotune configs (无 AITER_TRITON_CONFIGS_PATH/JSON 依赖)",
    }
    return out, ctx
