# aiter_impl.py — 1014_hstu_attention 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
# 适配器契约：run(inputs, init_kwargs: dict, device) —— 按名取参（构造参数往往
# 稀疏给出，位置式取值会错位，见 audit_model_class.py::case_init_kwargs）。
#
# 来源（aiter pinned 检出 .dcu_runs/aiter_pinned/，commit c39fff8c77df4e80617649e92fa3c2615f2c43d1）：
#   device_kernel : aiter/ops/triton/hstu_attention.py
#                   sha256 0939420dfbaa03b9da69a039a2eadc8684fb12add82f37698c406a0c190e8ca2
#   official_test : op_tests/triton_tests/test_hstu_attn.py
#                   sha256 90fcee48290e6e71c7647da5196511e9e08486b915c1c7853149b0fb9abfe5c6
#   语义参考      : op_tests/triton_tests/utils/hstu_attention_ref.py::torch_hstu_attention
#                   + _get_valid_attn_mask（题面 reference.py 与之同式）
#
# 入口签名（宿主函数 hstu_attention.py:936；官方测试经 autograd 包装
# _AttentionFunction.apply 间接调用同一函数，hstu_attention.py:1220）：
#
#   triton_hstu_attention_fwd(N: int, alpha: float, q, k, v, seq_offsets, causal: bool,
#                             num_targets, max_attn_len: int, contextual_seq_len: int,
#                             sort_by_length_indices, config: Optional[dict] = None)
#       -> torch.Tensor                       # [L, H, DimV]，empty_like(v) 的 dtype
#
# 该符号**未**被 aiter 的 __init__ 再导出（全仓 grep 只有本文件内的定义与调用），
# 故按模块路径直接 import（函数内 import，避开 aiter 顶层重导入）。
#
# 布局说明：aiter 期望 (L, H, D) 的 jagged 拼接形态，与题目 io 的
# q/k/v [total, num_heads, qk_dim|v_dim] 逐位一致，**无需 permute/reshape**；
# 输出 out = torch.empty_like(v) 为 [total, num_heads, v_dim]，与 reference.forward
# 的返回（单个张量、同形同 dtype、不打包）完全对齐，故本适配器不做打包。
# v_dim != qk_dim 时 aiter 分别按 BLOCK_D_Q=DimQ / BLOCK_D_V=DimV 处理
# （hstu_attention.py:1026-1027），与题面「v_dim 可不等于 qk_dim」一致。
#
# 语义对齐（与 sources/1014_hstu_attention.yaml 准入说明一致）：
#   - 掩码：one_block 内 invalid_mask = (offs_m == offs_n) | (dist > 0)
#     （hstu_attention.py:102、133），但它在 :142 是
#     `silu = tl.where(invalid_mask, silu, 0)` —— **变量名反了，该掩码实为 valid**
#     （True 保留 silu）。故实际语义 = 对角 valid 或 dist > 0，再与
#     dist <= max_attn_len 求交（:135）、contextual 行放宽（:136-139），与题面
#     reference.py:108-112 的 `eye | (dist > 0)`、`& (dist <= A)`、contextual
#     放宽逐条同式。dist 用重编号后的 offs（contextual 先 clamp(min=0) 再
#     clamp(max=M-g)，:104-129），即题面的 m[i] = min(max(i-c+1,0), M-g)。
#   - SiLU 以 fast_dividef(qk, 1+fast_expf(-qk)) 自实现并乘 1/MAX_SEQ_LEN
#     （:141），qk 已乘 alpha（:101），与 reference 的 SiLU(alpha·QK^T)/N 一致；
#     silu cast 到 v.dtype 后 tl.dot（:144-145），fp32 累加（:239）。
#   - MAX_SEQ_LEN = N = init 的 max_seq_len（:1015），即题面「缩放分母恒为 N」。
#   - 序列分段取 seq_offsets（:183-187），空序列（L=0）不产生输出行，
#     L == 0（total = 0）时宿主函数直接返回空 out（:978-979）。
#   - num_targets 恒为张量输入（题目 io 如此）：HAS_MULTIPLE_TARGETS=True 时
#     n_targets = tl.load(num_targets + off_z)（:196），g = 0 时 max_ids 不减、
#     clamp 为恒等，等价于官方 num_targets=None 路径。
#   - sort_by_length_indices=None（纯调度优化，语义不变）→ 传 None。
#   - 本题只做前向：不走 _AttentionFunction（它会 save_for_backward 建图），
#     直接调用宿主 forward 函数。
#
# ⚠️ autotune config：_get_fwd_config（hstu_attention.py:909-933）按
#   `{AITER_TRITON_CONFIGS_PATH}/hstu_attn/{arch_info.get_device()}-HSTU_ATTN_FWD.json`
#   读 JSON，缺文件即 FileNotFoundError。仓库只带了
#   aiter/ops/triton/configs/hstu_attn/{MI300X,MI350X}-HSTU_ATTN_FWD.json，而
#   arch_info._ARCH_TO_DEVICE 把 gfx936/gfx938 映射为 BW200/BW200B（arch_info.py:5-10）
#   —— 即 **DCU 真机上该 JSON 不存在**，走 config=None 会直接抛错。
#   因此本适配器：设备同名 JSON 存在 → config=None（完全走官方 config 通道）；
#   否则显式传 config（官方 API 的 config 形参就是给外部供 config 用的，:948），
#   取值与仓库内 MI300X/MI350X JSON 的三个 batch 桶逐字段相同，桶选择沿用
#   官方 _get_fwd_config 的 AUTOTUNE_Z 规则（<512 → small_batch，==512 →
#   batch_512，>512 → large_batch）。config 只影响分块/占用（性能），不影响数值。
#   ctx 里记录 config_source 便于终审核对。
#
# 其他说明：宿主函数不要求 q/k/v 连续（kernel 按 stride 取址，:1003-1008），
# 这里仍按官方测试的 switch_to_contiguous_if_needed 口径做一次布局归一。

import math

import torch

# 官方 config JSON 缺失时的显式回退（逐字段抄自
# aiter/ops/triton/configs/hstu_attn/MI300X-HSTU_ATTN_FWD.json，MI350X 同值）
_FALLBACK_FWD_CONFIGS = {
    "small_batch": {
        "BLOCK_M": 64,
        "BLOCK_N": 32,
        "num_warps": 4,
        "num_stages": 1,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "kpack": 2,
    },
    "batch_512": {
        "BLOCK_M": 128,
        "BLOCK_N": 32,
        "num_warps": 4,
        "num_stages": 1,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "kpack": 2,
    },
    "large_batch": {
        "BLOCK_M": 64,
        "BLOCK_N": 32,
        "num_warps": 4,
        "num_stages": 1,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "kpack": 2,
    },
}


def _prev_power_of_2(x: int) -> int:
    """与 aiter.ops.triton.utils.common_utils.prev_power_of_2 等价（:35-37）。"""
    out = 1
    while out * 2 <= x:
        out *= 2
    return out


def _batch_key(autotune_z: int) -> str:
    """官方 _get_fwd_config 的桶选择规则（hstu_attention.py:926-932）。"""
    if autotune_z < 512:
        return "small_batch"
    if autotune_z == 512:
        return "batch_512"
    return "large_batch"


def _resolve_fwd_config(autotune_z: int):
    """返回 (config, source)：设备同名 JSON 在则 None（官方通道），否则显式 config。"""
    try:
        import os

        from aiter.ops.triton.utils import arch_info
        from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH

        fpath = os.path.join(
            AITER_TRITON_CONFIGS_PATH,
            "hstu_attn",
            f"{arch_info.get_device()}-HSTU_ATTN_FWD.json",
        )
        if os.path.exists(fpath):
            return None, f"official_autotune_json:{fpath}"
        source = f"fallback_config(no {os.path.basename(fpath)})"
    except Exception as exc:  # pragma: no cover - 仅探测，失败即回退
        source = f"fallback_config(probe {type(exc).__name__})"
    key = _batch_key(autotune_z)
    return dict(_FALLBACK_FWD_CONFIGS[key]), f"{source}:{key}"


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方实现（hstu_attention.triton_hstu_attention_fwd）。

    inputs     : [q, k, v, seq_offsets, num_targets]（顺序同 reference.make_inputs）
                 q [total, H, DimQ] / k [total, H, DimQ] / v [total, H, DimV]（fp16/bf16）
                 seq_offsets [B + 1] int64（排他前缀和）/ num_targets [B] int32
    init_kwargs: {"max_seq_len": int, "alpha": float|None, "causal": bool,
                  "max_attn_len": int, "contextual_seq_len": int}
                 （稀疏给出；alpha 缺省/None → 1/sqrt(DimQ)，与 Model 缺省一致）
    device     : 目标设备（张量已在 device 上，仅用于 ctx 记录）

    返回 (out, ctx)；out 为 [total, H, DimV] 连续张量，dtype 同 v。
    """
    from aiter.ops.triton.hstu_attention import triton_hstu_attention_fwd

    q, k, v, seq_offsets, num_targets = inputs

    # ---- 构造参数（按名取参，越界 raise，绝不静默用错）----------------------
    if "max_seq_len" not in init_kwargs:
        raise ValueError("init_kwargs 缺 max_seq_len（Model.__init__ 的必需参数）")
    n = int(init_kwargs["max_seq_len"])
    if n <= 0:
        raise ValueError(f"max_seq_len 必须 > 0（缩放分母与长度上界），实际 {n}")

    alpha_in = init_kwargs.get("alpha", None)
    causal = bool(init_kwargs.get("causal", True))
    max_attn_len = int(init_kwargs.get("max_attn_len", 0) or 0)
    contextual_seq_len = int(init_kwargs.get("contextual_seq_len", 0) or 0)
    if max_attn_len < 0:
        raise ValueError(f"max_attn_len 必须 >= 0（0 = 不限距离），实际 {max_attn_len}")
    if contextual_seq_len < 0:
        raise ValueError(
            f"contextual_seq_len 必须 >= 0（0 = 关闭），实际 {contextual_seq_len}"
        )

    # ---- shape / dtype 合法性（题目全域约束，违反即 raise）-----------------
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(
            f"q/k/v 必须是 3 维 [total, heads, dim]，实际 "
            f"{tuple(q.shape)} / {tuple(k.shape)} / {tuple(v.shape)}"
        )
    total, h, dim_q = q.shape
    dim_v = v.shape[2]
    if k.shape != (total, h, dim_q):
        raise ValueError(f"k 形状须与 q 相同：q{tuple(q.shape)} k{tuple(k.shape)}")
    if v.shape[:2] != (total, h):
        raise ValueError(f"v 前两维须与 q 相同：q{tuple(q.shape)} v{tuple(v.shape)}")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"aiter hstu 前向按半精度 tl.dot 走（qk 与 silu 均 cast 到输入 dtype），"
            f"dtype 需为 float16/bfloat16，实际 {q.dtype}"
        )
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(f"q/k/v dtype 必须一致：{q.dtype} / {k.dtype} / {v.dtype}")
    for name, d in (("qk_dim", dim_q), ("v_dim", dim_v)):
        if d < 16 or (d & (d - 1)) != 0:
            raise ValueError(
                f"{name}({d}) 必须是 >= 16 的 2 的幂（BLOCK_D_Q/BLOCK_D_V 直接取该值，"
                "block pointer 要求 2 的幂，hstu_attention.py:1026-1027）"
            )

    seq_offsets_l = seq_offsets.to(torch.long).contiguous()
    num_targets_i32 = num_targets.to(torch.int32).contiguous()
    z = int(seq_offsets_l.numel()) - 1
    if z != int(num_targets_i32.numel()):
        raise ValueError(
            f"seq_offsets 长度须为 batch + 1（实际 {seq_offsets_l.numel()}），"
            f"num_targets 长度须为 batch（实际 {num_targets_i32.numel()}）"
        )
    if z > 0 and int(seq_offsets_l[-1]) != total:
        raise ValueError(
            f"seq_offsets 尾元素({int(seq_offsets_l[-1])})须等于 total({total})"
        )
    if bool((num_targets_i32 < 0).any()):
        raise ValueError("num_targets 必须 >= 0（题目不变式）")

    # alpha：Model 缺省 1/sqrt(qk_dim)（reference.py:80）
    alpha = 1.0 / math.sqrt(dim_q) if alpha_in is None else float(alpha_in)
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"alpha 必须是有限正数，实际 {alpha!r}")

    # ---- layout 归一（题目已是 aiter 期望的 [total, H, D] jagged 形态）------
    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()

    # ---- autotune config（DCU 无 BW200/BW200B 同名 JSON，见文件头说明）-----
    config, config_source = _resolve_fwd_config(_prev_power_of_2(z) if z > 0 else 1)

    out = triton_hstu_attention_fwd(
        N=n,
        alpha=alpha,
        q=q_c,
        k=k_c,
        v=v_c,
        seq_offsets=seq_offsets_l,
        causal=causal,
        num_targets=num_targets_i32,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        sort_by_length_indices=None,
        config=config,
    )

    torch.cuda.synchronize()

    ctx = {
        "impl": "aiter",
        "module": "aiter.ops.triton.hstu_attention",
        "path": "triton_hstu_attention_fwd",
        "batch": z,
        "num_heads": int(h),
        "qk_dim": int(dim_q),
        "v_dim": int(dim_v),
        "total": int(total),
        "max_seq_len": n,
        "alpha": alpha,
        "causal": causal,
        "max_attn_len": max_attn_len,
        "contextual_seq_len": contextual_seq_len,
        "has_multiple_targets": True,
        "sort_by_length_indices": None,
        "config_source": config_source,
        "config": config,
        "out_dtype": str(out.dtype),
        "device": str(device),
    }
    return out, ctx
