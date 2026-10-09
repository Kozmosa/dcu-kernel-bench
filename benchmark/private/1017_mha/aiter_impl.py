# aiter_impl.py — 1017_mha 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
# 适配器契约：run(inputs, init_kwargs: dict, device) —— 按名取参（构造参数稀疏
# 给出时位置式会错位，见 audit_model_class.py::case_init_kwargs 的说明）。
#
# 来源（aiter pinned 检出 .dcu_runs/aiter_pinned/，commit c39fff8c77df4e80617649e92fa3c2615f2c43d1）：
#   device_kernel : aiter/ops/triton/mha.py
#                   sha256 a756124404c326fb12042d82b66404fc2ebd3e4f260f9c33d8dfee73de7ee54e（已核对）
#                   （核心计算 _attn_fwd / _attn_fwd_inner，mha.py:194-357 / 360-872）
#   official_test : op_tests/triton_tests/test_mha.py
#                   sha256 92e59f0977cab90c27707f5426000cd482dc0dabdd1f71fa4b8a5b0042737beb（已核对）
#
# 入口签名（宿主函数 mha.py:1235，官方测试 test_mha.py:143-151 的调用口径）：
#
#   flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
#                   window_size=(-1, -1), bias=None, alibi_slopes=None,
#                   deterministic=True, return_lse=False, return_attn_probs=False,
#                   config=None)
#       -> out（return_lse=False）或 (out, softmax_lse)（return_lse=True）
#
#   q          [batch, seqlen_q, num_q_heads, head_dim]   （bshd，fp16/bf16）
#   k, v       [batch, seqlen_k, num_kv_heads, head_dim]
#   alibi_slopes [num_heads] 或 [batch, num_heads] fp32
#   out        [batch, seqlen_q, num_q_heads, head_dim]，与 q 同 dtype
#   softmax_lse [batch, num_q_heads, seqlen_q] fp32（自然对数，mha.py:824-856）
#
# 布局说明：题目 io 的 q/k/v 正是 aiter 期望的 bshd 形态（mha.py:945-952 走
# 非 varlen 分支，按 stride 取 [B,S,H,D]），**无需 permute**；只做 contiguous
# 归一。alibi_slopes 用题目给的原生 [batch, num_q_heads] fp32（mha.py:619-621
# 按 off_z*stride_alibi_z + off_q_head*stride_alibi_h 取值，且 mha.py:1282-1284
# 明确支持 (batch, nheads)），与 reference 的 alibi_slopes[:, :, None, None]
# 逐 (b,h) 语义一致。输出打包与 reference 完全同构（上游 bd991d0 的单张量协议）：
#   packed = cat([out.to(q.dtype).reshape(-1).to(float32), lse.reshape(-1)])
#   → 一维 float32，长 B*Sq*Hq*(head_dim+1)（reference.py:116）。
#   注意 lse 保持 fp32、**不降精度**到 q.dtype；out 先按输出 dtype 约定 cast 回
#   q.dtype 再转 fp32。（早期版本曾把 lse 降精度塞进 4 维最后一列，已废弃。）
#
# 语义对齐（与 sources/1017_mha.yaml 的准入说明一致）：
#   - 右下对齐因果掩码：offs_n_causal = offs_n + (seqlen_q - seqlen_k)，整行掩码行
#     out=0、lse=0（mha.py:541-582 早退路径 + 834-837 epilogue 掩码），允许 Sq > Sk；
#   - GQA：grp_sz = NUM_Q_HEADS // NUM_K_HEADS，off_k_head = off_q_head // grp_sz
#     （mha.py:584-588）；
#   - ALiBi：_compute_alibi_block 给出 -slope*|i + Sk - Sq - j|，以
#     `qk += alibi_block / SM_SCALE` 加到**未缩放** score 上（mha.py:185-190、291-298），
#     等价于 reference 的 `scores = qk*sm_scale - slope*|i+Sk-Sq-j|`；causal 与非
#     causal 两条路径都施加（mha.py:699-797 两段 inner 调用都传 alibi_slope）；
#   - fp32 在线 softmax（exp2 基），lse = m*sm_scale + ln(l)（mha.py:822-832），
#     与 reference 的自然对数 logsumexp 同义；
#   - 非 2 次幂 head_dim：BLOCK_DMODEL_POW2 = max(next_pow2(head_sz), 16) 且
#     PADDED_HEAD 掩码（mha.py:954-956、660-663）；head_dim % 8 != 0 时宿主侧先
#     F.pad 到 8 的倍数再裁回（mha.py:1106-1110、1143）。
#
# ⚠️ autotune config（needs_autotune_config=True）：mha.py 的 tile 参数来自
#   `_get_config()` → `{AITER_TRITON_CONFIGS_PATH}/{dev}-MHA-DEFAULT.json`
#   （mha.py:875-891 读文件、883 拼路径；dev 由 arch_info.get_device() 映射，
#   gfx936 → "BW200"）。**该 JSON 在 pinned commit c39fff8c 的
#   aiter/ops/triton/configs/ 下并不存在**（该目录只有 BW200/BW200B 的
#   EXTEND_ATTENTION / GROUPED_DECODE_ATTENTION 等配置文件），所以 config=None
#   的默认路径在 BW200 上会 FileNotFoundError。本适配器的处理：
#     ① 若真机 aiter 安装里存在 {dev}-MHA-DEFAULT.json → 传 config=None，完全
#        走 aiter 自己的官方 tuned 配置（_get_config 的唯一权威来源）；
#     ② 否则退化到 mha.py:1004-1023 源码注释里给出的默认 tile
#        （fp16/bf16: BLOCK_M=128/BLOCK_N=64/waves_per_eu=2/num_warps=4/
#         num_ctas=1/num_stages=1；fp32 或 dropout>0 用小 tile 组），
#        ctx["config_source"] 记录走的是哪条路，绝不静默。
#   注意：②下的绝对性能不是官方 tuned 值（官方 JSON 缺失），基线偏保守；
#   拿到该 JSON 后无需改代码即可切回①。

import math
import os

import torch

# mha.py:1004-1023 源码注释内的默认 tile（fallback 路径专用）
_FALLBACK_CONFIG_DEFAULT = {
    "BLOCK_M": 128,
    "BLOCK_N": 64,
    "waves_per_eu": 2,
    "num_warps": 4,
    "num_ctas": 1,
    "num_stages": 1,
}
# mha.py:1014-1022：dropout 或 fp32 时官方改用小 tile（VGRP 压力）
_FALLBACK_CONFIG_DROPOUT_OR_FP32 = {
    "BLOCK_M": 32,
    "BLOCK_N": 32,
    "waves_per_eu": 1,
    "num_warps": 2,
    "num_ctas": 1,
    "num_stages": 1,
}


def _resolve_config(dtype):
    """选 tile 配置：官方 JSON 存在则交回 aiter 自己读（config=None）。

    返回 (config, source, fpath)：config 为 None 表示由 aiter `_get_config` 读官方
    JSON；否则为 fallback dict。aiter 一律函数内 import（顶层 import 很重）。
    """
    fpath = None
    try:
        from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
        import aiter.ops.triton.utils.arch_info as arch_info

        dev = arch_info.get_device()  # gfx936 -> "BW200"
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/{dev}-MHA-DEFAULT.json"
        if os.path.exists(fpath):
            return None, "official_json", fpath
    except Exception:
        # 拿不到路径/设备名（或缺文件）→ 走 fallback，绝不静默改语义
        fpath = None

    if dtype == torch.float32:
        return dict(_FALLBACK_CONFIG_DROPOUT_OR_FP32), "fallback_source_comment", fpath
    return dict(_FALLBACK_CONFIG_DEFAULT), "fallback_source_comment", fpath


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方实现（aiter.ops.triton.mha.flash_attn_func）。

    inputs     : [q, k, v, causal, alibi_slopes]（顺序同 reference.make_inputs）
                 q [B, Sq, Hq, D] / k,v [B, Sk, Hkv, D]（bshd，fp16/bf16）
                 causal 0-dim int32（1=右下对齐因果，0=无掩码）
                 alibi_slopes [B, Hq] fp32（>=0，全 0 等价无 ALiBi）
    init_kwargs: {"head_dim": int[, "sm_scale": float | None]}（Model.__init__ 参数）
    device     : 目标设备（张量已在 device 上，仅用于 ctx 记录）

    返回 (out, ctx)；out 为与 q 同 dtype 的 [B, Sq, Hq, D+1] 打包张量
    （前 D 列 = attention out，最后一列 = lse），与 reference.forward 同形同 dtype。
    """
    from aiter.ops.triton.mha import flash_attn_func

    q, k, v, causal, alibi_slopes = inputs

    # ---- 构造参数（按名取参；缺/越界一律 raise，绝不静默用错）---------------
    head_dim = init_kwargs.get("head_dim", None)
    if head_dim is None:
        raise ValueError(
            "init_kwargs 缺少 head_dim：Model.__init__(head_dim, sm_scale=None) 的 "
            "sm_scale 缺省值 1/sqrt(head_dim) 只能由它决定（io.init_inputs 声明了 "
            "head_dim，正常路径必然给出）"
        )
    head_dim = int(head_dim)
    if head_dim < 1 or head_dim > 256:
        raise ValueError(f"head_dim={head_dim} 越界（题目不变式 1 <= head_dim <= 256）")

    sm_scale_kw = init_kwargs.get("sm_scale", None)
    if sm_scale_kw is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    else:
        sm_scale = float(sm_scale_kw)
        if not math.isfinite(sm_scale) or sm_scale <= 0.0:
            raise ValueError(f"sm_scale={sm_scale!r} 非法（须为正有限值）")

    # ---- shape / dtype 合法性（题目全域约束，违反即 raise）-----------------
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            f"q/k/v 必须是 4 维 bshd [B, S, H, D]，实际 {tuple(q.shape)} / "
            f"{tuple(k.shape)} / {tuple(v.shape)}"
        )
    batch, seqlen_q, nheads_q, head_dim_q = (int(x) for x in q.shape)
    seqlen_k, nheads_k = int(k.shape[1]), int(k.shape[2])
    if k.shape[0] != batch or v.shape[0] != batch or v.shape[1] != seqlen_k or v.shape[2] != nheads_k:
        raise ValueError(
            f"K/V 的 batch/seqlen/heads 必须一致：q{tuple(q.shape)} k{tuple(k.shape)} "
            f"v{tuple(v.shape)}"
        )
    if k.shape[3] != head_dim_q or v.shape[3] != head_dim_q:
        raise ValueError(
            f"Q/K/V 头维必须相同：q{tuple(q.shape)} k{tuple(k.shape)} v{tuple(v.shape)}"
        )
    if min(batch, seqlen_q, seqlen_k, nheads_q, nheads_k) < 1:
        raise ValueError(
            f"batch/seqlen_q/seqlen_k/heads 必须 >= 1，实际 "
            f"({batch}, {seqlen_q}, {seqlen_k}, {nheads_q}, {nheads_k})"
        )
    if nheads_q % nheads_k != 0:
        raise ValueError(
            f"num_q_heads({nheads_q}) 必须是 num_kv_heads({nheads_k}) 的整数倍（GQA 整除）"
        )
    if head_dim_q < 1 or head_dim_q > 256:
        raise ValueError(f"q 的头维 {head_dim_q} 越界（1 <= head_dim <= 256）")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            f"aiter flash_attn_func 走 tl.dot 半精度/fp32 路径，dtype 需为 "
            f"float16/bfloat16/float32，实际 {q.dtype}（fp8 需要 descale 张量，"
            "题目 io 不提供）"
        )
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(f"Q/K/V dtype 必须一致：{q.dtype} / {k.dtype} / {v.dtype}")

    # alibi_slopes：[batch, num_q_heads] fp32（与 reference 的逐 (b,h) 语义一致）
    if not torch.is_tensor(alibi_slopes):
        alibi_slopes = torch.as_tensor(alibi_slopes, device=q.device)
    if alibi_slopes.dim() != 2 or tuple(alibi_slopes.shape) != (batch, nheads_q):
        raise ValueError(
            f"alibi_slopes 必须是 [batch, num_q_heads]={batch, nheads_q}，实际 "
            f"{tuple(alibi_slopes.shape)}"
        )
    if alibi_slopes.numel() and float(alibi_slopes.min()) < 0.0:
        raise ValueError("alibi_slopes 必须非负（题目不变式）")

    # causal：0-dim int32 开关（评测器若 cast 成 fp32，0/1 可精确恢复）
    if not torch.is_tensor(causal):
        causal = torch.tensor(causal)
    causal_i = int(causal.to(torch.int32).reshape(-1)[0].item())
    if causal_i not in (0, 1):
        raise ValueError(f"causal={causal_i} 非法（题目不变式：1=因果掩码，0=无掩码）")
    is_causal = bool(causal_i)

    # ---- layout 归一（题目已是 aiter 期望的 bshd；只做 contiguous）----------
    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    alibi_c = alibi_slopes.to(torch.float32).contiguous()

    config, config_source, config_fpath = _resolve_config(q_c.dtype)

    out, lse = flash_attn_func(
        q_c,
        k_c,
        v_c,
        dropout_p=0.0,
        softmax_scale=sm_scale,
        causal=is_causal,
        window_size=(-1, -1),
        bias=None,
        alibi_slopes=alibi_c,
        deterministic=True,
        return_lse=True,
        return_attn_probs=False,
        config=config,
    )

    torch.cuda.synchronize()

    # ---- 打包（与 reference.py:116 逐句同构）------------------------------
    # 上游的单张量打包协议（1 维 float32）：out 先按输出 dtype 约定 cast 回
    # q.dtype，再转 float32 展平；lse 本身即 float32，直接展平。两段拼接。
    # 旧写法把 lse 降精度到 q.dtype 塞进 4 维最后一列，已随上游 bd991d0 废弃。
    packed = torch.cat([
        out.to(q_c.dtype).reshape(-1).to(torch.float32),   # 先 cast 回 q.dtype，再转 fp32
        lse.reshape(-1),                                   # lse 本身即 fp32，不降精度
    ])                                                    # [B*Sq*Hq*(D+1)] float32

    ctx = {
        "impl": "aiter",
        "module": "aiter.ops.triton.mha",
        "path": "flash_attn_func",
        "batch": batch,
        "seqlen_q": seqlen_q,
        "seqlen_k": seqlen_k,
        "num_q_heads": nheads_q,
        "num_kv_heads": nheads_k,
        "query_group_size": int(nheads_q // nheads_k),
        "head_dim": head_dim_q,
        "init_head_dim": head_dim,
        "is_causal": is_causal,
        "alibi": "slopes",
        "sm_scale": sm_scale,
        "dtype": str(q_c.dtype),
        "out_shape": tuple(packed.shape),
        "lse_shape": tuple(lse.shape),
        "config_source": config_source,
        "config_file": config_fpath,
        "config": config,
        "device": str(device),
    }
    return packed, ctx
