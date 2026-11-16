# aiter_impl.py — 1018_mha_fused_bwd 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用（契约：run(inputs, init_kwargs, device)
# → (out, ctx)），采集离线终审用的 aiter 基线；调用方拿 out 与 task reference 按
# task.yaml 容差比对，不一致就拒绝记录。
#
# ── 来源（证据见 benchmark/sources/1018_mha_fused_bwd.yaml）──────────────────
#   repo   : OpenDAS/aiter（remote: developer.sourcefind.cn/codes/OpenDAS/aiter.git）
#   commit : c39fff8c77df4e80617649e92fa3c2615f2c43d1
#   file   : aiter/ops/triton/mha_fused_bwd.py
#   sha256 : 565540d7c3336fdb6f92c89761b5aeadcdb18b5a23cc40be724609b305333b58
#
# ── 入口（公开 host 函数，mha_fused_bwd.py:1026）────────────────────────────
#   flash_attn_fused_backward(
#       do, q, k, v, o, softmax_lse, dq, dk, dv, dbias,
#       sm_scale, alibi_slopes, causal, cu_seqlens_q, cu_seqlens_k,
#       max_seqlen_q, max_seqlen_k, dropout_p,
#       philox_seed=0, philox_offset=0,
#       descale_q=None, descale_k=None, descale_v=None, descale_do=None,
#       USE_INT64_STRIDES=False, config=None)
#   非因果分支 → _bwd_kernel_dkdvdq_noncausal（kernel 定义 mha_fused_bwd.py:721，
#   启动点 mha_fused_bwd.py:1226）；delta=rowsum(dO∘O) 由 _bwd_preprocess
#   （kernel 定义 mha_fused_bwd.py:27，启动点 mha_fused_bwd.py:1135）先算。
#   官方调用点：mha.py:1187（_FlashAttnFunc.backward 的 _USE_FUSED_BWD_KERNEL 分支）；
#   官方测试：op_tests/triton_tests/test_mha.py:479 test_mha_backward(FUSED=True)。
#
# 入口需要的 o / softmax_lse 不在 task io 里（题面只给 q/k/v/do，输出是打包的
# dQ|dK|dV 一维张量），因此由 aiter **自己的**公开前向 flash_attn_func(...,
# return_lse=True)（mha.py:1235；官方测试 test_mha.py:526 同一用法）产出。
# 反向不是 autograd 反传：本适配器在 torch.no_grad() 下取 (o, lse)，再直接调用
# aiter 的 fused 反向 host 入口，全程不建计算图、不调 .backward()。
#
# ── layout 约定（全部按 aiter host 自身的 stride 解释）──────────────────────
#   mha_fused_bwd.py:1098-1108 非 varlen 分支：
#       q_strides    = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
#   即传给 kernel 的 (b, h, m, k) 是 **bshd** 的逻辑视图，题面 [B,S,H,D] 连续张量
#   直接可用，无需 permute；k/v 同理（GQA 时 Hq = Hk*group，由 noncausal kernel 的
#   `for hqid in range(hkid*GROUP_SIZE, hkid*GROUP_SIZE+GROUP_SIZE)`
#   （mha_fused_bwd.py:922）在组内对 query 头累加 dk/dv —— 与 reference 的
#   reshape(B,Hk,group,Sk,D).sum(2) 语义一致）。
#   o 必须是前向输出、输入 dtype、bshd [B,Sq,Hq,D]；softmax_lse 必须是
#   [B,Hq,Sq] fp32（host 侧 delta = zeros_like(softmax_lse)，两者 stride 必须一致，
#   mha_fused_bwd.py:1116-1122）。
#   dq/dk/dv 是同形同 dtype 的 bshd 缓冲：非因果 kernel 里 dq 走
#   tl.atomic_add（mha_fused_bwd.py:308），**必须先清零**；dk/dv 在 host 循环结束后
#   tl.store（mha_fused_bwd.py:1008-1010），清零只是保险。
#   打包与 reference 完全对齐：dQ|dK|dV 依序按各自 bshd 行主序平铺后 cat 成一维
#   （reference.py:102-105 的 transpose(1,2).reshape(-1) 对连续 bshd 缓冲即 reshape(-1)）。
#
# ── autotune config ────────────────────────────────────────────────────────
#   aiter 在 config=None 时读 {AITER_TRITON_CONFIGS_PATH}/{dev}-MHA-DEFAULT.json
#   （mha_fused_bwd.py:1014-1023 取 "bkwd_fused" 段；mha.py:875-891 取 "fwd" 段）。
#   dev 由 arch_info.get_device() 给出，gfx936 → "BW200"
#   （aiter/ops/triton/utils/arch_info.py:5-18），即
#   aiter/ops/triton/configs/BW200-MHA-DEFAULT.json —— **该文件在 pinned 检出里
#   不存在**（configs/ 根下只有 BW200-EXTEND_ATTENTION-*、BW200B-* 等），
#   `git ls-files "*MHA-DEFAULT*"` 为空，所以 config=None 的默认路径在 BW200 上会
#   FileNotFoundError。本适配器的处理（ctx["config_source"] 记录走的是哪条路）：
#     ① 部署侧若存在 {dev}-MHA-DEFAULT.json 且结构完整 → fwd/bwd 都传 config=None，
#        完全交给 aiter 自己的官方 tuned 配置（_get_config 是唯一权威来源）；
#     ② 否则退回内联兜底 tiling（见 _FALLBACK_*），偏保守但不影响正确性。
#   需要该文件：{AITER_TRITON_CONFIGS_PATH}/{dev}-MHA-DEFAULT.json，含
#   "fwd"."default" 与 "bkwd_fused"{preprocess_kernel.PRE_BLOCK,
#   dkdvdq_kernel_N64, dkdvdq_kernel_N128} —— 因此 needs_autotune_config=true。

import json
import math

import torch

# 兜底配置（**仅**在官方 {dev}-MHA-DEFAULT.json 读不到时使用，ctx 里会显式标注，
# 绝不静默换参）。数值取自同一套 kernel 的官方调参：
#   fwd : mha.py:1004-1023 源码注释里的默认 tile（fp16/bf16 默认组）
#   bwd : ROCm/aiter 上游 configs/<arch>/triton/attention/mha/DEFAULT.json 的
#         "bkwd_fused" 段（PRE_BLOCK=128、dkdvdq_kernel_N64/N128 各 BLOCK_M=16）
# 这些都是 tiling/启动参数，不影响正确性（kernel 自带 mask）；但绝对性能不是
# BW200 实测 tuned 值，基线因此偏保守。拿到 BW200-MHA-DEFAULT.json 后无需改
# 代码即自动切回官方 JSON（见 _resolve_configs）。
_FALLBACK_FWD = {
    "BLOCK_M": 128, "BLOCK_N": 64, "waves_per_eu": 2,
    "num_warps": 4, "num_ctas": 1, "num_stages": 1,
}
_FALLBACK_BWD = {
    "preprocess_kernel": {"PRE_BLOCK": 128},
    "dkdvdq_kernel_N64": {
        "BLOCK_M": 16, "BLOCK_N": 64, "BLK_SLICE_FACTOR": 1,
        "num_warps": 8, "num_stages": 1, "waves_per_eu": 2,
    },
    "dkdvdq_kernel_N128": {
        "BLOCK_M": 16, "BLOCK_N": 128, "BLK_SLICE_FACTOR": 1,
        "num_warps": 8, "num_stages": 1, "waves_per_eu": 2,
    },
}

_MAX_HEAD_DIM = 256  # task.yaml invariants: 1 <= head_dim <= 256


def _mha_config_path():
    """aiter 官方 MHA 调参 JSON 的路径；拿不到环境信息时返回 None。"""
    try:
        import aiter.ops.triton.utils.arch_info as arch_info
        from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH

        # dev: gfx936 -> "BW200"（arch_info.py:5-18）
        return f"{AITER_TRITON_CONFIGS_PATH}/{arch_info.get_device()}-MHA-DEFAULT.json"
    except Exception:
        return None


def _json_usable(tuned):
    """校验 JSON 是否含本题两段所需的键（缺键就不敢交给 aiter 读，退回兜底）。"""
    if not isinstance(tuned, dict):
        return False
    fwd_section = tuned.get("fwd")
    fwd = fwd_section.get("default") if isinstance(fwd_section, dict) else None
    if not (isinstance(fwd, dict) and "BLOCK_M" in fwd and "BLOCK_N" in fwd):
        return False
    bkwd = tuned.get("bkwd_fused")
    if not isinstance(bkwd, dict):
        return False
    pre = bkwd.get("preprocess_kernel")
    if not (isinstance(pre, dict) and isinstance(pre.get("PRE_BLOCK"), int)):
        return False
    for key in ("dkdvdq_kernel_N64", "dkdvdq_kernel_N128"):
        section = bkwd.get(key)
        if not (isinstance(section, dict)
                and {"BLOCK_M", "BLOCK_N", "BLK_SLICE_FACTOR"} <= set(section)):
            return False
    return True


def _resolve_configs():
    """返回 (fwd_cfg, bwd_cfg, config_source, json_path)。

    fwd_cfg/bwd_cfg 为 None 表示「交给 aiter 自己读官方 JSON」（config=None，
    mha_fused_bwd.py:1126-1127 / mha.py:1001-1002 的 _get_config 路径），即官方
    tuned 值原样生效、本适配器不掺任何参数；这是有 JSON 时的首选路径。
    读不到 / 结构不完整时返回内联兜底 dict，并在 config_source 里注明原因。
    """
    path = _mha_config_path()
    fallback = (dict(_FALLBACK_FWD), {k: dict(v) for k, v in _FALLBACK_BWD.items()})
    if path is None:
        return (*fallback, "inline_fallback:no_aiter_config_path", None)
    try:
        with open(path, "r") as f:
            tuned = json.load(f)
    except FileNotFoundError:
        return (*fallback, "inline_fallback:no_json", path)
    except Exception as exc:  # noqa: BLE001 - 解析失败也要能退化并说明
        return (*fallback, f"inline_fallback:{type(exc).__name__}", path)
    if _json_usable(tuned):
        return None, None, "official_json", path
    return (*fallback, "inline_fallback:json_incomplete", path)


def _pad_head(x, head_dim_padded):
    """把 head_dim 补齐到 8 的倍数（与官方 _FlashAttnFunc 的 pad 行为一致，
    mha.py:1107-1110）。只用 cat/zeros 做拼接，补零不改变前 D 维的注意力结果。"""
    if head_dim_padded == x.shape[-1]:
        return x
    pad = torch.zeros(
        (*x.shape[:-1], head_dim_padded - x.shape[-1]), dtype=x.dtype, device=x.device
    )
    return torch.cat([x, pad], dim=-1)


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方实现（fused 非因果反向）。

    inputs    : [q, k, v, do]，已搬到 device。
                q, do [B, Sq, Hq, D] / k, v [B, Sk, Hk, D]，同 dtype（fp16/bf16），
                bshd 连续。
    init_kwargs: {"head_size": int}（可选 "scale"；缺省 1/sqrt(head_size)，
                与 reference.Model.__init__ 一致）。

    返回 (out, ctx)：out 为与 reference 同形同 dtype 的一维张量
    （dQ|dK|dV 依序平铺拼接）。
    """
    # aiter 顶层 import 很重，一律函数内 import
    from aiter.ops.triton import mha as aiter_mha
    from aiter.ops.triton.mha import flash_attn_func
    from aiter.ops.triton.mha_fused_bwd import flash_attn_fused_backward

    if len(inputs) != 4:
        raise ValueError(f"1018 期望 4 个输入 [q, k, v, do]，收到 {len(inputs)} 个")
    q, k, v, do = inputs

    # ── 校验：题面约束（task.yaml io/invariants）越界就 raise，绝不静默算错 ──
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4 or do.dim() != 4:
        raise ValueError("1018 只支持密集定长 bshd 4-D 输入")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"1018 只支持 fp16/bf16 非量化输入，收到 {q.dtype}")
    if not (k.dtype == v.dtype == do.dtype == q.dtype):
        raise ValueError("q/k/v/do 必须同 dtype")

    batch, seqlen_q, num_q_heads, head_dim = q.shape
    batch_k, seqlen_k, num_kv_heads, head_dim_k = k.shape
    if (batch_k, head_dim_k) != (batch, head_dim) or tuple(v.shape) != tuple(k.shape):
        raise ValueError(f"k/v 必须是 [B, Sk, Hk, D]：q={tuple(q.shape)} k={tuple(k.shape)}")
    if tuple(do.shape) != tuple(q.shape):
        raise ValueError(f"do 必须与 q 同形：do={tuple(do.shape)} q={tuple(q.shape)}")
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
        raise ValueError(f"num_q_heads 必须是 num_kv_heads 的整数倍：{num_q_heads}/{num_kv_heads}")
    if not (1 <= head_dim <= _MAX_HEAD_DIM):
        raise ValueError(f"head_dim 必须在 1..{_MAX_HEAD_DIM}：{head_dim}")
    if seqlen_q < 1 or seqlen_k < 1:
        raise ValueError(f"seqlen_q/seqlen_k 必须 >= 1：{seqlen_q}/{seqlen_k}")

    # ── init_kwargs → aiter 参数（按名取参；缺 head_size 直接 raise）────────
    if "head_size" not in init_kwargs:
        raise ValueError(f"init_kwargs 缺 head_size（收到 {sorted(init_kwargs)}）")
    head_size = int(init_kwargs["head_size"])
    if head_size != head_dim:
        raise ValueError(f"init_kwargs['head_size']={head_size} 与 q 的 head_dim={head_dim} 不一致")
    scale = init_kwargs.get("scale", None)
    sm_scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_size)

    # 官方 _FlashAttnFunc 会把 head_dim 补到 8 的倍数再算（mha.py:1107-1110），
    # 反向同样在补齐后的 head_dim 上算，最后切回原 head_dim（mha.py:1214-1216）。
    head_dim_padded = head_dim if head_dim % 8 == 0 else head_dim + (8 - head_dim % 8)
    if head_dim_padded != head_dim:
        q_c = _pad_head(q, head_dim_padded).contiguous()
        k_c = _pad_head(k, head_dim_padded).contiguous()
        v_c = _pad_head(v, head_dim_padded).contiguous()
        do_c = _pad_head(do, head_dim_padded).contiguous()
    else:
        q_c, k_c, v_c, do_c = q.contiguous(), k.contiguous(), v.contiguous(), do.contiguous()

    # fwd/bwd 的 config：None 表示由 aiter 自己读官方 {dev}-MHA-DEFAULT.json
    fwd_cfg, bwd_cfg, cfg_src, tuned_path = _resolve_configs()

    # 官方路径的全局开关（mha.py:29 默认 True）
    use_int64_strides = bool(getattr(aiter_mha, "_USE_INT64_STRIDES", False))

    # ── 1) aiter 前向拿 o / softmax_lse（反向入口的必需输入）──────────────────
    # 显式 no_grad：不做任何 autograd 反传，只取前向数值。
    with torch.no_grad():
        fwd_out = flash_attn_func(
            q_c,
            k_c,
            v_c,
            dropout_p=0.0,
            softmax_scale=sm_scale,
            causal=False,
            return_lse=True,
            return_attn_probs=False,
            config=fwd_cfg,
        )
    if not isinstance(fwd_out, (tuple, list)) or len(fwd_out) < 2:
        raise RuntimeError(
            "flash_attn_func(return_lse=True) 未返回 (out, softmax_lse)，"
            f"实际返回 {type(fwd_out)}"
        )
    o, softmax_lse = fwd_out[0], fwd_out[1]
    if tuple(o.shape) != tuple(q_c.shape):
        raise RuntimeError(f"前向输出 o 形状异常：{tuple(o.shape)} vs q {tuple(q_c.shape)}")
    if tuple(softmax_lse.shape) != (batch, num_q_heads, seqlen_q):
        raise RuntimeError(
            f"softmax_lse 形状异常：{tuple(softmax_lse.shape)}，"
            f"期望 {(batch, num_q_heads, seqlen_q)}"
        )
    softmax_lse = softmax_lse.contiguous()
    if softmax_lse.dtype != torch.float32:
        softmax_lse = softmax_lse.to(torch.float32)

    # ── 2) aiter fused 反向：一次调用同时算 dQ/dK/dV ─────────────────────────
    # dq 由 kernel 内 tl.atomic_add 累加，必须清零；dk/dv 由 kernel 直接 store。
    dq = torch.zeros_like(q_c)
    dk = torch.zeros_like(k_c)
    dv = torch.zeros_like(v_c)

    flash_attn_fused_backward(
        do_c,                      # do
        q_c,                       # q
        k_c,                       # k
        v_c,                       # v
        o,                         # o（前向输出，输入 dtype，bshd）
        softmax_lse,               # softmax_lse [B, Hq, Sq] fp32
        dq,                        # dq（原子加目标，已清零）
        dk,                        # dk
        dv,                        # dv
        None,                      # dbias（源码 mha_fused_bwd.py:1054 非 None 直接 raise）
        sm_scale,                  # sm_scale
        None,                      # alibi_slopes（pinned fused 路径不使用）
        False,                     # causal=False → _bwd_kernel_dkdvdq_noncausal
        None,                      # cu_seqlens_q=None → IS_VARLEN=False（密集定长）
        None,                      # cu_seqlens_k
        seqlen_q,                  # max_seqlen_q
        seqlen_k,                  # max_seqlen_k
        0.0,                       # dropout_p
        philox_seed=0,
        philox_offset=0,
        descale_q=None,
        descale_k=None,
        descale_v=None,
        descale_do=None,
        USE_INT64_STRIDES=use_int64_strides,
        config=bwd_cfg,
    )

    # ── 3) 打包：与 reference 一致（dQ|dK|dV，各自 bshd 行主序平铺后 cat）────
    dq = dq[..., :head_dim].reshape(-1)
    dk = dk[..., :head_dim].reshape(-1)
    dv = dv[..., :head_dim].reshape(-1)
    out = torch.cat([dq, dk, dv])

    expected_numel = (
        batch * seqlen_q * num_q_heads * head_dim
        + 2 * batch * seqlen_k * num_kv_heads * head_dim
    )
    if out.numel() != expected_numel:
        raise RuntimeError(f"打包输出元素数 {out.numel()} != 期望 {expected_numel}")

    torch.cuda.synchronize()

    ctx = {
        "path": "flash_attn_fused_backward/noncausal_fused_dkdvdq",
        "entry": "aiter.ops.triton.mha_fused_bwd.flash_attn_fused_backward",
        "fwd_entry": "aiter.ops.triton.mha.flash_attn_func",
        "batch": batch,
        "seqlen_q": seqlen_q,
        "seqlen_k": seqlen_k,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "head_dim_padded": head_dim_padded,
        "sm_scale": sm_scale,
        "dtype": str(q.dtype),
        "varlen": False,
        "causal": False,
        "use_int64_strides": use_int64_strides,
        "config_source": cfg_src,
        "config_json_path": tuned_path,
        "config_fwd": fwd_cfg,
        "config_bwd": bwd_cfg,
        "config_note": (
            "config=None → aiter 自读官方 {dev}-MHA-DEFAULT.json（tuned 值原样生效）"
            if cfg_src == "official_json"
            else "官方 JSON 不可用 → 内联兜底 tiling（正确性不受影响，性能偏保守）"
        ),
        "packing": "cat([dQ, dK, dV]).reshape(-1) per bshd row-major",
    }
    return out, ctx
