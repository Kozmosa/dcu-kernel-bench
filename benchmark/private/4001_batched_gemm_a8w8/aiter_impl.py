# aiter_impl.py — 4001_batched_gemm_a8w8 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
# 适配器契约：run(inputs, init_kwargs: dict, device) —— 按名取参（构造参数可能
# 稀疏给出，位置式取值会错位，见 audit_model_class.py 的 case_init_kwargs）。
#
# ── 来源（aiter pinned 检出 .dcu_runs/aiter_pinned，commit c39fff8c）─────────
#   文件 : aiter/ops/triton/batched_gemm_a8w8.py
#   sha256: 81bac315146a0858445159b535adb2e68241b351051c304c664428146cc9ae16
#   入口 : batched_gemm_a8w8（:207-296），device kernel _batched_gemm_a8w8_kernel（:22-184）
#
#   入口签名（逐字，源码 :207-217）：
#     batched_gemm_a8w8(
#         XQ: torch.Tensor,                      # (B, M, K) int8
#         WQ: torch.Tensor,                      # (B, N, K) int8（行主，等价转置 GEMM）
#         x_scale: torch.Tensor,                 # (B, M, 1) float32，per-token
#         w_scale: torch.Tensor,                 # (B, 1, N) float32，per-channel
#         bias: Optional[torch.Tensor] = None,   # (B, 1, N)，输出 dtype
#         dtype: Optional[torch.dtype] = torch.bfloat16,   # 输出 dtype
#         splitK: Optional[int] = None,          # 断言必须 None（:248，无 Triton splitK 实现）
#         YQ: Optional[torch.Tensor] = None,     # 输出张量；None 时内部 empty 并返回
#         config: Optional[dict] = None,         # tile 配置；None 时读 autotune JSON
#     ) -> YQ  # (B, M, N)，dtype
#
#   官方调用约定（op_tests/triton_tests/test_batched_gemm_a8w8.py:70，sha256
#   27a10483575469ad37ab7f8e6c121f704dcc26ade5f6eab166e616e1a4c724bf）：
#     batched_gemm_a8w8(x, weight, x_scale, w_scale, bias, dtype, YQ=y)
#   官方语义权威 run_torch（同文件 :54-66）：逐 batch F.linear(fp32) × 外乘 scale
#   → cast 到 bias dtype → 加 bias → cast 到 dtype / dtype 的 cast。与题目
#   reference.forward 的融合顺序一致（fp32 累加 → 外乘 scale → cast → bias）。
#
# ── 布局对齐（本题无需任何 permute/reshape）──────────────────────────────────
#   aiter 入口要的正是题目 io 声明的 TN 布局：
#     XQ (B, M, K) 行主 —— 直接传 xq
#     WQ (B, N, K) 行主 —— 直接传 wq；宿主自己在 :251 做 WQ.transpose(1, 2)
#                            得到 (B, K, N) 的非连续视图，并把 stride(0/1/2) 传给
#                            kernel（:283-285，stride_bk=1、stride_bn=K），
#                            **不需要**调用方预先转置。
#     x_scale (B, M, 1) / w_scale (B, 1, N) / bias (B, 1, N) —— 直接传。
#   输出 te 由宿主 torch.empty((B, M, N), dtype, device) 分配（:260），连续。
#
#   kernel 内两处指针算术只吃「batch stride + 平面内偏移」，不吃 plane stride
#   （:140-141 的 a_scale_ptr + batch_id*stride_ascaleb + offs_a_scale、
#   :168 的 bias 同理），这要求三个张量沿最后一维 stride == 1。题面 io 声明它们
#   是 contiguous，故严格成立；本适配器仍显式 .contiguous() 兜住。
#   （:138-139 的 `pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) % M` 中取模
#   只作用于 arange，尾部块偏移会超出平面范围、但落在同一 batch 平面内且该位置值
#   不被使用——因为 kernel 只用它们的行/列子集；N 非 BLOCK_SIZE_N 整数倍时同理。
#   这是官方写法，不改写。）
#
# ── autotune config 依赖（needs_autotune_config = True）─────────────────────
#   宿主在 config=None 时调 _get_config(M, N, K)（:262-263 调用点，:187-204 定义），
#   读 `{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-BATCHED_GEMM-A8W8.json`，按
#   `M + N >= 4096` 选 "large"/"small" 键。dev 由 arch_info.get_device() 给出，
#   而 arch_info.py:5-10 的 _ARCH_TO_DEVICE 把 gfx936 映射为 **"BW200"** ——
#   即 DCU 真机会找 `gemm/BW200-BATCHED_GEMM-A8W8.json`。pinned 检出的
#   aiter/ops/triton/configs/gemm/ 下**只有 MI300X- 与 MI350X- 前缀的
#   BATCHED_GEMM-A8W8.json，没有 BW200- 前缀**（已用 git ls-tree / 目录列举核对），
#   故 config=None 在 DCU 上会 FileNotFoundError（:197 open 无 try/except）。
#
#   因此本适配器与 1014_hstu_attention / 1017_mha / 1018_mha_fused_bwd 同款处理：
#     ① 设备同名 JSON 在 → config=None，完全走 aiter 自己的官方 tuned 配置；
#     ② 不在 → 显式传 config（宿主形参 config 就是给外部供 config 用的，官方
#        API 与官方测试都这么用）。回退值逐字段抄自同 commit 的
#        configs/gemm/MI300X-BATCHED_GEMM-A8W8.json（MI350X 同文件逐字段相同，
#        两份都是 large={256,256,64,4} / small={128,128,32,1}）。
#   config 只决定分块与占用（性能），不改变数值语义；ctx["config_source"] 如实
#   标注来源，绝不静默换参。

import os

import torch

# 官方 config JSON 缺失时的显式回退（逐字段抄自
# aiter/ops/triton/configs/gemm/MI300X-BATCHED_GEMM-A8W8.json，MI350X 同值）。
# 键名 = kernel 的 meta-parameters（BLOCK_SIZE_*/GROUP_SIZE_M）+ triton 编译选项。
_FALLBACK_CONFIGS = {
    # M + N >= 4096（_get_config :201-202）
    "large": {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 4,
        "num_warps": 8,
        "num_stages": 2,
        "waves_per_eu": 2,
        "matrix_instr_nonkdim": 16,
    },
    # M + N < 4096
    "small": {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 1,
        "num_warps": 8,
        "num_stages": 2,
        "waves_per_eu": 2,
        "matrix_instr_nonkdim": 16,
    },
}

# 题目 io 声明只允许这两种输出 dtype（task.yaml io.outputs / 宿主 :244-247 的断言）
_SUPPORTED_OUT_DTYPES = ("bfloat16", "float16")


def _resolve_config(M, N, K):
    """返回 (config, source, fpath)。

    config is None  → 交回 aiter `_get_config()` 读官方 tuned JSON；
    否则为显式 config dict（官方 JSON 缺失，见文件头 autotune 段）。
    aiter 一律函数内 import（顶层 import 很重，真机部署走最小导入垫片）。
    """
    fpath = None
    try:
        import aiter.ops.triton.utils.arch_info as arch_info  # noqa: PLC0415
        from aiter.ops.triton.utils.core import (  # noqa: PLC0415
            AITER_TRITON_CONFIGS_PATH,
        )

        dev = arch_info.get_device()  # gfx936 -> "BW200"、gfx938 -> "BW200B"
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-BATCHED_GEMM-A8W8.json"
        if os.path.exists(fpath):
            return None, f"official_autotune_json:{fpath}", fpath
        source = f"fallback_config(no {os.path.basename(fpath)})"
    except Exception as exc:  # pragma: no cover - 仅探测，失败即回退
        source = f"fallback_config(probe {type(exc).__name__})"

    # 桶选择规则逐字复制 _get_config（:201-204）
    key = "large" if (M + N) >= 4096 else "small"
    return dict(_FALLBACK_CONFIGS[key]), f"{source}:{key}", fpath


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方实现（aiter.ops.triton.batched_gemm_a8w8.batched_gemm_a8w8）。

    inputs     : [xq, wq, x_scale, w_scale, bias]（顺序同 reference.make_inputs）
                 xq [B, M, K] int8 / wq [B, N, K] int8（行主，转置 GEMM 语义）
                 x_scale [B, M, 1] fp32（per-token）/ w_scale [B, 1, N] fp32（per-channel）
                 bias [B, 1, N] 输出 dtype（use_bias=False 时为占位张量，入口忽略）
    init_kwargs: {"use_bias": bool, "out_dtype": str}
                 （Model.__init__(use_bias: bool = True, out_dtype: str = "bfloat16")）
    device     : 目标设备（张量已在 device 上，仅用于 ctx 记录）

    返回 (out, ctx)；out 为 [B, M, N] 连续张量，dtype = init_kwargs["out_dtype"]，
    与 reference.forward 同形同 dtype（reference 返回单个未打包张量，本题无需打包）。
    """
    from aiter.ops.triton.batched_gemm_a8w8 import batched_gemm_a8w8  # noqa: PLC0415

    xq, wq, x_scale, w_scale, bias = inputs

    # ---- 构造参数（按名取参；越界一律 raise，绝不静默用错）-------------------
    use_bias = bool(init_kwargs.get("use_bias", True))
    out_dtype_name = init_kwargs.get("out_dtype", "bfloat16")
    if out_dtype_name not in _SUPPORTED_OUT_DTYPES:
        raise ValueError(
            f"out_dtype={out_dtype_name!r} 不在题目允许的 {_SUPPORTED_OUT_DTYPES} 内"
            "（宿主 :244-247 会对其它 dtype 直接断言失败）"
        )
    out_dtype = getattr(torch, out_dtype_name)

    # ---- 形状校验（题目不变式；不满足则 raise，不猜）------------------------
    if xq.dim() != 3 or wq.dim() != 3:
        raise ValueError(f"xq/wq 必须是 3 维 (B, M, K) / (B, N, K)，实际 {xq.shape} / {wq.shape}")
    B, M, K = xq.shape
    Bw, N, Kw = wq.shape
    if Bw != B:
        raise ValueError(f"batch 维不一致：xq {B} vs wq {Bw}（宿主 :242 断言）")
    if Kw != K:
        raise ValueError(f"K 维不一致：xq {K} vs wq {Kw}（宿主 :243 断言）")
    if min(B, M, N, K) < 1:
        raise ValueError(f"B/M/N/K 必须 >= 1，实际 {(B, M, N, K)}")
    if tuple(x_scale.shape) != (B, M, 1):
        raise ValueError(f"x_scale 必须是 (B, M, 1)={(B, M, 1)}，实际 {tuple(x_scale.shape)}")
    if tuple(w_scale.shape) != (B, 1, N):
        raise ValueError(f"w_scale 必须是 (B, 1, N)={(B, 1, N)}，实际 {tuple(w_scale.shape)}")

    # ---- dtype / 布局规整（允许的 torch 用途：layout 转换与打包）-------------
    # 评测器把输入统一 cast 成 fp32 后传入，码字在 [-20, 20) 内 fp32 可精确表示，
    # 故 .to(int8) 无损（题目 docstring 明确声明此约定）。
    xq = xq.to(torch.int8).contiguous()
    wq = wq.to(torch.int8).contiguous()
    # scale 必须 fp32 且最后一维 stride==1（kernel 只吃 batch stride，见文件头）
    x_scale = x_scale.to(torch.float32).contiguous()
    w_scale = w_scale.to(torch.float32).contiguous()

    if use_bias:
        # 与 reference 一致：bias 先 cast 到输出 dtype（reference :66 同款）
        if tuple(bias.shape) != (B, 1, N):
            raise ValueError(f"bias 必须是 (B, 1, N)={(B, 1, N)}，实际 {tuple(bias.shape)}")
        bias_arg = bias.to(out_dtype).contiguous()
    else:
        # use_bias=False：传 None，走 kernel 的 HAS_BIAS=False 分支（不读 bias）
        bias_arg = None

    # ---- autotune config ----------------------------------------------------
    config, config_source, config_fpath = _resolve_config(M, N, K)

    out = batched_gemm_a8w8(
        xq,
        wq,
        x_scale,
        w_scale,
        bias_arg,
        out_dtype,
        splitK=None,   # 断言必须 None（:248，Triton 侧无 splitK 实现）
        YQ=None,       # None -> 宿主 torch.empty((B, M, N), out_dtype) 并返回
        config=config,
    )

    torch.cuda.synchronize()
    return out, {
        "path": "aiter.ops.triton.batched_gemm_a8w8",
        "entry": "batched_gemm_a8w8",
        "layout": "TN（xq (B,M,K) / wq (B,N,K) 行主，宿主内部转置 WQ）",
        "B": B,
        "M": M,
        "N": N,
        "K": K,
        "use_bias": use_bias,
        "out_dtype": out_dtype_name,
        "splitK": None,
        "config_source": config_source,
        "config_file": config_fpath,
        "config": config,
    }
