# aiter_impl.py — 4004_batched_gemm_bf16 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
# 适配器契约（统一为「按名取参」，见 record_baseline.py 与 audit_model_class.py
# case_init_kwargs 的说明）：
#     run(inputs, init_kwargs: dict, device) -> (out, ctx)
#
# 来源：OpenDAS/aiter @ c39fff8c77df4e80617649e92fa3c2615f2c43d1
#   aiter/ops/triton/batched_gemm_bf16.py
#   sha256 8ab7b67f5ea7e2504b83e58d7984e3f25160b0417e41d67a7432f5431a07d55d
#   （与 benchmark/sources/4004_batched_gemm_bf16.yaml 的 device_kernel 证据一致）
#
# 公开 host 入口（kernel 是模块私有的 _batched_gemm_bf16_kernel，勿直接调用）：
#   aiter/ops/triton/batched_gemm_bf16.py:186
#     batched_gemm_bf16(XQ, WQ, bias=None, dtype=torch.bfloat16, splitK=None,
#                       YQ=None, config=None) -> YQ
#   语义：YQ[i] = XQ[i] @ WQ[i]^T，XQ (B,M,K) / WQ (B,N,K) 行主 / bias (B,1,N)
#   / YQ (B,M,N)；WQ 由入口内部 transpose(1,2) 成 (B,K,N)（:222），故调用侧
#   无需自行转置，保持题面布局直传即可。splitK 在源码中被断言禁用（:219）。
#
#   与题目 reference.py 的数值对齐依据（逐条对应源码行）：
#     - 累加 dtype：kernel :125 `acc_dtype = fp32`（c 非 int8 时）→ 全程 float32；
#     - bias 融合顺序：kernel :145-148 `accumulator.to(bias_ptr.element_ty) + bias`
#       ↔ reference `acc.to(x.dtype) + bias`（fp32 结果先 cast 到输入 dtype 再加
#       bias，bf16 算术）；
#     - 末段 cast：kernel :150 `accumulator.to(c_ptr.element_ty)`，而 c 的
#       element_ty 就是入口 dtype 参数（YQ 由 :231 按 dtype 分配）↔ reference
#       的 `.to(self.out_dtype)`。故 bfloat16 / float16 两种输出均与 reference
#       同形、同 dtype 对齐（入口 :215-218 的 assert 白名单也正好是这两种）。
#     - 输出 YQ (B,M,N)：入口 YQ=None 时自行分配（:230-231）并返回（:263）。
#
# 本适配器只做「取参 / dtype-layout 归一 / 传参 / 校验」，不做任何核心计算。
#
# 已知环境缺口（重要）：入口默认 config 来自 _get_config(M, N, K)（:166-183）
#   → f"{AITER_TRITON_CONFIGS_PATH}/gemm/{arch_info.get_device()}-BATCHED_GEMM-A16W16.json"，
#   而该函数**没有**缺文件兜底。pinned commit 的 configs/gemm/ 下只存在
#   MI300X- / MI350X- 两个 A16W16 文件；gfx936 → arch_info.get_device()=="BW200"
#   （aiter/ops/triton/utils/arch_info.py:5-10），BW200- 文件不存在，因此默认
#   路径在 DCU 上必然 FileNotFoundError。
#   → 适配器检测到设备专属文件缺失时，改读 aiter 自带的
#     MI300X-BATCHED_GEMM-A16W16.json，并按官方同一选档规则（`M + N >= 4096`
#     取 "large"，否则 "small"，见 :180-183）经公开参数 config= 传入
#     （入口 :233 仅在 config is None 时才去读文件）。设备专属文件若存在，
#     仍走官方默认路径，不做任何干预。

import json
import os

import torch

# 入口全名，写入 ctx 便于基线复核
_ENTRY = "aiter.ops.triton.batched_gemm_bf16.batched_gemm_bf16"
_ENTRY_FILE = "aiter/ops/triton/batched_gemm_bf16.py"

# 入口 dtype 白名单（源码 :215-218）→ 题目 init_kwargs["out_dtype"] 的取值
_SUPPORTED_OUT_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
# 输入侧恒为 bfloat16（题目 io.inputs 与 make_inputs 的 dtype）
_IN_DTYPE = torch.bfloat16

# _get_config 的选档阈值（源码 :180）
_LARGE_SHAPE_THRESHOLD = 4096
# config 文件名：f"{device}-BATCHED_GEMM-A16W16.json"（源码 :174）
_CONFIG_BASENAME = "BATCHED_GEMM-A16W16.json"
# 设备专属 config 缺失时的兜底表：aiter 随仓库分发的 MI300X 调参
_FALLBACK_DEVICE_TAG = "MI300X"


def _fallback_config(configs_path: str, M: int, N: int, K: int):
    """设备专属 config JSON 缺失时的兜底：读 aiter 自带的 MI300X 调参表选档。

    返回 (config_dict, source_str)。选档规则与官方 _get_config 一致
    （M + N >= 4096 → "large"，否则 "small"），只是换了文件名来源。
    """
    path = os.path.join(
        configs_path, "gemm", f"{_FALLBACK_DEVICE_TAG}-{_CONFIG_BASENAME}"
    )
    if not os.path.exists(path):
        raise RuntimeError(
            f"既没有设备专属的 {_CONFIG_BASENAME}，也没有兜底表 {path}；"
            "aiter 的 batched_gemm_bf16 入口无缺配置兜底，无法采集基线"
        )
    with open(path, "r") as f:
        table = json.load(f)
    key = "large" if M + N >= _LARGE_SHAPE_THRESHOLD else "small"
    if key not in table:
        raise RuntimeError(f"{path} 缺少 '{key}' 档：{sorted(table)}")
    return dict(table[key]), f"{_FALLBACK_DEVICE_TAG}:{_CONFIG_BASENAME}:{key}"


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方 batched_gemm_bf16。

    inputs     : [x, weight, bias]——顺序 = reference.make_inputs 的消费顺序
                 x      (B, M, K) bfloat16 行主
                 weight (B, N, K) bfloat16 行主（转置 GEMM 语义）
                 bias   (B, 1, N) bfloat16；use_bias=False 时为占位张量，忽略
    init_kwargs: {"use_bias": bool, "out_dtype": "bfloat16" | "float16"}
                 （即题面 Model(use_bias, out_dtype) 的构造参数）
    device     : 评测设备（torch.device）；入口从 XQ.device 分配输出，此处仅记录

    返回 (out, ctx)：out 为 (B, M, N)、dtype = out_dtype 的单个张量，
    与 reference.forward 的输出同形同 dtype。
    """
    # aiter 顶层 import 很重，真机部署走最小导入垫片：一律函数内 import
    from aiter.ops.triton.batched_gemm_bf16 import batched_gemm_bf16
    from aiter.ops.triton.utils import arch_info
    from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH

    if len(inputs) != 3:
        raise ValueError(f"4004 需要 [x, weight, bias] 三个输入，收到 {len(inputs)} 个")
    x, weight, bias = inputs

    # ---- init_kwargs（按名取参；越界即 raise，绝不静默用错参数）----
    use_bias = bool(init_kwargs.get("use_bias", True))
    out_dtype_name = str(init_kwargs.get("out_dtype", "bfloat16"))
    if out_dtype_name not in _SUPPORTED_OUT_DTYPES:
        raise ValueError(
            f"out_dtype={out_dtype_name!r} 不在 aiter 入口白名单 "
            f"{sorted(_SUPPORTED_OUT_DTYPES)} 内（batched_gemm_bf16.py:215-218）"
        )
    out_dtype = _SUPPORTED_OUT_DTYPES[out_dtype_name]

    # ---- shape / dtype 归一（题面 (B,M,K)/(B,N,K)/(B,1,N) 即 aiter 约定布局）----
    if x.dim() != 3 or weight.dim() != 3:
        raise ValueError(f"x/weight 必须是 3 维，收到 {x.shape} / {weight.shape}")
    B, M, K = x.shape
    N = weight.shape[1]
    if tuple(weight.shape) != (B, N, K):
        raise ValueError(f"weight 形状 {tuple(weight.shape)} 与 (B,N,K)=({B},{N},{K}) 不符")
    if K == 0 or N == 0 or M == 0 or B == 0:
        raise ValueError(f"零维输入不支持：x={tuple(x.shape)} weight={tuple(weight.shape)}")

    # x/weight 恒为 bfloat16：kernel 按 element_ty 载入后 tl.dot(fp32 累加)。
    # 已是 bf16 时 .to() 为 no-op，仅防御性归一；contiguous 由入口自身保证
    # （:209-210），此处不重复搬数据。
    x = x.to(_IN_DTYPE)
    weight = weight.to(_IN_DTYPE)

    if use_bias:
        # bias 必须与 reference 一样先 cast 到输入 dtype（bf16）：kernel :148
        # 用 bias_ptr.element_ty 做加前的 cast 目标，dtype 不符会改变融合顺序
        if bias is None:
            raise ValueError("use_bias=True 但 bias 为 None")
        if tuple(bias.shape) != (B, 1, N):
            raise ValueError(f"bias 形状 {tuple(bias.shape)} 与 (B,1,N)=({B},1,{N}) 不符")
        bias_arg = bias.to(_IN_DTYPE).contiguous()
    else:
        # use_bias=False：题面传入占位张量但 reference 忽略它；kernel 侧
        # HAS_BIAS=False 且不触碰 bias 指针（:145-148, :258），故传 None
        bias_arg = None

    # ---- config：优先官方默认路径，设备专属 JSON 缺失时用显式兜底 ----
    dev_tag = arch_info.get_device()
    primary = os.path.join(
        AITER_TRITON_CONFIGS_PATH, "gemm", f"{dev_tag}-{_CONFIG_BASENAME}"
    )
    if os.path.exists(primary):
        config = None  # 交给官方 _get_config(M, N, K)（同一文件、同一选档规则）
        config_source = f"{dev_tag}:{_CONFIG_BASENAME}"
    else:
        config, config_source = _fallback_config(AITER_TRITON_CONFIGS_PATH, M, N, K)

    # ---- 调用官方 host 入口（核心计算全在 _batched_gemm_bf16_kernel 内）----
    out = batched_gemm_bf16(x, weight, bias_arg, out_dtype, config=config)

    torch.cuda.synchronize()

    if out.shape != (B, M, N) or out.dtype != out_dtype:
        raise RuntimeError(
            f"aiter 输出 {tuple(out.shape)}/{out.dtype} 与题目契约 "
            f"({B}, {M}, {N})/{out_dtype} 不一致"
        )

    even_k = None
    if config is not None:
        even_k = (K % int(config["BLOCK_SIZE_K"]) == 0)
    ctx = {
        "aiter_entry": _ENTRY,
        "aiter_file": _ENTRY_FILE,
        "path": "TN 单 kernel（无 splitK）",
        "layout": "x (B,M,K) 行主 + weight (B,N,K) 行主；入口内部 transpose(1,2)",
        "dtype_in": "bfloat16",
        "dtype_out": out_dtype_name,
        "use_bias": use_bias,
        "shapes": {"B": B, "M": M, "N": N, "K": K},
        "device_tag": dev_tag,
        "config_source": config_source,
        "config": config if config is not None else "aiter._get_config(M, N, K)",
        "even_k": even_k,
        "device": str(device),
    }
    return out, ctx
