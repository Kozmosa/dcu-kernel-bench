# aiter_impl.py — 4018_quant 的 aiter 官方实现适配器
#
# 来源（准入记录 benchmark/sources/4018_quant.yaml）：
#   repo   OpenDAS/aiter @ c39fff8c77df4e80617649e92fa3c2615f2c43d1
#   file   aiter/ops/triton/quant.py
#          sha256 f95d4f81f54af62a3f0570d48fc6110c20ed9a2a357dea2c9a1aac768b951b6c（已核对）
#   语义权威 op_tests/triton_tests/test_quant.py
#          sha256 eddff9d2c47670a6617bd23c7a0a1882fc062bec42aed38b87ba341a760c4344（已核对）
#          其中 torch_static_per_tensor_quant_fp8_i8 / torch_dynamic_per_tensor_quant_fp8_i8 /
#          torch_dynamic_per_token_quant_fp8_i8 即题面 reference.py::Model 的逐行来源。
#
# 三个公开 host 入口（都在 aiter/ops/triton/quant.py，调用约定取自官方测试）：
#
#   static_per_tensor_quant_fp8_i8(qx, x_in, scale_in)          # L52-75
#       qx [M, N] 由调用方按目标 dtype（int8 / float8_e4m3fn）预分配，
#       x_in [M, N] float，scale_in [1] fp32（numel 断言 == 1）；
#       kernel 内是 x * (1/scale) 倒数乘法（L44-47），非直接除法。
#   dynamic_per_tensor_quant_fp8_i8(qx, x_in, scale_out)        # L98-135
#       scale_out [1] fp32 由调用方预分配且必须先置 0——kernel 用
#       tl.atomic_max 归约（L95），再内部复用 static kernel 完成量化；
#       DTYPE_MAX 由 qx.dtype 推出（L124-128）：int8 → iinfo.max=127，
#       fp8 → finfo.max=448，与题面 dtype_max 一致。
#   dynamic_per_token_quant_fp8_i8(qx, x_in, scale_out)         # L260-360
#       scale_out [M] fp32；按 cols/rows/CU 数分派 1d（L139）/ n128（L164）/
#       4row（L207）三条 kernel，三者公式一致：逐行 amax →
#       max(amax, 1e-10)/DTYPE_MAX → x * (1/s) 倒数乘法 → int8 nearbyint
#       （round-half-even）/ fp8 直接 .to(e4m3)。三个入口都断言 x/qx/scale_out
#       连续。
#
# 题面语义 ↔ aiter 入口映射：
#   static_per_tensor   → static_per_tensor_quant_fp8_i8(qx, x, scale)
#   dynamic_per_tensor  → dynamic_per_tensor_quant_fp8_i8(qx, x, scale_out)
#   dynamic_per_token   → dynamic_per_token_quant_fp8_i8(qx, x, scale_out)
# （aiter/ops/quant.py 的 per_tensor_quant_triton / per_token_quant_triton 只是
#   这三个入口的薄包装，准入记录已声明题面取 triton 层入口，故直接调用。）
#
# 布局与打包：题面 x 是 2D 连续 [rows, cols]，与 aiter 的 [M, N] 契约一致，
# 无需 permute/reshape。题面输出是单张量两段打包的一维 fp32
#   out[:rows*cols] = 码字段（int8 整数或 e4m3 精确值提升为 fp32）
#   out[rows*cols:] = scale 段（per-token rows 个行 scale；per-tensor/static 1 个）
# 适配器在外部用 cat 拼回（打包属输出整理，非核心计算）。
#
# 与 reference.py 的两处已知差异（准入 note 已记录，均不改输出契约）：
#   1) x 统一 cast 到 fp32 再喂 kernel。reference 的中间算术全程 fp32
#      （x.to(torch.float32)），题面实现约束也要求 fp32；cast 同时消除 kernel 内
#      m / DTYPE_MAX 的标量提升歧义（fp16/bf16 张量除 python float 常量的结果
#      精度取决于 Triton 的标量提升规则）。在线评测器与候选 forward 都要做同样
#      的 fp32 cast，故基线与候选口径一致。
#   2) static / dynamic_per_tensor 路径：kernel 用 x * (1/scale) 倒数乘法，
#      reference 用 x / scale 直接除法，相差 <=1 ulp。码字段只在商恰好落在取整
#      tie 的 1 ulp 邻域内才可能差 1（每元素概率 ~2^-24，本任务 case 规模下可忽略）；
#      dynamic_per_token 两条实现都是倒数乘法，逐位一致。该差异无法在不改 aiter
#      kernel 的前提下消除，题面 tolerance note 已记录其来源。
#
# 无 @triton.autotune、不读 AITER_TRITON_CONFIGS_PATH：三个入口全部是显式
# num_warps/num_stages 的普通 jit 启动，不需要任何 config JSON。

import torch

_QUANT_DTYPES = {"int8": torch.int8, "float8_e4m3": torch.float8_e4m3fn}
_MODES = ("static_per_tensor", "dynamic_per_tensor", "dynamic_per_token")


def _resolve_mode(init_kwargs: dict) -> str:
    mode = init_kwargs.get("mode", "dynamic_per_token")
    if mode not in _MODES:
        raise ValueError(f"4018_quant: mode 非法 {mode!r}，必须是 {_MODES} 之一")
    return mode


def _resolve_quant_dtype(init_kwargs: dict) -> str:
    quant_dtype = init_kwargs.get("quant_dtype", "int8")
    if isinstance(quant_dtype, torch.dtype):
        # 兼容以 torch.dtype 传入的调用方；题面/终审契约传的是字符串
        inverse = {torch.int8: "int8", torch.float8_e4m3fn: "float8_e4m3"}
        if quant_dtype not in inverse:
            raise ValueError(f"4018_quant: quant_dtype 非法 {quant_dtype}，只支持 int8 / float8_e4m3")
        return inverse[quant_dtype]
    if quant_dtype not in _QUANT_DTYPES:
        raise ValueError(
            f"4018_quant: quant_dtype 非法 {quant_dtype!r}，只支持 int8 / float8_e4m3"
        )
    return quant_dtype


def _per_token_path(rows: int, cols: int, torch_device) -> str:
    """复刻 aiter dynamic_per_token_quant_fp8_i8 的分派规则，仅用于 ctx 记录。

    阈值常量与 CU 计数辅助函数直接从 aiter 模块取（不复制字面量），避免与
    源码漂移；任何取不到的情况退化成 "unknown"，不影响主流程。
    """
    try:
        from aiter.ops.triton import quant as _q

        if cols == 128 and rows >= 256:
            return "n128_block16"
        if rows <= 1024:
            return "1d"
        num_cus = _q._per_token_quant_num_cus(torch.device(torch_device))
        if cols < 2048:
            four_row_limit = _q._PER_TOKEN_QUANT_4ROW_PROGRAMS_PER_CU * num_cus
        elif cols == 2048:
            four_row_limit = _q._PER_TOKEN_QUANT_4ROW_N2048_PROGRAMS_PER_CU * num_cus
        else:
            four_row_limit = None
        if four_row_limit is not None and rows >= four_row_limit:
            return "four_row_block4"
        return "1d"
    except Exception:
        return "unknown"


def run(inputs, init_kwargs: dict, device):
    """执行一次 aiter 官方实现。

    inputs      : [x]（dynamic_* 模式）或 [x, scale]（static_per_tensor 模式）
                  x 连续 2D [rows, cols]，scale [1] fp32 正数
    init_kwargs : {"mode": str, "quant_dtype": str}

    返回 (out, ctx)：out 是与 reference 同形同 dtype 的一维 fp32 张量
    [rows*cols + S]（码字段 | scale 段，S = rows 或 1）。
    """
    # aiter 顶层 import 很重，按最小导入垫片约定放在函数内
    from aiter.ops.triton.quant import (
        dynamic_per_tensor_quant_fp8_i8,
        dynamic_per_token_quant_fp8_i8,
        static_per_tensor_quant_fp8_i8,
    )

    mode = _resolve_mode(init_kwargs)
    quant_dtype = _resolve_quant_dtype(init_kwargs)
    if not isinstance(inputs, (list, tuple)) or len(inputs) == 0:
        raise ValueError("4018_quant: inputs 必须是非空列表，首元素为 x")

    x = inputs[0]
    if x.dim() != 2:
        raise ValueError(f"4018_quant: x 必须是 2D，实际 shape={tuple(x.shape)}")
    rows, cols = int(x.shape[0]), int(x.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"4018_quant: x 的 rows/cols 必须 >= 1，实际 {rows}x{cols}")

    # 与 reference 的 fp32 中间算术对齐（见文件头「已知差异 1」），并满足 aiter
    # 三个入口对 x 连续性的断言
    x_f32 = x.to(torch.float32).contiguous()
    qx = torch.empty((rows, cols), dtype=_QUANT_DTYPES[quant_dtype], device=device)

    if mode == "static_per_tensor":
        if len(inputs) < 2:
            raise ValueError("4018_quant: static_per_tensor 模式需要 inputs[1] 为 scale [1] fp32")
        scale_in = inputs[1].to(torch.float32).reshape(-1)
        if scale_in.numel() != 1:
            raise ValueError(f"4018_quant: scale 必须是单元素，实际 numel={scale_in.numel()}")
        # 不做 (scale > 0).all() 之类的取值校验：那会在计时路径上引入一次
        # device→host 同步；正 scale 由题面输入生成器保证（make_inputs 取
        # 0.5+rand ∈ [0.5, 1.5)），且 scale<=0 时 reference 与 kernel 同样溢出。
        scale_seg = scale_in.contiguous()
        # 该入口返回 qx；scale 段就是输入 scale 原样透传（与 reference 一致）
        static_per_tensor_quant_fp8_i8(qx, x_f32, scale_seg)
        entry = "static_per_tensor_quant_fp8_i8"
        dispatch = "n/a"
    elif mode == "dynamic_per_tensor":
        # atomic_max 的累加起点必须是 0（aiter kernel 只做 max，不清零）
        scale_seg = torch.zeros(1, dtype=torch.float32, device=device)
        dynamic_per_tensor_quant_fp8_i8(qx, x_f32, scale_seg)
        entry = "dynamic_per_tensor_quant_fp8_i8"
        dispatch = "n/a"
    else:
        scale_seg = torch.empty(rows, dtype=torch.float32, device=device)
        dynamic_per_token_quant_fp8_i8(qx, x_f32, scale_seg)
        entry = "dynamic_per_token_quant_fp8_i8"
        # 用 x.device（与 aiter 内部 _per_token_quant_num_cus(x_in.device) 同一
        # 设备对象，保证命中同一个 lru_cache 条目）
        dispatch = _per_token_path(rows, cols, x.device)

    torch.cuda.synchronize()

    # 单张量输出协议：[码字段 float32 | scale 段 float32]（与 reference 的 cat 同序）
    codes = qx.reshape(-1).to(torch.float32)
    out = torch.cat([codes, scale_seg.reshape(-1).to(torch.float32)])

    ctx = {
        "entry": f"aiter.ops.triton.quant.{entry}",
        "mode": mode,
        "quant_dtype": quant_dtype,
        "rows": rows,
        "cols": cols,
        "x_dtype": str(x.dtype),
        "scale_count": int(scale_seg.numel()),
        "out_len": int(out.numel()),
        "per_token_dispatch": dispatch,
    }
    return out, ctx
