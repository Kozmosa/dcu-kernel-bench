# aiter_impl.py — 4009_gemm_a8w8 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
# 适配器契约：run(inputs, init_kwargs: dict, device) —— 按名取参（构造参数可能
# 稀疏给出，位置式取值会错位，见 audit_model_class.py 的 case_init_kwargs）。
#
# ── 来源（aiter pinned 检出 .dcu_runs/aiter_pinned，commit c39fff8c）─────────
#   文件 : aiter/ops/triton/gemm_a8w8.py
#   sha256: 1ad4f10aad9ca8b8021f29e068080510f81a75bf5f0929ac73c20978d239beac
#           （与 benchmark/sources/4009_gemm_a8w8.yaml 记录逐字节一致，本机
#             Get-FileHash 复核通过）
#   入口 : gemm_a8w8（:193-259），device kernel _gemm_a8w8_kernel（:22-174），
#          EVEN_K / GRID_MN 由 @triton.heuristics 推出（:14-20）
#
#   入口签名（逐字，源码 :193-202）：
#     gemm_a8w8(
#         x: torch.Tensor,                        # (M, K) int8/fp8，行主
#         w: torch.Tensor,                        # (N, K) int8/fp8 同 dtype，行主
#         x_scale: torch.Tensor,                  # (M, 1) float32，per-token
#         w_scale: torch.Tensor,                  # (1, N) float32，per-channel
#         bias: Optional[torch.Tensor] = None,    # (1, N)，kernel 按它的 element_ty 加
#         dtype: Optional[float] = torch.bfloat16,# 输出 dtype；**默认是 bf16**
#         y: Optional[torch.Tensor] = None,       # 输出张量；None 时宿主自己 empty 并返回
#         config: Optional[dict] = None,          # tile 配置；None 时读 autotune JSON
#     ) -> y  # (M, N)，dtype，**单张量**
#
#   官方调用约定（op_tests/triton_tests/test_gemm_a8w8.py，sha256
#   7bef50e5a4195f1eb205ceea71c69fc5a656afd616cb4b855db22775e8285e6c，与 sources
#   yaml 一致）：
#     run_torch  :13-19   F.linear(x32, w32) → 外乘 matmul(x_scale, w_scale)
#                         → to(bias) 域加 bias → to(dtype)
#     run_triton :22-23   gemm_a8w8(x, weight, x_scale, w_scale, bias, dtype, y)
#     用例       :119-138 in_dtype ∈ {fp8e4m3, fp8e5m2, int8}、out_dtype=bf16、
#                         output∈{True,False}（y 预分配不影响返回值语义），
#                         assert_close(atol=0.01, rtol=1e-2)（:139）
#   kernel 融合顺序（:139-167）：fp32（c_ptr 非 int8 时）累加 ieee dot
#   → `accumulator *= a_scale[:, None] * b_scale[None, :]`（:159）
#   → `accumulator.to(bias_ptr.element_ty) + bias[None, :]`（:165）
#   → `.to(c_ptr.element_ty)`（:167）。因 bias 恒为 float32（题面 io 声明，官方生成器
#   :110 同款），:165 的 cast 是 no-op，与 reference.forward（reference.py:79-86：
#   fp32 GEMM → 外乘 scale → fp32 域加 bias → cast 输出 dtype）**逐项同序**，
#   admission note 已就此认定等价（sources/4009_gemm_a8w8.yaml:28-30）。
#
#   int8 路径的累加 dtype：kernel :139-140 `acc_dtype = tl.float32 if
#   c_ptr.type.element_ty != tl.int8 else tl.int32`，而本题输出必为 bf16/fp16，
#   故恒为 fp32 累加器（与 reference 的 fp32 matmul 一致）；int8 码字的乘积在
#   fp32 中精确，差异仅剩求和次序，远小于 task.yaml 的 2e-2 容差。
#
# ── 布局对齐（本题无需任何 permute/reshape）──────────────────────────────────
#   aiter 入口要的正是题目 io 声明的 TN 布局，与 reference 完全一致：
#     x (M, K) 行主 —— 直接传；宿主 :249-250 取 x.stride(0)/stride(1) 作
#                      stride_am/stride_ak。
#     w (N, K) 行主 —— 直接传；宿主自己在 :228 做 `w = w.T`，把转置后的
#                      stride(0)=1 / stride(1)=K 当 stride_bk / stride_bn 传给
#                      kernel（:251-252），kernel 侧按 (K, N) 视图索引 b_ptr
#                      （:131），因此**调用方不需要预先转置**，也**不要**传 (K, N)。
#     x_scale (M, 1) / w_scale (1, N) —— 直接传；kernel :134-137 按一维偏移加载
#                      （`pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) % M`
#                      与 N 侧同款写法：取模只作用于 arange，越界位置恒被掩码
#                      写回丢弃，是官方写法，不改写）。这要求两者**最后一维
#                      stride == 1 且元素数 == M / N**，题面 io 声明的 continuous
#                      形状严格满足；本适配器仍显式 .contiguous() 兜住。
#   → 输入输出形状/stride/dtype 与 reference 一一对应，无打包：reference.forward
#     返回单个 [M, N] 张量（reference.py:86），宿主 :231 的返回值也是单张量
#     （y），无 lse 之类需要拼接的额外输出，故本适配器直接返回 out。
#
#   ⚠️ dtype 形参必须显式传 out_dtype：入口默认 dtype=torch.bfloat16（:199），
#   hidden case `hidden_e4m3_nobias_fp16out`（out_dtype="float16"）下若沿用默认值，
#   宿主会按默认值 empty 出 bf16 输出（:231），形状对、dtype 对不上 →
#   record_baseline 直接判 FAIL。
#
#   ⚠️ bias 必须按 float32 传入（reference.py:85 也是 `bias.to(torch.float32)`）：
#   kernel :165 会先把累加器 cast 到 bias 的 dtype 再加，bias 若是 bf16/fp16 就
#   等价于「先舍入再加」，与 reference 的 fp32 域相加不同。本适配器显式
#   `.to(torch.float32)`，无论评测器/生成器给什么 bias dtype 都与 reference 对齐。
#   use_bias=False 时传 bias=None，走 kernel 的 HAS_BIAS=False 分支（不读 bias），
#   **不可**传占位 bias —— 那会把 bias 加进结果（reference.py:84-85 忽略之）。
#
# ── 量化码字恢复（in_dtype）────────────────────────────────────────────────
#   reference.forward :69-70 先把 x/w cast 回 `self.in_dtype`（评测器会统一 cast
#   成 fp32 传入，int8/fp8 码字在 fp32 中可精确表示，该 cast 无损），再做 GEMM。
#   本适配器**照抄同一映射表**（reference.py:58-60）：fp8e4m3→float8_e4m3fn、
#   fp8e5m2→float8_e5m2、int8→int8，不走 aiter 的 str_to_torch_dtype（那张表把
#   e4m3 指到 arch 相关的 get_fp8_dtypes()，与题面 reference 的映射无契约关系；
#   本题的 dtype 权威是 reference.py，必须与它逐位一致）。名字越界直接 raise。
#
# ── autotune config 依赖（needs_autotune_config = True）─────────────────────
#   宿主在 config=None 时调 _get_config(M, N, K)（:233-234 调用点，:177-190 定义），
#   读 `{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-A8W8.json` 并取 "any" 键
#   （:185-190，`open` 在 :186 无 try/except），dev 由 arch_info.get_device() 给出；
#   arch_info.py:5-10 的 _ARCH_TO_DEVICE 把 gfx936 映射为 **"BW200"**（gfx938 →
#   "BW200B"）—— 即 DCU 真机会找 `gemm/BW200-GEMM-A8W8.json`。pinned 检出
#   （commit c39fff8c）的 aiter/ops/triton/configs/gemm/ 下**只有
#   MI300X- / MI350X- 前缀的 GEMM-A8W8.json**，没有任何 BW200-/BW200B- 前缀
#   （已用 `git ls-files "aiter/ops/triton/configs/gemm/*.json"` 与目录列举双向核对，
#   返回为空）→ config=None 在 DCU 上必然 FileNotFoundError。**需要的文件**：
#     {AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-A8W8.json（dev=BW200/BW200B），
#     含 "any" 键，字段 = BLOCK_SIZE_M/N/K、GROUP_SIZE_M、num_warps、num_stages、
#     waves_per_eu、matrix_instr_nonkdim、kpack。
#   注意本题 _get_config 与 4006_gemm_a16w16 不同：它**不做 per-shape 覆盖、也
#   不分 small/any 桶**，恒返回 "any"（:190），因此回退表只有一键、无条件分支。
#
#   因此本适配器与 4001_batched_gemm_a8w8 / 4006_gemm_a16w16 / 1014 / 1017 / 1018
#   同款处理：
#     ① 设备同名 JSON 在 → config=None，完全走 aiter 自己的官方 tuned 配置
#        （_get_config 是唯一权威来源）；
#     ② 不在 → 显式传 config（宿主形参 config 本就是给外部供 config 用的，
#        官方 API 与官方测试都这么用）。回退值逐字段抄自同 commit 的
#        configs/gemm/MI300X-GEMM-A8W8.json 的 "any" 键 —— 本仓库既有适配器
#        （4001/4004/4006）统一以 MI300X 表兜底；MI350X 同文件仅
#        BLOCK_SIZE_M=64 / num_warps=8 / kpack=1 不同（MI300X=128/4/2），
#        两者都只是分块与占用，见下。
#   config 只决定分块与占用（性能），不改变数值语义；ctx["config_source"] 如实
#   标注来源，绝不静默换参。

import os

import torch

# 官方 config JSON 缺失时的显式回退：逐字段抄自
# aiter/ops/triton/configs/gemm/MI300X-GEMM-A8W8.json 的 "any" 键
# （_get_config:190 恒取 "any"，无 per-shape / 分桶逻辑）。
_FALLBACK_ANY_CONFIG = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 4,
    "num_warps": 4,
    "num_stages": 2,
    "waves_per_eu": 2,
    "matrix_instr_nonkdim": 16,
    "kpack": 2,
}

# 量化码字 dtype 映射：**逐字抄自题目 reference.py:58-60**（见文件头「量化码字恢复」）。
_IN_DTYPES = {
    "fp8e4m3": torch.float8_e4m3fn,
    "fp8e5m2": torch.float8_e5m2,
    "int8": torch.int8,
}

# 题目 io 声明只允许这两种输出 dtype（task.yaml io.outputs；reference.py:61 同款）
_OUT_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _resolve_config(M, N, K):
    """返回 (config, source, fpath)。

    config is None  → 交回 aiter `_get_config()` 读官方 tuned JSON；
    否则为显式 config dict（见文件头 autotune 段：DCU 设备同名 JSON 不存在）。
    aiter 一律函数内 import（顶层 import 很重，真机部署走最小导入垫片）。
    """
    source = "fallback_config(aiter probe failed)"
    fpath = None
    try:
        import aiter.ops.triton.utils.arch_info as arch_info  # noqa: PLC0415
        from aiter.ops.triton.utils.core import (  # noqa: PLC0415
            AITER_TRITON_CONFIGS_PATH,
        )

        dev = arch_info.get_device()  # gfx936 -> "BW200"、gfx938 -> "BW200B"
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-A8W8.json"
        if os.path.exists(fpath):
            # 官方通道可用：连 "any" 键的选取都交给 aiter 自己（_get_config:190）
            return None, f"official_autotune_json:{fpath}", fpath
        source = f"fallback_config(no {os.path.basename(fpath)})"
    except Exception as exc:  # pragma: no cover - 仅探测，失败即回退
        source = f"fallback_config(probe {type(exc).__name__})"

    # _get_config:190 恒取 "any"（本题无 per-shape 覆盖、无 small/large 分桶）
    return dict(_FALLBACK_ANY_CONFIG), f"{source}:any", fpath


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方实现（aiter.ops.triton.gemm_a8w8.gemm_a8w8）。

    inputs     : [x, w, x_scale, w_scale, bias]（顺序同 reference.make_inputs）
                 x        [M, K] in_dtype 码字（评测器可能已 cast 成 fp32，任一路径
                          都由本适配器 .to(in_dtype) 无损恢复）
                 w        [N, K] 与 x 同 dtype，行主（转置 GEMM 语义）
                 x_scale  [M, 1] float32（per-token）
                 w_scale  [1, N] float32（per-channel）
                 bias     [1, N] float32（use_bias=False 时为占位张量，入口不接收）
    init_kwargs: {"in_dtype": str, "out_dtype": str, "use_bias": bool}
                 （Model.__init__(in_dtype="fp8e4m3", out_dtype="bfloat16",
                 use_bias=True)，reference.py:56）
    device     : 目标设备（张量已在 device 上，仅用于 ctx 记录）

    返回 (out, ctx)；out 为 [M, N] 连续张量、dtype = init_kwargs["out_dtype"]，
    与 reference.forward 的输出同形同 dtype（reference 返回单个未打包张量，本题
    无需打包）。
    """
    from aiter.ops.triton.gemm_a8w8 import gemm_a8w8  # noqa: PLC0415

    x, w, x_scale, w_scale, bias = inputs

    # ---- 构造参数（按名取参；越界一律 raise，绝不静默用错）-------------------
    in_dtype_name = init_kwargs.get("in_dtype", "fp8e4m3")
    if in_dtype_name not in _IN_DTYPES:
        raise ValueError(
            f"in_dtype={in_dtype_name!r} 不在题目允许的 {tuple(_IN_DTYPES)} 内"
            "（reference.py:58-60 的量化 dtype 映射表为准）"
        )
    out_dtype_name = init_kwargs.get("out_dtype", "bfloat16")
    if out_dtype_name not in _OUT_DTYPES:
        raise ValueError(
            f"out_dtype={out_dtype_name!r} 不在题目允许的 {tuple(_OUT_DTYPES)} 内"
            "（task.yaml io.outputs / reference.py:61）"
        )
    use_bias = bool(init_kwargs.get("use_bias", True))
    in_dtype = _IN_DTYPES[in_dtype_name]
    out_dtype = _OUT_DTYPES[out_dtype_name]

    # ---- 形状校验（题目不变式；不满足则 raise，不猜）------------------------
    if x.dim() != 2 or w.dim() != 2:
        raise ValueError(
            f"x/w 必须是 2 维 (M, K) / (N, K)，实际 {tuple(x.shape)} / {tuple(w.shape)}"
        )
    M, K = x.shape
    N, Kw = w.shape
    if Kw != K:
        raise ValueError(f"K 维不一致：x {K} vs w {Kw}（宿主 :222 断言）")
    if min(M, N, K) < 1:
        raise ValueError(f"M/N/K 必须 >= 1，实际 {(M, N, K)}")
    if tuple(x_scale.shape) != (M, 1):
        raise ValueError(
            f"x_scale 必须是 (M, 1)={(M, 1)}，实际 {tuple(x_scale.shape)}"
            "（kernel :134-136 按一维 M 元素加载）"
        )
    if tuple(w_scale.shape) != (1, N):
        raise ValueError(
            f"w_scale 必须是 (1, N)={(1, N)}，实际 {tuple(w_scale.shape)}"
            "（kernel :135-137 按一维 N 元素加载）"
        )

    # ---- dtype / 布局规整（允许的 torch 用途：layout 转换与码字恢复）---------
    # reference.forward :69-72 同款：码字 cast 回 in_dtype、scale 归 fp32；
    # 题面 io 声明三者 contiguous，显式 .contiguous() 兜住非连续输入。
    x = x.to(in_dtype).contiguous()
    w = w.to(in_dtype).contiguous()
    x_scale = x_scale.to(torch.float32).contiguous()
    w_scale = w_scale.to(torch.float32).contiguous()

    if use_bias:
        if tuple(bias.shape) != (1, N):
            raise ValueError(f"bias 必须是 (1, N)={(1, N)}，实际 {tuple(bias.shape)}")
        # 必须 fp32：kernel :165 先按 bias 的 element_ty 舍入累加器再加（见文件头）
        bias_arg = bias.to(torch.float32).contiguous()
    else:
        # use_bias=False：reference :84-85 忽略 bias，故传 None 走 HAS_BIAS=False
        bias_arg = None

    # ---- autotune config ----------------------------------------------------
    config, config_source, config_fpath = _resolve_config(M, N, K)

    out = gemm_a8w8(
        x,
        w,
        x_scale,
        w_scale,
        bias=bias_arg,
        dtype=out_dtype,  # 必须显式传（入口默认 bf16，:199）
        y=None,           # None -> 宿主 torch.empty((M, N), out_dtype) 并返回（:231）
        config=config,
    )

    torch.cuda.synchronize()
    return out, {
        "path": "aiter.ops.triton.gemm_a8w8",
        "entry": "gemm_a8w8",
        "layout": "TN（x (M,K) 行主 / w (N,K) 行主，宿主内部转置 w），无需 permute",
        "packed": False,
        "M": M,
        "N": N,
        "K": K,
        "in_dtype": in_dtype_name,
        "out_dtype": out_dtype_name,
        "use_bias": use_bias,
        "out_shape": tuple(out.shape),
        # 只有显式 config 时才算得准；走官方 JSON 时 aiter 自己按 EVEN_K 启发式决定，
        # 此处留 None，不拿回退表的 BLOCK_SIZE_K 冒充官方值。
        "even_k": None if config is None else bool(K % config["BLOCK_SIZE_K"] == 0),
        "config_source": config_source,
        "config_file": config_fpath,
        "config": config,
    }
