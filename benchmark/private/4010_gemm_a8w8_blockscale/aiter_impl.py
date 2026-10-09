# aiter_impl.py — 4010_gemm_a8w8_blockscale 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
# 适配器契约：run(inputs, init_kwargs: dict, device) —— 按名取参（构造参数可能
# 稀疏给出，位置式取值会错位，见 audit_model_class.py 的 case_init_kwargs）。
#
# ── 来源（aiter pinned 检出 .dcu_runs/aiter_pinned，commit c39fff8c）──────────
#   文件 : aiter/ops/triton/gemm_a8w8_blockscale.py
#   sha256: 4de9781184866cb738753fbadc9b3fe994bce61972f3319e1023b35647f446e9
#           （与 benchmark/sources/4010_gemm_a8w8_blockscale.yaml 的 device_kernel
#            证据逐字节一致，本机 Get-FileHash 复核通过）
#   入口 : gemm_a8w8_blockscale（:205-279），device kernel
#          _gemm_a8w8_blockscale_kernel（:22-185，模块私有，勿直接调用）；
#          EVEN_K / GRID_MN 由 @triton.heuristics 推出（:14-20）
#   官方 oracle : op_tests/triton_tests/test_gemm_a8w8_blockscale.py
#          sha256 995646361d29983fb49f01a2cc3d116d503b8553a30a21a9dbe687e81bd8f8d3
#          （run_torch :15-31 为 reference 语义的唯一权威，sources yaml 同款）
#
#   入口签名（逐字，源码 :205-213）：
#     gemm_a8w8_blockscale(
#         x: torch.Tensor,                          # (M, K) fp8e4m3 码字，行主
#         w: torch.Tensor,                          # (N, K) fp8e4m3 码字，行主
#         x_scale: torch.Tensor,                    # (M, ceil(K/GROUP_K)) float32
#         w_scale: torch.Tensor,                    # (ceil(N/GROUP_N), ceil(K/GROUP_K)) float32
#         dtype: Optional[float] = torch.bfloat16,  # 输出 dtype
#         y: Optional[torch.Tensor] = None,         # None 时宿主 torch.empty 并返回
#         config: Optional[dict] = None,            # tile 配置；None 时读 autotune JSON
#     ) -> y  # (M, N)、dtype，**单张量**
#
# ── 布局对齐（本题无需任何 permute / reshape / 打包）─────────────────────────
#   aiter 入口要的正是题目 io 声明的 TN 布局，与 reference 完全一致：
#     x (M, K) 行主  —— 直接传；宿主 :266-267 取 x.stride(0)/stride(1) 作
#                       stride_am/stride_ak。
#     w (N, K) 行主  —— 直接传；宿主自己在 :237 做 `w = w.T`，把转置后的
#                       stride(0)=1 / stride(1)=K 当 stride_bk / stride_bn 传给
#                       kernel（:268-269），kernel 侧按 (K, N) 视图索引 b_ptr
#                       （:138）。故调用方**不要**预先转置，也不要传 (K, N)。
#     x_scale (M, ceil(K/gk)) / w_scale (ceil(N/gn), ceil(K/gk)) —— 直接传；
#                       宿主 :238 转置 w_scale，再用转置后的形状反推 GROUP_K /
#                       GROUP_N（:248-249，见下节）。题面形状与 kernel 的
#                       per-dim 索引一致：a 侧列号 = 分块起点 // GROUP_K
#                       （:142-143 与 :172-176），b 侧列号 = offs_bn // GROUP_N
#                       （:144-145，逐元素），与 reference.py:93-96 的
#                       `k // gk` / `n // gn` 同粒度。
#   → reference.forward 返回单个 [M, N] 张量（reference.py:102），宿主 :240-241
#     的返回值也是单张量 y（(M, N)、行主、dtype=dtype），无 lse/多输出需要打包，
#     故本适配器直接返回 out，不做任何 cat/stack。
#
#   ⚠️ dtype 形参必须显式传 out_dtype：入口默认 dtype=torch.bfloat16（:210），
#   perf case `perf_m4096_n8192_k1024_g128_fp16`（out_dtype="float16"）若沿用
#   默认值，宿主会 empty 出 bf16 输出（:240-241）→ 形状对、dtype 对不上，
#   record_baseline 直接判 FAIL。
#
#   ⚠️ 码字恢复：reference.forward :77-78 先把 x/w `.to(torch.float8_e4m3fn)`
#   （评测器统一 cast 成 fp32 传入时，fp8 有限值在 fp32 精确表示，该 cast 无损；
#   离线路径直接给 fp8 张量时为 no-op）。本适配器照抄同一映射，不走 aiter 的
#   str_to_torch_dtype / get_fp8_dtypes（那两张表与题面 reference 无契约关系）。
#
# ── 数值等价（与 reference.py:91-102 逐项对应）──────────────────────────────
#   kernel :147 累加器在输出非 int8 时恒为 fp32；:164-166 每轮
#   `accumulator += tl.dot(a, b, input_precision="ieee") * a_scale[:, None]
#    * b_scale[None, :]`；:178 `c = accumulator.to(c_ptr.element_ty)`，而 c 就是
#   宿主按 dtype 分配的 y（:240-241）。同一 K 分块内 a_scale/b_scale 对每个 (m,n)
#   都是常量，故「先乘 scale 再累加」与 reference 的「逐元素反量化后 fp32 GEMM」
#   数学上相等（仅求和次序不同；fp8 码字乘积在 fp32 中精确）。末段 cast 到 y 的
#   dtype 即 reference 的 `.to(self.out_dtype)`。task.yaml 的 atol=rtol=2e-2 覆盖
#   该差异（官方测试 atol=0.01 / rtol=1e-2 亦通过，:154）。
#
# ── GROUP_K/GROUP_N 反推值与题面 group_k/group_n 的一致性（显式校验）────────
#   入口 :248-249 用 `next_power_of_2(cdiv(K, w_scale.T.shape[0]))` 反推 GROUP_K
#   （GROUP_N 用转置后第 1 维同理）。题面 group_k 为 2 的幂时
#   cdiv(K, ceil(K/gk)) ∈ (gk/2, gk] → next_power_of_2 恒等于 gk；但 K < gk 之类
#   边界下反推值可能小于 gk（如 K=1、gk=128 → GROUP_K=1），此时数值**仍然正确**
#   （只有一个块）。故本适配器不比对两个 GROUP 值本身，而按源码 :142-145 与
#   :172-176 的索引规则精确判定「块内元素所属 scale 列号」是否逐一相符
#   （_k_scale_mapping_ok / _n_scale_mapping_ok），不符就 raise —— 绝不静默采
#   一份与 reference 语义不同的基线。
#   本仓库既有先例：3009_routing 对启发式表未覆盖的 shape 同样直接 raise，
#   而不是换一份 config 硬凑。实际影响为零——perf_cases.json 的三个 case 都是
#   group_k = group_n = 128、K % 128 == 0，BLOCK_SIZE_K=128 恒满足该条件
#   （本机以假 torch / 假 aiter 桩件做契约烟测逐 case 通过；**非真机运行**，
#    本机无 GPU、无 aiter）；只有 hidden 里 group_k=64（小于官方分块 128）那类
#   shape 会被拒绝——它不是 perf case，不参与基线采集。
#
# ── autotune config 依赖（needs_autotune_config = True）─────────────────────
#   宿主在 config=None 时调 _get_config(M, N, K)（:243-244 调用点，:188-202 定义），
#   读 `{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM_BLOCKSCALE-A8W8.json` 并取
#   "any" 键（:202；`open` 在 :198 无 try/except，**无缺文件兜底**）。dev 来自
#   arch_info.get_device()，arch_info.py:5-10 的 _ARCH_TO_DEVICE 把 gfx936 映射为
#   **"BW200"**（gfx938 → "BW200B"）—— 即 DCU 真机会找
#   `gemm/BW200-GEMM_BLOCKSCALE-A8W8.json`。pinned 检出（commit c39fff8c）的
#   aiter/ops/triton/configs/gemm/ 下**只有 MI300X- / MI350X-GEMM_BLOCKSCALE-A8W8.json**
#   （`git ls-files "*BLOCKSCALE*"` 与目录列举双向核对，无 BW200- 前缀）→
#   config=None 在 DCU 上必然 FileNotFoundError。**需要的文件**：
#     {AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM_BLOCKSCALE-A8W8.json（dev=BW200/BW200B），
#     含 "any" 键，字段 = BLOCK_SIZE_M/N/K、GROUP_SIZE_M、num_warps、num_stages、
#     waves_per_eu、matrix_instr_nonkdim、kpack。
#   注意本题 _get_config 与 4006_gemm_a16w16 不同：它**不做 per-shape 覆盖、不分桶**，
#   恒返回 "any"（:202），因此回退表只有一键、无选档逻辑。
#
#   因此本适配器与 4001 / 4004 / 4006 / 4009 / 1014 / 1017 / 1018 同款处理：
#     ① 设备同名 JSON 在 → config=None，完全走 aiter 自己的官方 tuned 配置通道
#        （_get_config 是唯一权威来源；只为语义校验读一下同文件的 BLOCK_SIZE_K）；
#     ② 不在 → 显式传 config（宿主形参 config 本就是给外部供 config 用的，官方
#        API 与官方测试都这么用）。回退值逐字段抄自同 commit 的
#        configs/gemm/MI300X-GEMM_BLOCKSCALE-A8W8.json 的 "any" 键（MI350X 同值，
#        仅 kpack 2 vs 1 不同）。
#   config 只决定分块与占用（性能），不改变数值语义（GROUP_K/GROUP_N 由宿主自行
#   覆盖）；ctx["config_source"] 如实标注来源，绝不静默换参。

import json
import os

import torch

# 官方 config JSON 缺失时的显式回退：逐字段抄自
# aiter/ops/triton/configs/gemm/MI300X-GEMM_BLOCKSCALE-A8W8.json 的 "any" 键
# （_get_config:202 恒取 "any"，无 per-shape 覆盖、无分桶）。
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

# 题目 io 声明只允许这两种输出 dtype（task.yaml io.outputs：bfloat16 / float16；
# Model.__init__ 的 out_dtype 就是 getattr(torch, out_dtype)，reference.py:72）。
_OUT_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _next_power_of_2(n: int) -> int:
    """等价于宿主 :248-249 用的 triton.next_power_of_2（n >= 1）。"""
    return 1 << (n - 1).bit_length()


def _k_scale_mapping_ok(K, block_size_k, group_kernel, group_true):
    """kernel 的 K 侧 scale 列号是否与题面的逐元素归属一致。

    kernel 每个 K 分块只加载一次 a_scale（:160），列号由 :142-143 的初始值加
    :172-176 的增量推进给出；该望远镜求和恰为 floor(k0 / GROUP_K)（k0 = 块起点）。
    题面要求块内每个元素 k 用第 k // group_true 列（块内必须同属一个题面块）。
    """
    for k0 in range(0, K, block_size_k):
        k_hi = min(k0 + block_size_k, K) - 1
        idx_kernel = k0 // group_kernel
        if idx_kernel != k0 // group_true or idx_kernel != k_hi // group_true:
            return False
    return True


def _n_scale_mapping_ok(N, group_n_kernel, group_n_true):
    """kernel 的 N 侧 scale 列号是逐元素算的（:144-145），逐 n 比对即可。"""
    return all(n // group_n_kernel == n // group_n_true for n in range(N))


def _resolve_config():
    """返回 (config, source, fpath, block_size_k)。

    config is None → 交回 aiter `_get_config()` 读官方 tuned JSON（设备同名文件存在）；
    否则为显式 config dict（见文件头 autotune 段：DCU 设备同名 JSON 不存在）。
    block_size_k 供 K 侧 scale 归属校验用，取自宿主实际会用的那份配置。
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
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM_BLOCKSCALE-A8W8.json"
        if os.path.exists(fpath):
            with open(fpath, "r") as fh:
                block_size_k = int(json.load(fh)["any"]["BLOCK_SIZE_K"])
            # 官方通道可用：连 "any" 键的选取都交给 aiter 自己（_get_config:202）
            return None, f"official_autotune_json:{fpath}", fpath, block_size_k
        source = f"fallback_config(no {os.path.basename(fpath)})"
    except Exception as exc:  # pragma: no cover - 仅探测，失败即回退
        source = f"fallback_config(probe {type(exc).__name__})"

    # _get_config:202 恒取 "any"（本题无 per-shape 覆盖、无 small/large 分桶）
    return (
        dict(_FALLBACK_ANY_CONFIG),
        f"{source}:any",
        fpath,
        int(_FALLBACK_ANY_CONFIG["BLOCK_SIZE_K"]),
    )


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方实现（aiter.ops.triton.gemm_a8w8_blockscale.gemm_a8w8_blockscale）。

    inputs     : [x, w, x_scale, w_scale]（顺序同 reference.make_inputs）
                 x        [M, K] fp8e4m3 码字（评测器可能已 cast 成 fp32，任一路径
                          都由本适配器 .to(float8_e4m3fn) 无损恢复，同 reference:77）
                 w        [N, K] fp8e4m3 码字，行主（转置 GEMM 语义）
                 x_scale  [M, ceil(K/group_k)] float32（M 逐行、K 按 group_k 分块）
                 w_scale  [ceil(N/group_n), ceil(K/group_k)] float32（N/K 二维分块）
    init_kwargs: {"group_k": int, "group_n": int, "out_dtype": str}
                 （Model.__init__(group_k=128, group_n=128, out_dtype="bfloat16")，
                 reference.py:64；兼容单参 [group_k, group_n, out_dtype] 序列形态，
                 reference.py:66-72）
    device     : 目标设备（张量已在 device 上，宿主从 x.device 分配输出；仅用于 ctx）

    返回 (out, ctx)；out 为 [M, N] 连续张量、dtype = init_kwargs["out_dtype"]，
    与 reference.forward 的输出同形同 dtype（reference 返回单个未打包张量，本题
    无需打包）。
    """
    from aiter.ops.triton.gemm_a8w8_blockscale import (  # noqa: PLC0415
        gemm_a8w8_blockscale,
    )

    if len(inputs) != 4:
        raise ValueError(
            f"4010 需要 [x, w, x_scale, w_scale] 四个输入（reference.make_inputs "
            f"的消费顺序），收到 {len(inputs)} 个"
        )
    x, w, x_scale, w_scale = inputs

    # ---- 构造参数（按名取参；越界一律 raise，绝不静默用错）-------------------
    head_size = init_kwargs.get("head_size")
    single_arg_form = (
        "group_k" not in init_kwargs
        and isinstance(head_size, (list, tuple))
        and len(head_size) == 3
    )
    if single_arg_form:
        # 终审 harness 的单参序列形态 [group_k, group_n, out_dtype]（reference.py:66-72）
        group_k, group_n, out_dtype_name = head_size
    else:
        group_k = init_kwargs.get("group_k", 128)
        group_n = init_kwargs.get("group_n", 128)
        out_dtype_name = init_kwargs.get("out_dtype", "bfloat16")
    group_k = int(group_k)
    group_n = int(group_n)
    out_dtype_name = str(out_dtype_name)

    for name, val in (("group_k", group_k), ("group_n", group_n)):
        if val < 1 or val & (val - 1):
            raise ValueError(
                f"{name}={val} 必须是 2 的幂（task.yaml shape.invariants：1 ~ 1024）"
            )
    if out_dtype_name not in _OUT_DTYPES:
        raise ValueError(
            f"out_dtype={out_dtype_name!r} 不在题目允许的 {tuple(_OUT_DTYPES)} 内"
            "（task.yaml io.outputs / reference.py:72）"
        )
    out_dtype = _OUT_DTYPES[out_dtype_name]

    # ---- 形状校验（题目不变式；不满足则 raise，不猜）------------------------
    if x.dim() != 2 or w.dim() != 2:
        raise ValueError(
            f"x/w 必须是 2 维 (M, K) / (N, K)，实际 {tuple(x.shape)} / {tuple(w.shape)}"
        )
    M, K = x.shape
    N, Kw = w.shape
    if Kw != K:
        raise ValueError(f"K 维不一致：x {K} vs w {Kw}（宿主 :234 断言）")
    if min(M, N, K) < 1:
        raise ValueError(f"M/N/K 必须 >= 1，实际 {(M, N, K)}")
    scale_k = _ceil_div(K, group_k)
    scale_n = _ceil_div(N, group_n)
    if tuple(x_scale.shape) != (M, scale_k):
        raise ValueError(
            f"x_scale 形状应为 (M, ceil(K/group_k))=({M}, {scale_k})，"
            f"实际 {tuple(x_scale.shape)}（reference.py:86-88 同款断言）"
        )
    if tuple(w_scale.shape) != (scale_n, scale_k):
        raise ValueError(
            f"w_scale 形状应为 (ceil(N/group_n), ceil(K/group_k))="
            f"({scale_n}, {scale_k})，实际 {tuple(w_scale.shape)}"
            "（reference.py:88-89 同款断言）"
        )

    # ---- dtype / 布局规整（允许的 torch 用途：layout 转换与码字恢复）---------
    # reference.forward :77-78 同款：码字 cast 回 fp8e4m3、scale 归 fp32；
    # 题面 io 声明四者 contiguous，显式 .contiguous() 兜住非连续输入。
    x = x.to(torch.float8_e4m3fn).contiguous()
    w = w.to(torch.float8_e4m3fn).contiguous()
    x_scale = x_scale.to(torch.float32).contiguous()
    w_scale = w_scale.to(torch.float32).contiguous()

    # ---- autotune config ----------------------------------------------------
    config, config_source, config_fpath, block_size_k = _resolve_config()
    if block_size_k < 1:
        raise ValueError(f"config 的 BLOCK_SIZE_K={block_size_k} 非法（必须 >= 1）")

    # ---- 语义校验：kernel 的 scale 归属必须与题面逐元素一致（见文件头）------
    # 宿主 :248-249 从转置后的 w_scale 形状反推 GROUP_K / GROUP_N
    # （w_scale.T.shape = (scale_k, scale_n)，与本适配器已校验的形状一致）。
    group_k_kernel = _next_power_of_2(_ceil_div(K, scale_k))
    group_n_kernel = _next_power_of_2(_ceil_div(N, scale_n))
    if not _k_scale_mapping_ok(K, block_size_k, group_k_kernel, group_k):
        raise ValueError(
            f"aiter kernel 的 K 侧 scale 归属与题面不一致：GROUP_K={group_k_kernel}"
            f"（由 scale_k={scale_k} 反推，:248）、BLOCK_SIZE_K={block_size_k}、"
            f"题面 group_k={group_k}、K={K}。kernel 每个 K 分块只加载一次 a_scale"
            "（:160-166），要求 BLOCK_SIZE_K <= group_k 且分块不跨题面 scale 块；"
            "不满足时块内元素会取到错误的 scale。按 3009_routing 的先例不静默换 "
            "config，直接拒绝采集该 case 的基线"
        )
    if not _n_scale_mapping_ok(N, group_n_kernel, group_n):
        raise ValueError(
            f"aiter kernel 的 N 侧 scale 归属与题面不一致：GROUP_N={group_n_kernel}"
            f"（由 scale_n={scale_n} 反推，:249）、题面 group_n={group_n}、N={N}"
            "（kernel :144-145 逐元素取 offs_bn // GROUP_N）"
        )

    # ---- 调用官方 host 入口（核心计算全在 _gemm_a8w8_blockscale_kernel 内）---
    out = gemm_a8w8_blockscale(
        x,
        w,
        x_scale,
        w_scale,
        dtype=out_dtype,  # 必须显式传（入口默认 bf16，:210）
        y=None,           # None -> 宿主 torch.empty((M, N), out_dtype) 并返回（:240-241）
        config=config,
    )

    torch.cuda.synchronize()

    if tuple(out.shape) != (M, N) or out.dtype != out_dtype:
        raise RuntimeError(
            f"aiter 输出 {tuple(out.shape)}/{out.dtype} 与题目契约 "
            f"({M}, {N})/{out_dtype} 不一致"
        )

    return out, {
        "path": "aiter.ops.triton.gemm_a8w8_blockscale",
        "entry": "gemm_a8w8_blockscale",
        "layout": "TN（x (M,K) 行主 / w (N,K) 行主，宿主 :237 内部转置 w 及其 w_scale），无需 permute",
        "packed": False,
        "M": M,
        "N": N,
        "K": K,
        "group_k": group_k,
        "group_n": group_n,
        "scale_k": scale_k,
        "scale_n": scale_n,
        "group_k_kernel": group_k_kernel,
        "group_n_kernel": group_n_kernel,
        "block_size_k": block_size_k,
        "in_dtype": "float8_e4m3fn",
        "out_dtype": out_dtype_name,
        "out_shape": tuple(out.shape),
        "even_k": bool(K % block_size_k == 0),
        "config_source": config_source,
        "config_file": config_fpath,
        "config": config,
        "device": str(device),
    }
