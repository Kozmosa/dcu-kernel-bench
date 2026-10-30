# aiter_impl.py — 4006_gemm_a16w16 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
# 适配器契约：run(inputs, init_kwargs: dict, device) —— 按名取参（构造参数可能
# 稀疏给出，位置式取值会错位，见 audit_model_class.py 的 case_init_kwargs）。
#
# ── 来源（aiter pinned 检出 .dcu_runs/aiter_pinned，commit c39fff8c）─────────
#   文件 : aiter/ops/triton/gemm_a16w16.py
#   sha256: 003b1343a79ae408e8d30eb4e0168b7ae46ff8b86065661c196e6c6ccd809f4f
#           （与 benchmark/sources/4006_gemm_a16w16.yaml 记录逐字节一致，本机
#             Get-FileHash 复核通过）
#   入口 : gemm_a16w16（:142-191），device kernel _gemm_a16_w16_kernel（:22-108），
#          EVEN_K / GRID_MN 由 @triton.heuristics 推出（:15-21）
#
#   入口签名（逐字，源码 :142-148）：
#     gemm_a16w16(
#         x: torch.Tensor,                        # (M, K) fp16/bf16，行主
#         w: torch.Tensor,                        # (N, K) fp16/bf16，行主（线性层权重布局）
#         dtype: Optional[float] = torch.bfloat16,# 输出 dtype；**默认是 bf16**
#         y: Optional[torch.Tensor] = None,       # 输出张量；None 时宿主自己 empty 并返回
#         config: Optional[dict] = None,          # tile 配置；None 时读 autotune JSON
#     ) -> y  # (M, N)，dtype，**单张量**
#
#   官方调用约定（op_tests/triton_tests/test_gemm_a16w16.py:77-87，sha256
#   97c6d57874167ffe0784ff1f259277cf5f8d6c6e0d296a85b8ba4a85cdceb252）：
#     torch_out  = F.linear(x, w, bias=None)              # :80
#     triton_out = gemm_a16w16(x, w, out_dtype, y)        # :83 / :85
#     triton.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)  # :87
#   与题目 reference.forward（reference.py:57-66）逐项对齐：x [M,K] 与 weight [N,K]
#   行主 TN、无 bias、fp32 累加（kernel :95 `tl.dot(a, b, input_precision="ieee")`，
#   累加器 fp32，见 :77-78）、K 尾越界按 0 累加（:87-93 掩码 other=0.0）、
#   M/N 越界行/列取模回绕加载 + 掩码写回（:72-73、:104-108），输出只经历一次
#   dtype 收缩（:101），与 reference 的 `torch.mm(x32, w32.t()).to(x.dtype)` 同语义。
#
# ── 布局对齐（本题无需任何 permute/reshape）──────────────────────────────────
#   aiter 入口要的正是题目 io 声明的 TN 布局，与 reference 完全一致：
#     x (M, K) 行主 —— 直接传；kernel :74 用 x.stride(0)/stride(1) 取 a_ptr 偏移。
#     w (N, K) 行主 —— 直接传；宿主自己在 :164 做 `w = w.T`，把转置后的
#                      stride(0)=1 / stride(1)=K 当作 stride_bk / stride_bn 传给
#                      kernel（:184-185），kernel 侧按 (K, N) 视图索引 b_ptr
#                      （:75），因此**调用方不需要预先转置**，也**不要**传 (K, N)。
#   → 输入输出形状/stride/dtype 与 reference 一一对应，无打包（reference 返回单个
#     [M, N] 张量，宿主 :158 的返回值类型为单张量，无 lse 之类需要拼接的额外输出）。
#
#   ⚠️ dtype 形参必须显式传 x.dtype：入口默认 dtype=torch.bfloat16（:145），
#   fp16 case 下若沿用默认值，宿主会按默认值 empty 出 bf16 输出（:167），
#   形状对、dtype 对不上 → record_baseline 直接判 FAIL。本适配器显式传
#   dtype=x.dtype（reference.py:66 也是 `out.to(x.dtype)`）。
#
# ── autotune config 依赖（needs_autotune_config = True）─────────────────────
#   宿主在 config=None 时调 _get_config(M, N, K)（:170 调用点，:111-139 定义），
#   读 `{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-A16W16.json`（:120-122，
#   `open` 无 try/except），dev 由 arch_info.get_device() 给出；arch_info.py:5-10
#   的 _ARCH_TO_DEVICE 把 gfx936 映射为 **"BW200"**（gfx938 → "BW200B"）——即 DCU
#   真机会找 `gemm/BW200-GEMM-A16W16.json`。pinned 检出的
#   aiter/ops/triton/configs/gemm/ 下**只有 MI300X- 与 MI350X- 前缀的
#   GEMM-A16W16*.json，没有 BW200- 前缀**（已用目录列举核对），且仓库内
#   `BW200` 的 27 处命中里没有一处是 gemm_a16w16 的配置 → config=None 在 DCU 上
#   会 FileNotFoundError（:122 open 无兜底）。**需要的文件**：
#     {AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-A16W16.json（dev=BW200/BW200B），
#     含 "any" 键（M<128 时可选 "small" 键），字段 = BLOCK_SIZE_M/N/K、
#     GROUP_SIZE_M、num_warps、num_stages、waves_per_eu、matrix_instr_nonkdim、
#     cache_modifier、kpack。
#
#   因此本适配器与 4001_batched_gemm_a8w8 / 1014 / 1017 / 1018 同款处理：
#     ① 设备同名 JSON 在 → config=None，完全走 aiter 自己的官方 tuned 配置
#        （_get_config 是唯一权威来源，含 per-shape 覆盖与 small/any 分桶）；
#     ② 不在 → 按官方 _get_config:117-139 的逻辑就地复刻一遍（先看 per-shape
#        文件 `{dev}-GEMM-A16W16-N={N}-K={K}.json`，缺失则退到 default，再按
#        `M < 128 and "small" in cfg` 选桶），两份都缺时用显式回退 config：
#        字段值逐字抄自同 commit 的 configs/gemm/MI300X-GEMM-A16W16.json
#        （sha256 518b3bbb9ebeb991ccea7b5608b82c9f529b9471b6aacedf62c5fdfcf6c7a913；
#        MI350X 同文件仅 matrix_instr_nonkdim 为 32，其余逐字段相同）。
#   config 只决定分块与占用（性能），不改变数值语义；ctx["config_source"] 如实
#   标注来源，绝不静默换参。
#
# ── 构造参数（init_kwargs）──────────────────────────────────────────────────
#   Model.__init__(in_features: int, out_features: int)（reference.py:52-55）：
#   in_features 对应 K、out_features 对应 N，但 reference 明确声明它们是
#   **实例元数据**，"forward 一律以运行期张量的实际形状为准，不与 init 参数做强
#   一致检查"（reference.py:34-36）。而 case 只给 m/n/k，构造参数由
#   audit_model_class.case_init_kwargs 用 get_init_inputs()=[1024, 8192] 补齐
#   （perf case 的 k=8192/n=4096 与之并不相等），故**绝不可**用它们校验/覆盖
#   aiter 入口的 M/N/K，否则会把基线采集直接 raise 掉。本适配器只把它们记进
#   ctx 并如实标注是否与运行期形状一致（init_metadata_consistent）。

import json
import os

import torch

# 官方 config JSON 缺失时的显式回退：逐字段抄自
# aiter/ops/triton/configs/gemm/MI300X-GEMM-A16W16.json（该文件只有 "any" 一键，
# 故 M<128 时官方 _get_config 的 "small" 分支也不会命中，行为与官方一致）。
_FALLBACK_DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 256,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 4,
    "num_warps": 8,
    "num_stages": 2,
    "waves_per_eu": 2,
    "matrix_instr_nonkdim": 16,
    "cache_modifier": None,
    "kpack": 1,
}

# kernel 的 acc_dtype 由 c_ptr.element_ty 决定（:77），ieee dot 对 fp32 同样成立；
# 题目 io 声明 fp16/bf16 两种，reference 另声明「输入被评测器 cast 成 fp32 时走
# 同一路径」（reference.py:58-60），故 fp32 也放行。
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


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
        gemm_dir = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
        fpath = f"{gemm_dir}/{dev}-GEMM-A16W16.json"
        if os.path.exists(fpath):
            # 官方通道可用：连 per-shape 文件与 small/any 分桶都交给 aiter 自己
            return None, f"official_autotune_json:{fpath}", fpath

        # 官方默认文件缺失 —— 就地复刻 _get_config:117-139 的 per-shape 通道
        per_shape = f"{gemm_dir}/{dev}-GEMM-A16W16-N={N}-K={K}.json"
        if os.path.exists(per_shape):
            with open(per_shape, "r") as handle:
                cfg = json.load(handle)
            bucket = "small" if (M < 128 and "small" in cfg) else "any"
            return dict(cfg[bucket]), f"per_shape_autotune_json:{per_shape}:{bucket}", per_shape
        source = f"fallback_config(no {os.path.basename(fpath)})"
    except Exception as exc:  # pragma: no cover - 仅探测，失败即回退
        source = f"fallback_config(probe {type(exc).__name__})"

    # _FALLBACK_DEFAULT_CONFIG 已经是 MI300X-GEMM-A16W16.json 里 "any" 桶的内容
    # （扁平字典，见 :92-106），此处**不可**再按桶索引一次——那会 KeyError: 'any'。
    # 官方 _get_config:136-139 的分桶规则在无 "small" 键时也恒取 "any"，故直接返回。
    return dict(_FALLBACK_DEFAULT_CONFIG), f"{source}:any", fpath


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方实现（aiter.ops.triton.gemm_a16w16.gemm_a16w16）。

    inputs     : [x, weight]（顺序同 reference.make_inputs）
                 x      [M, K] fp16/bf16/fp32，行主连续
                 weight [N, K] 与 x 同 dtype，行主连续（线性层权重布局，TN）
    init_kwargs: {"in_features": int, "out_features": int}（仅实例元数据，不参与
                 计算，见文件头；照 reference.py:34-36 不做强一致检查）
    device     : 目标设备（张量已在 device 上，仅用于 ctx 记录）

    返回 (out, ctx)；out 为 [M, N] 连续张量、dtype = x.dtype，与 reference.forward
    的输出同形同 dtype（reference 返回单个未打包张量，本题无需打包）。
    """
    from aiter.ops.triton.gemm_a16w16 import gemm_a16w16  # noqa: PLC0415

    x, weight = inputs

    # ---- 形状 / dtype 校验（题目不变式；不满足则 raise，不猜）-----------------
    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError(f"x/weight 必须是 2 维 (M, K) / (N, K)，实际 {tuple(x.shape)} / {tuple(weight.shape)}")
    M, K = x.shape
    N, Kw = weight.shape
    if Kw != K:
        raise ValueError(f"K 维不一致：x {K} vs weight {Kw}（reference.py:63 断言）")
    if min(M, N, K) < 1:
        raise ValueError(f"M/N/K 必须 >= 1，实际 {(M, N, K)}")
    if x.dtype != weight.dtype:
        raise ValueError(f"x 与 weight dtype 必须一致，实际 {x.dtype} vs {weight.dtype}")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"dtype={x.dtype} 不在题目允许的 fp16/bf16（另放行 fp32）内")

    # ---- 布局规整（允许的 torch 用途：layout 转换）---------------------------
    # 题面 io 声明两者 contiguous；显式 .contiguous() 兜住非连续输入（kernel :49-54
    # 只要求各 stride > 0，但行主连续下 offs_k/offs_bn 的向量化访问才是最优）。
    x = x.contiguous()
    weight = weight.contiguous()

    # ---- autotune config ----------------------------------------------------
    config, config_source, config_fpath = _resolve_config(M, N, K)

    # dtype 必须显式传（入口默认 bf16，:145；fp16 case 下会用错输出 dtype）
    out = gemm_a16w16(
        x,
        weight,
        x.dtype,
        None,        # y=None -> 宿主 torch.empty((M, N), dtype) 分配（:166-167）
        config,
    )

    torch.cuda.synchronize()

    # init 参数只作元数据：reference 不与运行期形状做强一致检查（reference.py:34-36）
    in_features = init_kwargs.get("in_features")
    out_features = init_kwargs.get("out_features")
    return out, {
        "path": "aiter.ops.triton.gemm_a16w16",
        "entry": "gemm_a16w16",
        "layout": "TN（x (M,K) 行主 / weight (N,K) 行主，宿主内部转置 weight）",
        "M": M,
        "N": N,
        "K": K,
        "dtype": str(x.dtype),
        "in_features": in_features,
        "out_features": out_features,
        "init_metadata_consistent": (in_features == K and out_features == N),
        "out_shape": tuple(out.shape),
        "even_k": (K % config["BLOCK_SIZE_K"] == 0) if config else None,
        "config_source": config_source,
        "config_file": config_fpath,
        "config": config,
    }
