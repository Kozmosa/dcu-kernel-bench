# aiter_impl.py — 3003_moe_op 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
# 适配器契约（按名取参，构造参数稀疏给出）：run(inputs, init_kwargs: dict, device)。
#
# 来源（本地 pinned 检出 .dcu_runs/aiter_pinned/，commit c39fff8c）：
#   aiter/ops/triton/moe_op.py
#     sha256 47431706f05b3df699e4b6b0d9500aaaaf111dd169a81f97fc316398ec737a63
#   op_tests/triton_tests/test_moe.py（语义唯一权威，sha256 3be33f91ad3d...f4b2）
#
# 公开 host 入口（moe_op.py:2561-2592；官方测试 test_moe.py:9-11 以 `triton_moe`
# 之名导入，test_moe.py:987-1007 为调用样板）：
#
#   aiter.ops.triton.moe_op.fused_moe(
#       A, B, C, A_scale, B_scale, B_zp, topk_weights, topk_ids,
#       sorted_token_ids, sorted_weights, expert_ids, num_tokens_post_padded,
#       mul_routed_weight, top_k, compute_type,
#       use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
#       use_int4_w4a16=False, use_int4_w4a8=False, use_mxfp4_w4a4=False,
#       per_channel_quant=False, block_shape=None, c_sorted=False,
#       bottom_a_use_mls_load=False, ck_sorting=False, ck_topk=8,
#       scale_bias_with_routed_weight=False, B_bias=None,
#       config=None) -> None            # void：结果就地写进 C
#
# 题面变体 → 入口参数的映射（对应准入记录的「rx 变体子集」）：
#   非量化 fp16/bf16 权重与激活 → A_scale=B_scale=B_zp=None 且 use_* 全 False
#     （moe_op.py:2607-2609 断言 A_scale/B_scale 必须为 None）；
#   含路由加权 → mul_routed_weight=True（MUL_ROUTED_WEIGHT 分支，moe_op.py:1603-1613）；
#   无 silu/gelu 融合、无 splitk/persistent 调度 → 走 fused_moe_kernel
#     （moe_op.py:2960-3019；mul_routed_weight=True 时 SPLIT_K 被强制为 0，
#      moe_op.py:2818-2820，故与 SPLITK_SIZE 环境变量无关）；
#   块对齐布局由题面直接给出（sorted_token_ids/expert_ids/
#     num_tokens_post_padded，block_m=64），与官方
#     torch_moe_align_block_size_ref/_moe_align_block_size（test_moe.py:325-436）
#     在 block_size=64 下逐元素一致，故不需要在适配器里做对齐。
#
# 布局：题面布局与 aiter 完全一致，无需任何 permute/transpose。
#   kernel 用「槽位」p = m*top_k + j 索引各处：
#     - C 行：c_ptrs = c_ptr + stride_cm * offs_token（moe_op.py:1650-1658），
#       offs_token 即 sorted_token_ids 里的 p；C 连续时 stride_cm = N，
#       等价于把 (M, top_k, N) 视作 (M*top_k, N) → 与 reference.py:95 的
#       flat_out 视角一致；
#     - A 行：offs_token // top_k（moe_op.py:1402）→ m；
#     - 路由权重：topk_weights_ptr + offs_token（moe_op.py:1610）→
#       topk_weights.reshape(-1)[p]。
#   因此 A (M,K) / B (E,N,K) / topk_weights (M,top_k) 原样传入，
#   (E,N,K) 与 kernel 的 stride_be/stride_bk=B.stride(2)/stride_bn=B.stride(1)
#   （moe_op.py:2977-2981）匹配，无需换轴。
#
# 输出：入口是 void，但写出的缓冲区形状与 reference 完全相同（单个张量
# (M, top_k, N)，dtype 同 A）。适配器按官方测试的写法自己分配
# C = torch.zeros((M, top_k, N))（test_moe.py:698 就是这么分配的；kernel 注释
# 也要求 C 是零初始化缓冲，moe_op.py:1375-1377），所以与题面单张量契约自然对齐，
# 不需要打包。
#
# 显式传 config（本项目里是必须的，不是性能偏好）：
#   BLOCK_SIZE_M = block_m（= 64）, BLOCK_SIZE_N = 64, BLOCK_SIZE_K = 32,
#   GROUP_SIZE_M = 8, COMBINE_SCALE_LOAD = False, USE_MLS_LOAD = False
#   1) config=None 在 c39fff8c 上对 mul_routed_weight=True 会直接崩：
#      fused_moe 用 is_bottom=mul_routed_weight 调
#      get_optimal_moe_config_func（moe_op.py:2615-2625），而
#      try_get_optimal_moe_config 在 is_bottom=True 时**返回 (config,
#      max_block_m) 二元组**（moe_config_utils.py:255-260 与 :268-270），
#      紧接着的 config["USE_MLS_LOAD"] = False（moe_op.py:2627-2628）会
#      TypeError: 'tuple' object does not support item assignment。
#   2) 官方 config JSON 以 (E, N=out_features, device_name, is_bottom) 为键
#      （moe_config_utils.py:15-29，目录 aiter/ops/triton/configs/moe/）。
#      题面 shape 族（E=16/8/64，N=256/512/1024）没有对应文件；命中不到时官方
#      兜底到 get_default_config（moe_config_utils.py:57-117）并只打一条
#      warning，所以缺文件**不会失败**，但显式 config 让基线可复现、不依赖
#      AITER_TRITON_CONFIGS_PATH（本机/真机都不去读那个目录）。
#   3) BLOCK_SIZE_M 必须整除布局的块对齐粒度 block_m（task.yaml:55 的
#      invariant）：一个 program 只读 expert_ids[pid_m]（moe_op.py:1372），
#      因此它负责的 BLOCK_SIZE_M 行必须落在同一个专家块内。取
#      BLOCK_SIZE_M == block_m 还额外保证 fused_moe 的小 batch 优化
#      （moe_op.py:2634-2640，EM = min(len(sorted), num_tokens*top_k*BLOCK_SIZE_M)）
#      覆盖全部有效块：npad <= num_valid*block_m 且 npad <= len(sorted_token_ids)，
#      所以 cdiv(EM, BLOCK_SIZE_M) >= cdiv(npad, BLOCK_SIZE_M) = num_pid_m。
#      （若取 block_m 的真因子 16/32，该优化在 num_tokens 很小时会少算尾部块，
#        例如 M=4/E=32/top_k=2 时 npad=512 而 EM 只覆盖 128 行 → 会静默算错；
#        BLOCK_SIZE_M=64 没有这个缺口。）
#   BLOCK_SIZE_N/BLOCK_SIZE_K 沿用 get_default_config 非量化分支的值
#   (moe_config_utils.py:109-116)；num_warps/num_stages 不显式给，走 Triton
#   默认值，与该兜底 config 的行为一致（它也不带这两项）。

import torch
import triton.language as tl


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方 MoE grouped GEMM（非量化、含路由加权）。

    inputs      : [A, B, topk_weights, topk_ids, sorted_token_ids, expert_ids,
                   num_tokens_post_padded]，顺序与 reference.py::make_inputs 一致
                   A (M, K) / B (E, N, K) / topk_weights (M, top_k) f32 /
                   topk_ids (M, top_k) i32 / sorted_token_ids (EN,) i32 /
                   expert_ids (M*top_k + E,) i32 / num_tokens_post_padded (1,) i32
    init_kwargs : {"num_experts": int, "top_k": int}（Model.__init__ 的参数）

    返回 (out, ctx)；out 为单张量 (M, top_k, N)，dtype 同 A。
    """
    # aiter 顶层 import 很重，按约定在函数内做最小导入
    from aiter.ops.triton.moe_op import fused_moe

    (A, B, topk_weights, topk_ids, sorted_token_ids, expert_ids,
     num_tokens_post_padded) = inputs

    # ---- init_kwargs → 入口参数（越界直接 raise，绝不静默用错参数）----
    if "num_experts" not in init_kwargs or "top_k" not in init_kwargs:
        raise ValueError(
            f"init_kwargs 缺 num_experts/top_k：{sorted(init_kwargs)}")
    num_experts = int(init_kwargs["num_experts"])
    top_k = int(init_kwargs["top_k"])

    M, K_in = A.shape
    E, N_out, K_b = B.shape
    if E != num_experts:
        raise ValueError(f"B.shape[0]={E} 与 init_kwargs['num_experts']={num_experts} 不一致")
    if K_b != K_in:
        raise ValueError(f"B.shape[2]={K_b} 与 A.shape[1]={K_in} 不一致")
    if top_k <= 0 or top_k > E:
        raise ValueError(f"top_k={top_k} 越界（要求 0 < top_k <= num_experts={E}）")
    if topk_ids.shape != (M, top_k) or topk_weights.shape != (M, top_k):
        raise ValueError(
            f"topk_ids{tuple(topk_ids.shape)}/topk_weights{tuple(topk_weights.shape)} "
            f"与 (M, top_k)=({M}, {top_k}) 不一致")
    if A.dtype != B.dtype:
        raise ValueError(f"A.dtype={A.dtype} 与 B.dtype={B.dtype} 不一致")
    if A.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"本题只覆盖非量化 fp16/bf16，收到 A.dtype={A.dtype}")

    num_valid = M * top_k  # 哨兵值 = topk_ids.numel()，kernel 的 num_valid_tokens

    # ---- 从布局反推块对齐粒度 block_m，并校验与 kernel 的假设自洽 ----
    span = int(sorted_token_ids.numel()) - num_valid
    if span < 0 or E == 0 or span % E != 0:
        raise ValueError(
            f"sorted_token_ids.numel()={sorted_token_ids.numel()} 与 "
            f"M*top_k={num_valid}、num_experts={E} 不构成合法块对齐布局")
    block_m = span // E + 1
    if block_m < 16 or (block_m & (block_m - 1)) != 0:
        raise ValueError(f"反推出的 block_m={block_m} 不是 >=16 的 2 次幂，无法作为 BLOCK_SIZE_M")

    # ---- dtype/layout 归一：只做无损 cast 与 contiguous（真机评测器已保证
    #      输入就是 case 生成的 dtype，这里是防御性处理，不改变数值）----
    A = A.contiguous()
    B = B.contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()
    sorted_token_ids = sorted_token_ids.to(torch.int32).contiguous()
    expert_ids = expert_ids.to(torch.int32).contiguous()
    num_tokens_post_padded = num_tokens_post_padded.to(torch.int32).contiguous()

    compute_type = tl.float16 if A.dtype == torch.float16 else tl.bfloat16

    # ---- 输出缓冲：与官方测试一致用 zeros（test_moe.py:698）----
    C = torch.zeros((M, top_k, N_out), dtype=A.dtype, device=A.device)

    # ---- 显式 config：BLOCK_SIZE_M == block_m（理由见文件头注释）----
    config = {
        "BLOCK_SIZE_M": int(block_m),
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "COMBINE_SCALE_LOAD": False,
        "USE_MLS_LOAD": False,
    }

    fused_moe(
        A,
        B,
        C,
        None,                    # A_scale：非量化必须为 None
        None,                    # B_scale：非量化必须为 None
        None,                    # B_zp
        topk_weights,
        topk_ids,
        sorted_token_ids,
        None,                    # sorted_weights：官方测试同传 None（test_moe.py:985）
        expert_ids,
        num_tokens_post_padded,
        True,                    # mul_routed_weight：题面含路由加权
        top_k,
        compute_type,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        use_int4_w4a8=False,
        use_mxfp4_w4a4=False,
        per_channel_quant=False,
        block_shape=None,
        c_sorted=False,
        bottom_a_use_mls_load=False,
        ck_sorting=False,        # sorted_token_ids 是题面原生布局，非 ck 打包
        ck_topk=top_k,
        scale_bias_with_routed_weight=False,
        B_bias=None,
        config=config,
    )
    torch.cuda.synchronize()

    ctx = {
        "path": "aiter.ops.triton.moe_op.fused_moe → fused_moe_kernel",
        "mul_routed_weight": True,
        "quant": "none (fp16/bf16)",
        "compute_type": str(compute_type),
        "config": dict(config),
        "shape": {
            "M": M, "K": K_in, "E": E, "N": N_out, "top_k": top_k,
            "block_m": block_m, "num_valid_tokens": num_valid,
            "EM": int(sorted_token_ids.numel()),
            "num_tokens_post_padded_len": int(num_tokens_post_padded.numel()),
        },
        "out_shape": tuple(C.shape),
        "out_dtype": str(C.dtype),
    }
    return C, ctx
