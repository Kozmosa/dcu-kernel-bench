# aiter_impl.py — 3007_moe_op_silu_fused 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用（契约 run(inputs, init_kwargs, device)，
# 按名取参），采集离线终审用的 aiter 基线。
#
# 来源：OpenDAS/aiter @ c39fff8c77df4e80617649e92fa3c2615f2c43d1
#   aiter/ops/triton/moe_op_silu_fused.py
#   sha256 d271cd67f6a7e4e4f2654a86b86b15c03e432bf1818df263e03688e645ccc3cd
#
# 公开算子入口（含类型标注的完整签名，行号 1004-1024）：
#   fused_moe_silu(A, B, C, A_scale, B_scale, B_zp, topk_weights, topk_ids,
#                  sorted_token_ids, expert_ids, num_tokens_post_padded,
#                  mul_routed_weight, top_k, compute_type, use_fp8_w8a8,
#                  use_int8_w8a16, use_int4_w4a16,
#                  block_shape: Optional[List[int]] = None,
#                  config: Optional[Dict[str, Any]] = None) -> None
# 返回值是 None：结果原地写进 C。调用约定取自官方测试
#   op_tests/triton_tests/test_moe.py::test_fused_moe（sha256
#   3be33f91ad3ddb402cc291e74725213d91f0e5db772c1bc20955e8748a85f4b2）中的
#   silu_fused 分支（第 985-1007 行的调用），即本题 3007 的语义唯一权威。
#
# 本题 use_int8_w8a16=use_int4_w4a16=False，因此文件内走 else 分支
# （moe_op_silu_fused.py:1158-1244）→ 非 persistent 的 _fused_moe_silu_kernel
# （moe_op_silu_fused.py:559-782；模块级 _USE_MOE_PERSISTENT_KERNEL 缺省 False，
# 第 20 行）。use_fp8_w8a8=True 且 block_shape=[block_n, block_k] 时由该 kernel 的
# fp8 分块分支（第 739-748 行：acc += dot(a,b) * a_scale[:,None] * b_scale[None,:]）
# 承担反量化，与 reference.py 的 A_scale/B_scale 逐 K 组、逐 N 组的反量化数学一致。
#
# 布局对齐（逐项核对过，无需任何 permute）：
#   * A (M, K) / B (E, 2N, K) / C (M*top_k, N)：kernel 用 offs_token = sorted_token_ids
#     作 C 行下标（m*top_k + j，与 reference 的 rows[:,0]*top_k + rows[:,1] 相同），
#     A 行取 offs_token // top_k = m（第 691-693 行）。
#   * C 的行数是 M*top_k（reference 输出 (M*top_k, N) 的原样展平），列数是
#     B.shape[1]//2（kernel 第 781 行 `offs_cn < N // 2`，其 N 即 B.shape[1]）。
#   * gate/up 的 chunked 布局：kernel 用 offs_bn = (pid_n*BLOCK_SIZE_N//2 + i//2)
#     + (i%2)*(N//2) 交错取 gate 列与 up 列（第 679-688 行），再
#     reshape(BM, BSH, 2).split() 得到 (gate, up)（第 771-775 行）——
#     与 reference 的 x[:, :N] / x[:, N:] 完全对应。
#   * 路由权重乘在激活之前、以高精度乘（第 757-759 行），与 reference 一致。
#   * 输出 dtype = compute_type = topk_weights.dtype（题面 outputs.dtype =
#     same_as_topk_weights）。
#
# block_size_m：题面 io.init_inputs 不含 block_size_m（它只喂 make_inputs），但
# kernel 的 BLOCK_SIZE_M 必须等于调度三件套的对齐粒度，否则 expert_ids[pid_m] 会
# 跨专家错位（expert_ids 只有 M*top_k + E 项，一 entry 一 block_size_m 块）。
# 这里按 task.yaml 声明的不变量 len(sorted_token_ids) = M*top_k + E*(block_size_m-1)
# 反解出来，越界直接 raise，绝不猜。
#
# config：fused_moe_silu 必须显式收 config（第 1053 行无条件下标访问
# config["BLOCK_SIZE_M"]），且 _fused_moe_silu_kernel 没有 @triton.autotune /
# 不读 AITER_TRITON_CONFIGS_PATH（只带 @triton.heuristics 求 EVEN_K），故本适配器
# 自带一份 4 键 config，不依赖任何外部调优 JSON（needs_autotune_config=false）。
# N/K/GROUP_SIZE_M 取 aiter 缺省族（dense: 64/32/8；fp8_block: block_n/block_k/8），
# BLOCK_SIZE_M 固定为反解出的 block_size_m（正确性所需）。

import torch
import triton.language as tl

# compute_type 映射（compute_type 是输出/累加舍入 dtype；fp8 模式下 A/B 是 e4m3
# 码字，而内核仍以模型 dtype 做 compute_type，见 test_moe.py:1001 的
# torch_to_triton_dtype[dtype]）。
_TORCH_TO_TL = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方 fused_moe_silu。

    inputs（= reference.py::make_inputs 的返回顺序，元素已在 device 上）:
        A                        (M, K)
        B                        (E, 2N, K)
        A_scale                  fp8_block: (M, K//block_k) float32；dense: (1,) 占位
        B_scale                  fp8_block: (E, 2N//block_n, K//block_k) float32；dense: (1,) 占位
        topk_weights             (M, top_k)
        topk_ids                 (M, top_k) int64
        sorted_token_ids         (M*top_k + E*(block_size_m-1),) int32
        expert_ids               (M*top_k + E,) int32
        num_tokens_post_padded   (1,) int32
    init_kwargs: {"mul_routed_weight": bool, "quant": "dense"|"fp8_block",
                  "block_n": int, "block_k": int}（后两个可缺省 = 128）

    返回 (out, ctx)；out = C，形状 (M*top_k, N)，dtype 与 topk_weights 一致。
    """
    from aiter.ops.triton.moe_op_silu_fused import fused_moe_silu

    (A, B, A_scale, B_scale, topk_weights, topk_ids,
     sorted_token_ids, expert_ids, num_tokens_post_padded) = inputs

    mul_routed_weight = bool(init_kwargs.get("mul_routed_weight", True))
    quant = str(init_kwargs.get("quant", "dense"))
    block_n = int(init_kwargs.get("block_n", 128))
    block_k = int(init_kwargs.get("block_k", 128))

    if quant not in ("dense", "fp8_block"):
        raise ValueError(
            f"init_kwargs['quant']={quant!r} 越界：题面只声明 dense / fp8_block"
        )
    if block_n < 1 or block_k < 1:
        raise ValueError(
            f"init_kwargs block_n/block_k 必须 >= 1，实际 {block_n}/{block_k}"
        )

    # ---------- shape ----------
    if A.dim() != 2 or B.dim() != 3:
        raise ValueError(f"A 应为 2 维、B 应为 3 维，实际 {tuple(A.shape)} / {tuple(B.shape)}")
    M, K = int(A.shape[0]), int(A.shape[1])
    E, N2, K_b = int(B.shape[0]), int(B.shape[1]), int(B.shape[2])
    if K_b != K:
        raise ValueError(f"B 的末维 {K_b} 与 A 的 K={K} 不一致")
    if N2 % 2 != 0:
        raise ValueError(f"B 的输出维 2N={N2} 必须为偶数（chunked gate|up 布局）")
    N = N2 // 2
    top_k = int(topk_ids.shape[1])
    if int(topk_ids.shape[0]) != M:
        raise ValueError(f"topk_ids 的 M={int(topk_ids.shape[0])} 与 A 的 M={M} 不一致")
    if top_k < 1:
        raise ValueError(f"top_k={top_k} 必须 >= 1")
    if int(num_tokens_post_padded.numel()) != 1:
        raise ValueError(
            f"num_tokens_post_padded 应为 (1,)，实际 {tuple(num_tokens_post_padded.shape)}"
        )

    # ---------- block_size_m：由调度三件套的长度反解（题面 io 不变量） ----------
    len_sorted = int(sorted_token_ids.numel())
    len_expert_ids = int(expert_ids.numel())
    if len_expert_ids != M * top_k + E:
        raise ValueError(
            f"len(expert_ids)={len_expert_ids} 与题面不变量 M*top_k + E="
            f"{M * top_k + E} 不符，调度三件套不是本题约定的布局"
        )
    extra = len_sorted - M * top_k
    if extra < 0 or extra % E != 0:
        raise ValueError(
            f"len(sorted_token_ids)={len_sorted} 与题面不变量 M*top_k + E*(block_size_m-1) "
            f"（M*top_k={M * top_k}, E={E}）不符，无法反解 block_size_m"
        )
    block_size_m = extra // E + 1
    if block_size_m < 16:
        raise RuntimeError(
            f"反解出的 block_size_m={block_size_m}：kernel 的 BLOCK_SIZE_M 必须等于该对齐"
            "粒度（否则 expert_ids[pid_m] 跨专家错位），而 tl.dot 要求 M >= 16，"
            "aiter 的 _fused_moe_silu_kernel 无法承载此 case"
        )

    # ---------- dtype ----------
    out_dtype = topk_weights.dtype
    if out_dtype not in _TORCH_TO_TL:
        raise RuntimeError(
            f"topk_weights.dtype={out_dtype} 越界：本适配器期望 make_inputs 产出的原生"
            " fp16/bfloat16（记录基线用的 record_baseline.py 不做 fp32 cast）"
        )
    compute_type = _TORCH_TO_TL[out_dtype]
    use_fp8_w8a8 = quant == "fp8_block"

    if use_fp8_w8a8:
        if A.dtype != torch.float8_e4m3fn or B.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                f"quant='fp8_block' 要求 A/B 为 float8_e4m3fn 码字，实际 "
                f"{A.dtype} / {B.dtype}"
            )
        if K % block_k != 0:
            raise ValueError(f"fp8_block 要求 K % block_k == 0，实际 {K} % {block_k} != 0")
        if N2 % block_n != 0:
            raise ValueError(f"fp8_block 要求 2N % block_n == 0，实际 {N2} % {block_n} != 0")
        if block_n < 16 or block_k < 16:
            raise RuntimeError(
                f"block_n={block_n} / block_k={block_k}：fp8 分块模式下 BLOCK_SIZE_N/K 取"
                " block_n/block_k，tl.dot 要求 >= 16"
            )
        if tuple(A_scale.shape) != (M, K // block_k):
            raise ValueError(
                f"A_scale 形状应为 (M, K//block_k)=({M}, {K // block_k})，实际 "
                f"{tuple(A_scale.shape)}"
            )
        if tuple(B_scale.shape) != (E, N2 // block_n, K // block_k):
            raise ValueError(
                f"B_scale 形状应为 (E, 2N//block_n, K//block_k)="
                f"({E}, {N2 // block_n}, {K // block_k})，实际 {tuple(B_scale.shape)}"
            )
        a_scale_arg = A_scale.to(torch.float32).contiguous()
        b_scale_arg = B_scale.to(torch.float32).contiguous()
        block_shape = [block_n, block_k]
        config = {
            "BLOCK_SIZE_M": block_size_m,   # 必须 = 对齐粒度（见文件头说明）
            "BLOCK_SIZE_N": block_n,        # 对齐 block_shape[0]
            "BLOCK_SIZE_K": block_k,        # 对齐 block_shape[1]：同一 K tile 内 scale 恒定
            "GROUP_SIZE_M": 8,              # 仅影响 tile 访问顺序，不影响数值
        }
    else:
        if A.dtype != B.dtype or A.dtype != out_dtype:
            raise RuntimeError(
                f"quant='dense' 要求 A/B/topk_weights 同 dtype（题面 make_inputs 的 dtype），"
                f"实际 {A.dtype} / {B.dtype} / {out_dtype}"
            )
        if A_scale.numel() != 1 or B_scale.numel() != 1:
            raise ValueError(
                "dense 模式下 A_scale/B_scale 应为 (1,) 占位张量（题面 io），实际 "
                f"{tuple(A_scale.shape)} / {tuple(B_scale.shape)}"
            )
        # aiter 在 dense 分支显式断言 A_scale is None and B_scale is None
        # （moe_op_silu_fused.py:1049-1050）：占位张量按题面语义被忽略，传 None。
        a_scale_arg = None
        b_scale_arg = None
        block_shape = None
        config = {
            "BLOCK_SIZE_M": block_size_m,   # 必须 = 对齐粒度（见文件头说明）
            "BLOCK_SIZE_N": 64,             # aiter dense 缺省族
            "BLOCK_SIZE_K": 32,             # aiter dense 缺省族；EVEN_K 由 heuristics 处理尾块
            "GROUP_SIZE_M": 8,
        }

    # ---------- layout / 索引准备（全部为无拷贝或纯布局操作） ----------
    A = A.contiguous()
    B = B.contiguous()
    topk_weights = topk_weights.contiguous()            # aiter 断言 stride(1) == 1
    sorted_token_ids = sorted_token_ids.to(torch.int32).contiguous()  # aiter 断言 stride(0) == 1
    expert_ids = expert_ids.to(torch.int32).contiguous()
    num_tokens_post_padded = num_tokens_post_padded.to(torch.int32).contiguous()

    # C 为原地输出：行 = M*top_k（sorted_token_ids 的有效前缀覆盖每一行，题面保证
    # 行行有主、专家间无归约），列 = N = B.shape[1] // 2。
    out = torch.empty((M * top_k, N), dtype=out_dtype, device=A.device)

    fused_moe_silu(
        A=A,
        B=B,
        C=out,
        A_scale=a_scale_arg,
        B_scale=b_scale_arg,
        B_zp=None,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        block_shape=block_shape,
        config=config,
    )
    torch.cuda.synchronize()

    ctx = {
        "path": "aiter.ops.triton.moe_op_silu_fused.fused_moe_silu"
                " -> _fused_moe_silu_kernel (non-persistent)",
        "quant": quant,
        "use_fp8_w8a8": use_fp8_w8a8,
        "mul_routed_weight": mul_routed_weight,
        "block_shape": block_shape,
        "config": dict(config),
        "M": M, "N": N, "K": K, "E": E, "top_k": top_k,
        "block_size_m": block_size_m,
        "out_shape": [M * top_k, N],
        "out_dtype": str(out_dtype),
    }
    return out, ctx
