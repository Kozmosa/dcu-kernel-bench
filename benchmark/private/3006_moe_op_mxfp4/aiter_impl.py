# aiter_impl.py — 3006_moe_op_mxfp4 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
# 适配器契约：run(inputs, init_kwargs: dict, device) -> (out, ctx)。
#
# 来源（本地 pinned 检出 .dcu_runs/aiter_pinned，commit c39fff8c77df4e80617649e92fa3c2615f2c43d1）：
#   aiter/ops/triton/moe_op_mxfp4.py
#     sha256 e0edcdf6436e8d42f34058324063121daf5441a68995ab6784586cabbb40367e
#     （与 benchmark/sources/3006_moe_op_mxfp4.yaml 记录的哈希一致）
#     准入声明的核心计算即该文件的 device kernel `_fused_moe_kernel_mxfp4`
#     （module:31）：nibble 不解包——直接以 uint8 打包码字喂 tl.dot_scaled
#     (a_format=b_format="e2m1")、e8m0 块 scale 由 MX_SCALE_BLOCK_K_* 指针按
#     BLOCK_SIZE_K/32 步进加载（module:184-264）、fp32 累加后乘路由权重
#     （module:355-360），与题面 Model.forward 的语义逐条对应。
#
# 公开 host 入口（本适配器调用的唯一算子入口）：
#
#   fused_moe_mxfp4(                                    # moe_op_mxfp4.py:369
#       A: torch.Tensor,              # [M, K//2]      uint8 打包 mxfp4 码字
#       B: torch.Tensor,              # [E, N, K//2]   uint8 打包 mxfp4 码字
#       C: torch.Tensor,              # [M, top_k, N]  bfloat16 输出（in-place 写）
#       A_scale: torch.Tensor,        # [1] fp32  标量（aiter 内 tl.load 单标量）
#       B_scale: torch.Tensor,        # [E] fp32  每专家标量
#       A_mx_scale: torch.Tensor,     # [M, K//32]      uint8 e8m0 块 scale
#       B_mx_scale: torch.Tensor,     # [E, N, K//32]   uint8 e8m0 块 scale
#       topk_weights: torch.Tensor,   # [M, top_k]（要求 stride(1) == 1）
#       topk_ids: torch.Tensor,       # [M, top_k]  仅用其 numel 作 num_valid_tokens
#       sorted_token_ids: torch.Tensor,      # [M*top_k + E*(BS_M-1)] int32
#       expert_ids: torch.Tensor,            # [ceil(L/BS_M)] int32
#       num_tokens_post_padded: torch.Tensor,# [1] int32
#       mul_routed_weight: bool,
#       top_k: int,
#       swizzle_mx_a: bool,
#       swizzle_mx_b: bool,
#       config: Dict[str, Any],       # BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, ...
#       compute_type: tl.dtype,       # 累加器输出 dtype（题面 = tl.bfloat16）
#   ) -> None                         # void：结果写回 C
#
# 官方调用样例（语义权威，op_tests/triton_tests/test_moe_mx.py，sha256
# 9a9aeb0bb009a206053561430e77ed5b590352177dd19c2c03952926c87056b6）：
#   test:280-282  sorted_token_ids, expert_ids, num_tokens_post_padded =
#                     torch_moe_align_block_size_ref(topk_ids, config["BLOCK_SIZE_M"], E)
#   test:316-335  fused_moe_mxfp4(a_tri, b_tri, c_tri, a_scale, b_scale,
#                     a_mx_scales, b_mx_scales, topk_weights, topk_ids,
#                     sorted_token_ids, expert_ids, num_tokens_post_padded,
#                     routed_weight, top_k, swizzle, swizzle, config,
#                     torch_to_triton_dtype[c_tri.dtype])
#   （test:257-258 a_scale = [1.0]、b_scale = [1.0]*E：反量化已在输入侧完成，
#     这两个张量在 kernel 里只是 "accumulator *= a_scale * b_scale" 的恒等因子。）
#
# 与题面 io 的差口（唯一一处，且不引入非 aiter 计算）：
#   题面 io.inputs 只有 a_q/a_scales/b_q/b_scales/topk_weights/topk_ids，
#   **不含** token-expert 对齐调度元数据（sorted_token_ids / expert_ids /
#   num_tokens_post_padded）——task.yaml 与 Model docstring 明确写明这三者是
#   "实现细节，不在输入契约内"。而 aiter 的 host 入口按官方设计把它们当入参
#   （vLLM/aiter 的生产链路里由 aiter 自己的 moe_align_block_size 先生成）。
#   本适配器因此用 **aiter 自己的公开 host 入口** 生成这份元数据：
#     aiter/ops/triton/moe_align_block_size.py::moe_align_block_size_triton
#     （sha256 f950b961fff08ab75e3f29ac4c4788d0096a200883c7b949cf2b9350b485b279，
#      stage1-4 纯 Triton kernel；同目录兄弟适配器 private/3002_moe_align_block_size
#      用的是同一个入口）。没有用任何 torch 高层算子去替代 aiter 的计算。
#   block_size 取 config["BLOCK_SIZE_M"]——必须如此：kernel 内
#   num_pid_m = cdiv(num_tokens_post_padded, BLOCK_SIZE_M)（module:136），
#   只有按同一 BLOCK_SIZE_M 对齐分桶，"一个 M 块只属于一个专家" 才成立，
#   官方测试 test:281 也是这么传的。
#
# 输出契约：官方入口是 void + in-place 写 C；题面 forward 返回单张量
# [M, top_k, N] bfloat16，故适配器预分配 C 并原样返回，无需任何打包/拆分。
# A/B 的 nibble 打包布局（偶下标低 4 位）与 e8m0 解码（2^(s-127)）由 kernel 的
# tl.dot_scaled(..., "e2m1") + uint8 mx_scale 直接消费，与题面 make_inputs 的
# _mxfp4_quant / reference 的 _dequant_mxfp4 布局逐位一致，**不需要**做任何
# permute/repack；题面张量本就是 kernel 要的 [M,K//2] / [E,N,K//2] /
# [M,K//32] / [E,N,K//32] 布局（K 为逻辑特征维）。
#
# autotune：本文件不依赖 autotune——_fused_moe_kernel_mxfp4 是 @triton.jit +
# @triton.heuristics（moe_op_mxfp4.py:25-30），config 由调用方显式给出，
# 不读 AITER_TRITON_CONFIGS_PATH，缺 config JSON 不会失败或退化。
#
# 架构注意（真机基线的前提，非适配器逻辑问题）：官方测试 test:244-245 只在
# 「MXFP4 硬件/编译器可用」的架构上启用本 kernel
# （`arch not in ("gfx946"): pytest.skip("MXFP4 not supported on this architecture")`）。
# 准入记录 admission.note 已登记该 skip 与本题照常出题的理由。若目标 DCU 的
# DTK Triton 不为 uint8/e2m1 的 tl.dot_scaled 提供 lowering，本 kernel 会在
# 首次编译时失败——此时本题的 aiter 基线在真机上采不到（题面 reference 与生成
# 实现不受影响）。

import torch
import triton
import triton.language as tl

# 官方测试 test_moe_mx.py:264-274 的固定 config（本 kernel 唯一的 config 来源）
_OFFICIAL_CONFIG = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 4,
    "num_warps": 8,
    "num_stages": 2,
    "waves_per_eu": 0,
    "matrix_instr_nonkdim": 16,
    "kpack": 1,
}

# 与 kernel 内 MX_PACK_DIVISOR 一致（moe_op_mxfp4.py:102）
_MX_PACK_DIVISOR = 32
# 题面 io.outputs 固定 bfloat16（Model.forward 末段 .to(torch.bfloat16)）
_OUT_DTYPE = torch.bfloat16


def _int_kw(init_kwargs, name, default, *, minimum=1, multiple_of=None):
    """按名取整型构造参数；越界立即 raise（绝不静默用错参数）。

    本题 io.init_inputs 为 []，正常路径下 init_kwargs 恒为 {}，这些键只作为
    显式覆盖口存在（例如真机上想换个 BLOCK 组合复采基线）。
    """
    raw = init_kwargs.get(name, default)
    value = int(raw)
    if value < minimum:
        raise ValueError(
            f"init_kwargs['{name}']={raw} 非法：要求 >= {minimum}"
        )
    if multiple_of is not None and value % multiple_of != 0:
        raise ValueError(
            f"init_kwargs['{name}']={raw} 非法：要求是 {multiple_of} 的整数倍"
            "（mxfp4 的 BLOCK_SIZE_K 必须能被 32 整除，见 kernel 内 "
            "static_assert BLOCK_SIZE_K % MX_PACK_DIVISOR == 0）"
        )
    return value


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方 fused_moe_mxfp4（mxfp4 × mxfp4 MoE grouped GEMM）。

    inputs（顺序 = reference.py::make_inputs 的返回序）：
      a_q          [M, K//2]     uint8  打包 e2m1 激活码字（偶下标在低 4 位）
      a_scales     [M, K//32]    uint8  e8m0 块 scale 字节
      b_q          [E, N, K//2]  uint8  打包 e2m1 专家权重码字
      b_scales     [E, N, K//32] uint8  e8m0 块 scale 字节
      topk_weights [M, top_k]    路由权重（bf16/fp16/fp32，> 0）
      topk_ids     [M, top_k]    选中专家编号（int64，∈ [0, E)）
    init_kwargs：{}（本题无超参；可选覆盖见 _int_kw 的说明）
    device     ：输入所在设备

    返回 (out, ctx)：out = [M, top_k, N] bfloat16，与 reference 输出同形同 dtype。
    """
    # aiter 顶层 import 很重，按部署约定在函数内最小导入（两个模块本身只依赖
    # torch/triton，不触发 aiter 的 C++ 模块 JIT）
    from aiter.ops.triton.moe_op_mxfp4 import fused_moe_mxfp4
    from aiter.ops.triton.moe_align_block_size import moe_align_block_size_triton

    a_q, a_scales, b_q, b_scales, topk_weights, topk_ids = inputs

    # ---------- config（官方固定值 + 显式覆盖口） ----------
    block_size_m = _int_kw(
        init_kwargs, "block_size_m", _OFFICIAL_CONFIG["BLOCK_SIZE_M"], minimum=16
    )
    block_size_n = _int_kw(
        init_kwargs, "block_size_n", _OFFICIAL_CONFIG["BLOCK_SIZE_N"], minimum=16
    )
    block_size_k = _int_kw(
        init_kwargs,
        "block_size_k",
        _OFFICIAL_CONFIG["BLOCK_SIZE_K"],
        minimum=_MX_PACK_DIVISOR,
        multiple_of=_MX_PACK_DIVISOR,
    )
    group_size_m = _int_kw(
        init_kwargs, "group_size_m", _OFFICIAL_CONFIG["GROUP_SIZE_M"], minimum=1
    )
    num_warps = _int_kw(
        init_kwargs, "num_warps", _OFFICIAL_CONFIG["num_warps"], minimum=1
    )
    num_stages = _int_kw(
        init_kwargs, "num_stages", _OFFICIAL_CONFIG["num_stages"], minimum=1
    )
    config = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_K": block_size_k,
        "GROUP_SIZE_M": group_size_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
        # 以下三项是官方 config 里的后端 launch 选项，原样透传（HYGON/AMD
        # Triton 后端识别；官方测试也是这么传的）
        "waves_per_eu": _OFFICIAL_CONFIG["waves_per_eu"],
        "matrix_instr_nonkdim": _OFFICIAL_CONFIG["matrix_instr_nonkdim"],
        "kpack": _OFFICIAL_CONFIG["kpack"],
    }

    # ---------- 入口无损恢复 dtype（与 reference.py:167-172 同义） ----------
    # 离线路径（record_baseline.py:170）直接搬 make_inputs 的张量，dtype 已是
    # uint8/int64；在线/其它 harness 可能统一 cast 成 fp32，码字 0..255 与专家
    # 编号（小整数）在 fp32 中精确可表示，此处按 reference 同款方式转回。
    a_q = a_q.to(torch.uint8).contiguous()
    a_scales = a_scales.to(torch.uint8).contiguous()
    b_q = b_q.to(torch.uint8).contiguous()
    b_scales = b_scales.to(torch.uint8).contiguous()
    if topk_ids.dtype not in (torch.int32, torch.int64):
        topk_ids = topk_ids.to(torch.long)
    topk_ids = topk_ids.contiguous()
    topk_weights = topk_weights.contiguous()   # kernel 要求 stride(1) == 1

    # ---------- shape 校验（题面 io / invariants） ----------
    if a_q.dim() != 2 or a_scales.dim() != 2 or b_q.dim() != 3 or b_scales.dim() != 3:
        raise ValueError(
            "输入的 ndim 与题面不符：a_q/a_scales 应为 2D，b_q/b_scales 应为 3D，"
            f"实得 {tuple(a_q.shape)}/{tuple(a_scales.shape)}/"
            f"{tuple(b_q.shape)}/{tuple(b_scales.shape)}"
        )
    if topk_ids.dim() != 2 or topk_weights.dim() != 2:
        raise ValueError(
            "topk_ids / topk_weights 必须是 2D [M, top_k]，实得 "
            f"{tuple(topk_ids.shape)} / {tuple(topk_weights.shape)}"
        )

    M, k_packed = a_q.shape
    E, N, b_k_packed = b_q.shape
    top_k = int(topk_ids.shape[1])
    K = k_packed * 2                                   # 逻辑特征维（reference 口径）

    if b_k_packed != k_packed:
        raise ValueError(
            f"激活与权重的 K 维不一致：a_q {tuple(a_q.shape)} vs b_q {tuple(b_q.shape)}"
        )
    if K % _MX_PACK_DIVISOR != 0:
        raise ValueError(
            f"K={K} 不满足题面不变量 K % 32 == 0（mxfp4 块 scale 粒度）"
        )
    if a_scales.shape != (M, K // _MX_PACK_DIVISOR):
        raise ValueError(
            f"a_scales 形状应为 {(M, K // _MX_PACK_DIVISOR)}，实得 {tuple(a_scales.shape)}"
        )
    if b_scales.shape != (E, N, K // _MX_PACK_DIVISOR):
        raise ValueError(
            f"b_scales 形状应为 {(E, N, K // _MX_PACK_DIVISOR)}，实得 {tuple(b_scales.shape)}"
        )
    if topk_weights.shape != (M, top_k):
        raise ValueError(
            f"topk_weights 形状应为 {(M, top_k)}，实得 {tuple(topk_weights.shape)}"
        )
    if M < 1 or E < 1 or N < 1 or top_k < 1 or top_k > E:
        raise ValueError(
            f"维度越界：M={M}, E={E}, N={N}, top_k={top_k}"
            "（题面要求 M/E/N >= 1 且 1 <= top_k <= E）"
        )

    dev = a_q.device
    num_valid_tokens = M * top_k             # = topk_ids.numel()（kernel 的 token 上界）
    # 注：不在此处做 topk_ids 取值域的 min/max 扫描——那是一次全量归约 + 设备
    # 同步，会污染本适配器被计时的路径；取值合法性由题面不变量与 make_inputs
    # 契约保证（与兄弟适配器 private/3002 的口径一致）。

    # ---------- 1) token-expert 对齐调度元数据（aiter 官方 triton 入口） ----------
    # 长度取 aiter fused_moe.moe_align_block_size 的同款上界（fused_moe.py:388-398）：
    # sorted 容量 = numel + E*(BS_M-1)，expert_ids 容量 = ceil(容量 / BS_M)，
    # sorted 全域预填哨兵 numel（kernel 内 token_mask = offs_token < num_valid_tokens
    # 因此把填充槽整块屏蔽掉，不会写脏 C）。
    max_num_tokens_padded = num_valid_tokens + E * (block_size_m - 1)
    sorted_token_ids = torch.full(
        (max_num_tokens_padded,), num_valid_tokens, dtype=torch.int32, device=dev
    )
    expert_ids = torch.zeros(
        (triton.cdiv(max_num_tokens_padded, block_size_m),),
        dtype=torch.int32,
        device=dev,
    )
    num_tokens_post_padded = torch.empty((1,), dtype=torch.int32, device=dev)

    moe_align_block_size_triton(
        topk_ids,
        E,
        block_size_m,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    )

    # ---------- 2) aiter 官方 mxfp4 MoE grouped GEMM ----------
    # a_scale / b_scale 是 kernel 内 "accumulator *= a_scale * b_scale" 的恒等因子：
    # 题面输入已是反量化前的码字 + e8m0 块 scale，故取 1.0（官方测试同款，
    # test_moe_mx.py:257-258）。
    a_scale = torch.ones((1,), dtype=torch.float32, device=dev)
    b_scale = torch.ones((E,), dtype=torch.float32, device=dev)

    out = torch.empty((M, top_k, N), dtype=_OUT_DTYPE, device=dev)

    fused_moe_mxfp4(
        a_q,                 # A: [M, K//2] uint8 打包 mxfp4（tl.dot_scaled "e2m1"）
        b_q,                 # B: [E, N, K//2] uint8
        out,                 # C: [M, top_k, N] bfloat16，in-place 写
        a_scale,             # A_scale: [1] fp32
        b_scale,             # B_scale: [E] fp32
        a_scales,            # A_mx_scale: [M, K//32] uint8 e8m0
        b_scales,            # B_mx_scale: [E, N, K//32] uint8 e8m0
        topk_weights,        # [M, top_k]，stride(1) == 1
        topk_ids,            # 仅用 numel 作 num_valid_tokens
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        True,                # mul_routed_weight：题面恒为 True（乘 topk_weights）
        top_k,
        False,               # swizzle_mx_a：官方测试 swizzle_mx_scale=False
        False,               # swizzle_mx_b
        config,
        tl.bfloat16,         # compute_type：题面输出 bfloat16
    )
    torch.cuda.synchronize()

    ctx = {
        "aiter_path": "aiter.ops.triton.moe_op_mxfp4.fused_moe_mxfp4 "
                      "(+ aiter.ops.triton.moe_align_block_size.moe_align_block_size_triton "
                      "生成 sorted_token_ids/expert_ids/num_tokens_post_padded)",
        "aiter_source_sha256": "e0edcdf6436e8d42f34058324063121daf5441a68995ab6784586cabbb40367e",
        "M": int(M),
        "N": int(N),
        "K_logical": int(K),
        "K_packed": int(k_packed),
        "E": int(E),
        "top_k": int(top_k),
        "num_valid_tokens": int(num_valid_tokens),
        "max_num_tokens_padded": int(max_num_tokens_padded),
        "num_m_blocks": int(triton.cdiv(max_num_tokens_padded, block_size_m)),
        "mul_routed_weight": True,
        "swizzle_mx_a": False,
        "swizzle_mx_b": False,
        "compute_type": "tl.bfloat16",
        "config": {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_size_n,
            "BLOCK_SIZE_K": block_size_k,
            "GROUP_SIZE_M": group_size_m,
        },
        "note": "题面 io 不含调度元数据（task.yaml: token-expert 对齐元数据不在输入契约内），"
                "由 aiter 自己的 triton 对齐入口生成；a_scale/b_scale 取 1.0（恒等因子）",
    }
    return out, ctx
