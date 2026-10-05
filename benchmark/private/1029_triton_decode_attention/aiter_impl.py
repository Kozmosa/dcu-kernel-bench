# aiter_impl.py — 1029_triton_decode_attention 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
#
# 来源（aiter pinned 检出 commit c39fff8c77df4e80617649e92fa3c2615f2c43d1）：
#   aiter/ops/triton/triton_decode_attention.py
#   sha256 b35f2d795254cda45f863ed627e7b2a7782f3310ef45c01c5606a961d19b8d44
#      —— device_kernel：_fwd_kernel_stage1 / _fwd_grouped_kernel_stage1（QK^T +
#         在线 softmax + PV）与 _fwd_kernel_stage2（跨 split 的 max 归约）
#
# 公开 host 入口（本文件唯一调用的算子入口，行号指上述源文件）：
#   decode_attention_fwd(q, k_buffer, v_buffer, o, req_to_token, b_seq_len,
#                        attn_logits, num_kv_splits, sm_scale,
#                        page_size=1, logit_cap=0.0)          # L724-L769
#   按 kv_group_num = q.shape[1] // v_buffer.shape[-2] 自行分派：
#     == 1 -> decode_attention_fwd_normal  (_decode_att_m_fwd,       L189)
#     >  1 -> decode_attention_fwd_grouped (_decode_grouped_att_m_fwd, L469)
#   两路都以 _decode_softmax_reducev_fwd（L625, stage2）收尾。
#
# 调用约定取自 aiter 官方测试
#   op_tests/triton_tests/test_decode_attention.py (L11, L112-L160)
#   sha256 7a72a4224bd84d9c43f9c5c9f6a95191356d5502c4239ec7f41e4913e9470f86
#   —— attn_logits 为 [B, H_Q, num_kv_splits, D_V + 1] 的 **float32** 临时缓冲，
#      最后一列放 stage1 的部分 logsumexp（e_max + log(e_sum)），stage2 读回后
#      做跨 split 重归一化；o 的 dtype 与 q 一致（半精度，stage2 内 fp32 累加后
#      直接 store 到半精度目标）。
#
# 布局对齐（题面 -> aiter，逐张量）：
#   q           [B, H_Q, D_QK]                        -> 原样（stride(0)/stride(1) 传参）
#   key_cache   [num_pages, page_size, H_KV, D_QK]    -> 原样。kernel 以
#               stride(-3)=页步长、stride(-2)=头步长 寻址（L230-L231），
#               故必须保持 (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM) 尾三维布局；
#               题面 make_inputs 已产出该布局，无需 permute。
#   value_cache 同 key_cache（D_V 维）                 -> 原样
#   page_tables [B, max_pages_per_seq] i32            -> req_to_token。kernel 只按
#               Req_to_tokens.stride(0) + (offs_n // PAGE_SIZE) 索引（L125-L131），
#               与官方测试要求的 [B, num_pages, 1] 尾维 1 在寻址上等价（准入记录
#               冲突处理 ②）。max_pages_per_seq 必须 >= ceil(max(seq_lens)/page_size)。
#   seq_lens    [B] i32                               -> b_seq_len
#   输出 out    [B, H_Q, D_V]，dtype 同 q —— 与题面 reference 输出同形同 dtype，
#               单张量、无需打包。

import math

import torch

# 与 aiter/ops/triton/triton_decode_attention.py L37 一致：该源文件仅面向 HIP
# （BLOCK_N 与 num_warps 的选择依赖 is_hip_）。
_IS_HIP = True


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方 decode attention 实现。

    inputs     : [q, key_cache, value_cache, page_tables, seq_lens]
                 q           [B, H_Q, D_QK]                           fp16/bf16
                 key_cache   [num_pages, page_size, H_KV, D_QK]       同 q
                 value_cache [num_pages, page_size, H_KV, D_V]        同 q
                 page_tables [B, max_pages_per_seq]                   int32
                 seq_lens    [B]                                      int32
    init_kwargs: page_size / head_dim_qk / head_dim_v / num_kv_splits / scale
                 （与 Model.__init__ 同名；scale 缺省 1/sqrt(head_dim_qk)）

    返回 (out, ctx)；out 为 [B, H_Q, D_V]，dtype 与 q 一致。
    """
    from aiter.ops.triton.triton_decode_attention import decode_attention_fwd

    q, key_cache, value_cache, page_tables, seq_lens = inputs

    page_size = int(init_kwargs.get("page_size", 0))
    head_dim_qk = int(init_kwargs.get("head_dim_qk", 0))
    head_dim_v = int(init_kwargs.get("head_dim_v", 0))
    num_kv_splits = int(init_kwargs.get("num_kv_splits", 8))
    scale = init_kwargs.get("scale", None)
    sm_scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim_qk)

    if not _IS_HIP:
        raise RuntimeError(
            "该 aiter 源文件仅面向 HIP/DCU（页面大小与 num_warps 选择依赖 is_hip_）"
        )
    if page_size < 1:
        raise ValueError(f"page_size 必须 >= 1，收到 {page_size}")
    if head_dim_qk < 16 or head_dim_v < 1:
        raise ValueError(
            f"head_dim_qk/head_dim_v 越界: {head_dim_qk}/{head_dim_v}"
            "（task.yaml 保证 head_dim_qk/head_dim_v >= 16）"
        )
    if num_kv_splits < 1:
        raise ValueError(f"num_kv_splits 必须 >= 1，收到 {num_kv_splits}")

    B, H_Q, D_QK = q.shape
    num_pages, cache_page_size, H_KV, D_QK_cache = key_cache.shape
    D_V = value_cache.shape[-1]

    if D_QK != head_dim_qk or D_V != head_dim_v:
        raise ValueError(
            f"输入头维与 init_kwargs 不一致: q/k {D_QK}/{D_QK_cache} vs {head_dim_qk}，"
            f"v {D_V} vs {head_dim_v}"
        )
    if cache_page_size != page_size:
        raise ValueError(
            f"key_cache 页大小 {cache_page_size} 与 init page_size {page_size} 不一致"
        )
    if H_Q % H_KV != 0:
        raise ValueError(f"H_Q={H_Q} 必须是 H_KV={H_KV} 的整数倍（GQA 整除约束）")
    if D_QK_cache != D_QK:
        raise ValueError(f"q 与 key_cache 的 QK 头维不一致: {D_QK} vs {D_QK_cache}")

    kv_group_num = H_Q // H_KV
    path = "normal_mha" if kv_group_num == 1 else "grouped_gqa"

    # kernel 按 int32 指针读取 Req_to_tokens / B_Seqlen；评测器可能把输入统一
    # cast 成 fp32（在线路径），此处显式转回整型。
    page_tables_i32 = page_tables.to(torch.int32).contiguous()
    seq_lens_i32 = seq_lens.to(torch.int32).contiguous()

    max_pages_per_seq = (int(seq_lens_i32.max().item()) + page_size - 1) // page_size
    if page_tables_i32.shape[1] < max_pages_per_seq:
        raise ValueError(
            f"page_tables 列数 {page_tables_i32.shape[1]} 不足以寻址最长序列 "
            f"{int(seq_lens_i32.max().item())} 个 KV（page_size={page_size}，"
            f"需要 {max_pages_per_seq} 列）"
        )

    # q 必须是 [B, H_Q, D_QK] 连续布局（kernel 直接取 stride(0)/stride(1)）。
    q_c = q.contiguous()
    key_cache = key_cache.contiguous()
    value_cache = value_cache.contiguous()

    # stage1 的部分结果缓冲（对照官方测试 test_decode_attention.py L112-L116）：
    # [B, H_Q, num_kv_splits, D_V + 1]，最后一列是部分 logsumexp。
    # stage2 对 split_kv_end <= split_kv_start 的切片不读（L603-L607），
    # 与 stage1 不写（L121）范围一致，故未初始化槽位是安全的。
    attn_logits = torch.empty(
        (B, H_Q, num_kv_splits, D_V + 1),
        dtype=torch.float32,
        device=device,
    )
    if attn_logits.shape[2] != num_kv_splits:  # 对应源码 L737 的 assert
        raise ValueError("attn_logits 的 split 维与 num_kv_splits 不一致")

    out = torch.empty((B, H_Q, D_V), dtype=q.dtype, device=device)

    decode_attention_fwd(
        q_c,
        key_cache,
        value_cache,
        out,
        page_tables_i32,
        seq_lens_i32,
        attn_logits,
        num_kv_splits,
        sm_scale,
        page_size,
    )

    torch.cuda.synchronize()
    return out, {
        "path": path,
        "aiter_entry": "decode_attention_fwd",
        "kv_group_num": kv_group_num,
        "page_size": page_size,
        "num_kv_splits": num_kv_splits,
        "sm_scale": sm_scale,
        "logit_cap": 0.0,
        "shape": {
            "B": B,
            "H_Q": H_Q,
            "H_KV": H_KV,
            "D_QK": D_QK,
            "D_V": D_V,
            "num_pages": num_pages,
            "max_pages_per_seq": int(page_tables_i32.shape[1]),
        },
    }
