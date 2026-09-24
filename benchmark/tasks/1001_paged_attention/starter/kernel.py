# 1001_paged_attention — Starter（Triton track）
#
# 评测约定：
#   - 补全 paged_attention_decode_kernel 与 paged_attention_decode；
#     函数签名与 constexpr 参数语义不可改。
#   - 核心计算（QK^T、在线 softmax、PV 累加）必须在本文件的 Triton kernel
#     中完成；task.yaml 的 forbidden 列表会被静态审计，禁止转调 torch/ATen
#     或任何注意力库。
#   - 中间累加用 float32；输出 cast 回输入 dtype。
#   - 参考 task.yaml 的语义描述与 reference.py 的行为（评测以 reference 为准）。

import triton
import triton.language as tl


# TODO(Agent): 实现设备 kernel。
# 建议 grid: (num_seqs, num_kv_heads)（GQA 一组 query heads 共享 KV 加载）
# 或 (num_seqs * num_q_heads,)（MHA 简单路径）；都必须掩蔽 seq_lens 与
# HEAD_SZ_POW2 / KV_BLK_SZ_POW2 的 padding。
@triton.jit
def paged_attention_decode_kernel(
    out_ptr,            # [num_seqs, num_q_heads, head_size]
    q_ptr,              # [num_seqs, num_q_heads, head_size]
    k_ptr,              # [num_blocks, num_kv_heads, block_size, head_size]
    v_ptr,              # [num_blocks, num_kv_heads, block_size, head_size]
    bt_ptr,             # [num_seqs, max_blocks_per_seq] int32
    seq_lens_ptr,       # [num_seqs] int32
    scale,              # float
    stride_q_s, stride_q_h, stride_q_d,
    stride_o_s, stride_o_h, stride_o_d,
    stride_k_b, stride_k_h, stride_k_bs, stride_k_d,
    stride_v_b, stride_v_h, stride_v_bs, stride_v_d,
    stride_bt_s,
    HEAD_SZ: tl.constexpr,        # 真实 head_size（可为非 2 次幂，如 96）
    HEAD_SZ_POW2: tl.constexpr,   # next_power_of_2(head_size)
    KV_BLK_SZ: tl.constexpr,      # 真实 block_size
    KV_BLK_SZ_POW2: tl.constexpr, # next_power_of_2(block_size)
    GROUP: tl.constexpr,          # num_q_heads // num_kv_heads
):
    # TODO(Agent):
    #   seq_idx / head 索引 -> 逐物理块加载 K/V（经 bt_ptr 间接寻址）
    #   在线 softmax：max_logit / exp_sum / acc，float32 累加
    #   out = acc / exp_sum，cast 到 out_ptr.dtype.element_ty
    ...


# Host 启动入口。输入输出均为已在当前设备上的 contiguous tensor。
# TODO(Agent): 计算 constexpr 值、选择 grid 并启动 kernel；返回 out。
def paged_attention_decode(query, key_cache, value_cache,
                           block_tables, seq_lens, scale, out):
    raise NotImplementedError("替换为真实实现")
