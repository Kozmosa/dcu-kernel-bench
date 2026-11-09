# 1001_paged_attention -- Starter (Triton track)
#
# Evaluation contract:
#   - Complete paged_attention_decode_kernel and paged_attention_decode; the
#     function signatures and the semantics of the constexpr parameters must not
#     be changed.
#   - The core computation (QK^T, online softmax, PV accumulation) must be done
#     in the Triton kernel inside this file; the forbidden list in task.yaml is
#     enforced by the static audit, so calling into torch/ATen or any attention
#     library is not allowed.
#   - Accumulate intermediates in float32; cast the output back to the input
#     dtype.
#   - Follow the semantic description in task.yaml and the behaviour of
#     reference.py (the evaluation is based on the reference).

import triton
import triton.language as tl


# TODO(Agent): implement the device kernel.
# Suggested grid: (num_seqs, num_kv_heads) (one GQA group of query heads shares
# the KV load) or (num_seqs * num_q_heads,) (the simple MHA path); either way you
# must mask seq_lens as well as the HEAD_SZ_POW2 / KV_BLK_SZ_POW2 padding.
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
    HEAD_SZ: tl.constexpr,        # the real head_size (may be non-power-of-two, e.g. 96)
    HEAD_SZ_POW2: tl.constexpr,   # next_power_of_2(head_size)
    KV_BLK_SZ: tl.constexpr,      # the real block_size
    KV_BLK_SZ_POW2: tl.constexpr, # next_power_of_2(block_size)
    GROUP: tl.constexpr,          # num_q_heads // num_kv_heads
):
    # TODO(Agent):
    #   seq_idx / head indices -> load K/V block by block (indirect addressing through bt_ptr)
    #   online softmax: max_logit / exp_sum / acc, accumulated in float32
    #   out = acc / exp_sum, cast to out_ptr.dtype.element_ty
    ...


# Host launch entry point. Inputs and outputs are contiguous tensors already on
# the current device.
# TODO(Agent): compute the constexpr values, pick the grid, launch the kernel and
# return out.
def paged_attention_decode(query, key_cache, value_cache,
                           block_tables, seq_lens, scale, out):
    raise NotImplementedError("replace this with the real implementation")
