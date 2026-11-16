# 1002_paged_attention -- the model_class (KernelBench compatible) variant of 1001.
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here. The computation is
# line-by-line identical to 1001's reference.py (tests/test_1002_model_class.py
# checks that the two produce equal outputs).
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=1002, backend=triton.

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """Paged attention in the decode phase (single-token query), supporting both
    MHA and GQA.

    The KV cache is stored page-wise with a fixed block_size in the physical block
    pools key_cache / value_cache, and each sequence addresses them indirectly
    through the physical block ids in block_tables; the KV length that actually
    participates in attention for each sequence is given by seq_lens[s], while the
    remaining slots in the pool hold garbage data and must not participate in the
    computation. For query head h of sequence s (whose KV head is
    h // query_group_size):
      logits[l] = scale * <query[s,h,:], key_cache[bt[s,l//bs], h_kv, l%bs, :]>
      p = softmax(logits[0:seq_lens[s]])
      out[s,h,:] = sum_l p[l] * value_cache[bt[s,l//bs], h_kv, l%bs, :]
    The softmax uses an online (numerically stable) formulation; intermediates are
    accumulated in float32 and the output is cast back to the input dtype.

    Implementation constraints (violations are scored as failures):
      - The core computation (QK^T / softmax / PV) must be completed inside the
        submitted file; calling scaled_dot_product_attention / sdpa / flash_attn /
        aiter / torch.matmul / torch.bmm / torch.einsum / torch.softmax is
        forbidden.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Accumulate intermediates in float32; cast the output back to the input
        dtype.

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: single-token decode, non-quantized KV, no ALiBi, no sliding window.
    """

    def __init__(self, head_size: int, scale=None):
        super().__init__()
        self.scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_size)

    def forward(self, query, key_cache, value_cache, block_tables, seq_lens):
        # The KernelBench evaluator casts every input tensor to the evaluation
        # precision (fp32 for the triton backend); block ids and sequence lengths
        # are small integers exactly representable in fp32, restored losslessly
        # here.
        block_tables = block_tables.to(torch.long)
        seq_lens = seq_lens.to(torch.int32)

        B, H_Q, D = query.shape
        H_KV = key_cache.shape[1]
        bs = key_cache.shape[2]
        group = H_Q // H_KV
        assert H_Q == H_KV * group, "num_q_heads must be an integer multiple of num_kv_heads"

        out = torch.empty_like(query)
        for s in range(B):
            seq_len = int(seq_lens[s])
            num_blocks = (seq_len + bs - 1) // bs
            physical = block_tables[s, :num_blocks]

            # Gather this sequence's K/V according to the block table:
            # [num_blocks, H_KV, bs, D] -> [H_KV, seq_len, D]
            k = key_cache[physical].permute(1, 0, 2, 3).reshape(H_KV, num_blocks * bs, D)[:, :seq_len].to(torch.float32)
            v = value_cache[physical].permute(1, 0, 2, 3).reshape(H_KV, num_blocks * bs, D)[:, :seq_len].to(torch.float32)

            # GQA: the K/V of one KV head is replicated to every query head in the group
            if group > 1:
                k = k.repeat_interleave(group, dim=0)   # [H_Q, seq_len, D]
                v = v.repeat_interleave(group, dim=0)

            q = query[s].to(torch.float32) * self.scale      # [H_Q, D]
            logits = torch.einsum("hd,hld->hl", q, k)   # [H_Q, seq_len]
            probs = torch.softmax(logits, dim=-1)
            out[s] = torch.einsum("hl,hld->hd", probs, v).to(query.dtype)
        return out


def get_init_inputs():
    return [64]  # head_size; scale defaults to 1/sqrt(head_size)


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the
    # evaluator calls this function after set_seed, so multiple correctness
    # trials therefore see diverse inputs.
    num_seqs = 8
    num_kv_heads = 8
    query_group_size = 4      # num_q_heads = 32
    head_size = 64
    block_size = 16
    max_seq_len = 1024

    num_q_heads = num_kv_heads * query_group_size
    max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    # The physical block pool is larger than what is actually used: unreferenced
    # slots hold garbage data, which exercises the masking.
    num_blocks = num_seqs * max_blocks_per_seq + 8

    seq_lens = torch.randint(1, max_seq_len + 1, (num_seqs,), dtype=torch.int32)
    perm = torch.randperm(num_blocks)
    block_tables = perm[: num_seqs * max_blocks_per_seq].to(torch.int32).reshape(num_seqs, max_blocks_per_seq)

    query = torch.randn(num_seqs, num_q_heads, head_size).to(torch.float16)
    key_cache = torch.randn(num_blocks, num_kv_heads, block_size, head_size).to(torch.float16)
    value_cache = torch.randn(num_blocks, num_kv_heads, block_size, head_size).to(torch.float16)
    return [query, key_cache, value_cache, block_tables, seq_lens]
