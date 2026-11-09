# 1016_lean_atten_paged -- ragged-batch decode attention over a paged KV cache
# (KernelBench compatible task file).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=1016, backend=triton.

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """Batched decode attention over a paged KV cache: ragged batch, 16 queries per
    request, one independent physical block table per head, and blocks of the
    physical pool that nothing references are garbage data.

    The KV cache is paged with a fixed page_size: the h-th head slice of k / v
    (shape [total_kv_len, head_size]) is split into total_kv_len / page_size
    physical blocks; the block table of head h, kv_block_tables[h], lists in order
    the physical block ids referenced by each request, where the blocks
    referenced by request b are
      kv_block_tables[h, batch_num_block_n[b-1] : batch_num_block_n[b]]
    (starting at 0 for b=0; batch_num_block_n is the cumulative sum of the
    per-request block counts, and its last element is the total number of used
    blocks total_used_blocks). Physical blocks that no table entry references are
    garbage data and must not take part in the computation.

    Mathematical definition: let s_b be the starting block offset of request b
    (0 for b=0, otherwise s_b = batch_num_block_n[b-1]); its logical KV length is
    L_b = block count * page_size, and logical position j maps to physical row
    row(j) = kv_block_tables[h, s_b + j // page_size] * page_size + j % page_size.
    For every head h and request b (query row range [b*n_ctx_q, (b+1)*n_ctx_q)):
      scores[i, j] = sm_scale * <q[h, b*n_ctx_q+i, :], k[h, row(j), :]>
      p[i, :] = softmax_j(scores[i, :])
      out[h, b*n_ctx_q+i, :] = sum_j p[i, j] * v[h, row(j), :]
    softmax uses a numerically stable (online) implementation, intermediate
    accumulation is in float32, and the output is cast back to the input dtype.
    Boundary behaviour: every request's KV length is a positive integer multiple
    of page_size (a partially filled tail block is outside the task scope), so the
    softmax denominator is always positive; requests and heads do not affect each
    other.

    Input/output specification:
      q                 [num_heads, n_ctx_q * batch, head_size] float16/bfloat16
      k, v              [num_heads, total_kv_len, head_size] same dtype as q (physical block pool)
      kv_block_tables   [num_heads, total_used_blocks] int32 (0-based physical block ids,
                        values < total_kv_len / page_size)
      batch_num_block_n [batch] int32, cumulative sum of the per-request block counts, monotonically increasing
      out               [num_heads, n_ctx_q * batch, head_size] same dtype as q
    Global-domain constraints: head_size in {16, 32, 64, 128, 256}; page_size is
    fixed at 64; n_ctx_q = 16 (the decode form is fixed); each request's KV length
    is a positive integer multiple of page_size and >= page_size;
    num_kv_heads == num_q_heads (MHA, no GQA).

    Implementation constraints (violations are scored as failures):
      - The core computation (paged KV gathering / QK^T / softmax / PV) must be
        done inside the submitted file; calling external attention operator
        libraries such as scaled_dot_product_attention / sdpa / flash_attn, and
        torch.matmul / torch.bmm / torch.mm / torch.baddbmm / torch.einsum /
        torch.softmax (including ATen-equivalent shortcuts such as F.softmax,
        Tensor.matmul and the @ operator) is forbidden.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Use float32 for intermediate accumulation; cast the output back to the
        input dtype.
      - The scheduling strategy (persistent single kernel or a regular grid) is
        free to choose; only semantic correctness and performance are assessed.

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: ragged-batch decode, non-quantized fp16/bf16, no causal mask, no
    ALiBi/sliding window, no GQA.
    """

    def __init__(self, head_size: int, page_size: int = 64, sm_scale=0.5):
        super().__init__()
        self.head_size = int(head_size)
        self.page_size = int(page_size)
        self.sm_scale = float(sm_scale)

    def forward(self, q, k, v, kv_block_tables, batch_num_block_n):
        # The KernelBench evaluator casts every input tensor to the evaluation
        # precision (fp32 for the triton backend); the block ids and the cumulative
        # block counts are small integers that fp32 represents exactly, so they are
        # recovered losslessly here
        kv_block_tables = kv_block_tables.to(torch.long)
        batch_num_block_n = batch_num_block_n.to(torch.int32)

        num_heads, m_total, head_dim = q.shape
        batch = int(batch_num_block_n.numel())
        n_ctx_q = m_total // batch
        page = self.page_size
        assert num_heads == k.shape[0] == v.shape[0], "the head counts of q/k/v must match"
        assert k.shape[1] == v.shape[1], "the KV pool lengths of k/v must match"

        offs_in_block = torch.arange(page, device=q.device)
        out = torch.empty_like(q)
        for h in range(num_heads):
            for b in range(batch):
                s0 = 0 if b == 0 else int(batch_num_block_n[b - 1])
                s1 = int(batch_num_block_n[b])
                blocks = kv_block_tables[h, s0:s1]                     # [num_blocks_b]
                # logical position j -> physical row: row(j) = table[s0 + j//page]*page + j%page
                rows = (blocks[:, None] * page + offs_in_block[None, :]).reshape(-1)
                kb = k[h].index_select(0, rows).to(torch.float32)      # [L_b, D]
                vb = v[h].index_select(0, rows).to(torch.float32)      # [L_b, D]
                qb = q[h, b * n_ctx_q:(b + 1) * n_ctx_q, :].to(torch.float32)
                scores = torch.matmul(qb, kb.transpose(0, 1)) * self.sm_scale
                probs = torch.softmax(scores, dim=-1)                  # [n_ctx_q, L_b]
                out[h, b * n_ctx_q:(b + 1) * n_ctx_q, :] = torch.matmul(probs, vb).to(q.dtype)
        return out


def get_init_inputs():
    return [64, 64, 0.5]   # head_size, page_size (fixed at 64 by the task), sm_scale (always 0.5 officially)


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the evaluator
    # calls this function after set_seed, so multiple correctness trials therefore
    # see diverse inputs
    num_heads = 32
    batch = 4
    n_ctx_q = 16
    head_size = 64
    page_size = 64         # fixed by the task
    extra_blocks = 4       # the physical block pool exceeds the actual usage: unreferenced slots are garbage data

    ctx_lens = (torch.randint(1, 17, (batch,)) * page_size)   # 1..16 blocks per request
    blocks_per = ctx_lens // page_size
    used_blocks = int(blocks_per.sum())
    pool_blocks = used_blocks + extra_blocks
    # one independent block table per head: the first used_blocks entries of a random permutation of the pool
    kv_block_tables = torch.stack(
        [torch.randperm(pool_blocks)[:used_blocks] for _ in range(num_heads)]
    ).to(torch.int32)
    batch_num_block_n = torch.cumsum(blocks_per, dim=0).to(torch.int32)

    total_kv_len = pool_blocks * page_size
    q = (torch.randn(num_heads, n_ctx_q * batch, head_size) * 0.5).to(torch.float16)
    k = (torch.randn(num_heads, total_kv_len, head_size) * 0.5).to(torch.float16)
    v = (torch.randn(num_heads, total_kv_len, head_size) * 0.5).to(torch.float16)
    return [q, k, v, kv_block_tables, batch_num_block_n]


def make_inputs(num_heads: int, batch: int, n_ctx_q: int, head_size: int,
                ctx_lens=None, dtype: str = "float16", extra_blocks: int = 0,
                page_size: int = 64, seed: int = 0, seq_lens=None):
    """Generate inputs from a public/hidden/perf case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device). ctx_lens (or its alias seq_lens -- the conventional parameter
    name of the offline final-audit harness, the two are equivalent) may be given
    explicitly (each request's KV length, which must be a positive integer
    multiple of page_size); when both are absent, each request takes 1..16 random
    blocks. When extra_blocks > 0 the physical block pool gets that many extra
    garbage blocks appended. The task fixes page_size=64.
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)

    if ctx_lens is None:
        ctx_lens = seq_lens
    if ctx_lens is None:
        ctx_lens = (torch.randint(1, 17, (batch,), generator=gen) * page_size).tolist()
    assert len(ctx_lens) == batch, "ctx_lens length must equal batch"
    for s in ctx_lens:
        assert s > 0 and s % page_size == 0, "each request's KV length must be a positive integer multiple of page_size"

    blocks_per = torch.tensor([s // page_size for s in ctx_lens], dtype=torch.int64)
    used_blocks = int(blocks_per.sum())
    pool_blocks = used_blocks + int(extra_blocks)
    kv_block_tables = torch.stack(
        [torch.randperm(pool_blocks, generator=gen)[:used_blocks] for _ in range(num_heads)]
    ).to(torch.int32)
    batch_num_block_n = torch.cumsum(blocks_per, dim=0).to(torch.int32)

    total_kv_len = pool_blocks * page_size
    q = (torch.randn(num_heads, n_ctx_q * batch, head_size, generator=gen) * 0.5).to(dt)
    k = (torch.randn(num_heads, total_kv_len, head_size, generator=gen) * 0.5).to(dt)
    v = (torch.randn(num_heads, total_kv_len, head_size, generator=gen) * 0.5).to(dt)
    return q, k, v, kv_block_tables, batch_num_block_n
