# 1011_extend_attention -- variable-length attention mixing an extend
# (incremental prefill) segment with a prefix segment: the prefix segment is
# gathered token by token from the KV pool through kv_indices, the extend segment
# is stored contiguously; GQA and a causal mask are supported. model_class
# (KernelBench compatible) task file.
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=1011, backend=triton.

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """Variable-length attention mixing the extend (incremental prefill) phase with a prefix segment (GQA, causal optional).

    Inside the batch each sequence's KV has two segments: the prefix segment is
    already written into the KV pool (k_buffer / v_buffer) and is addressed
    token by token indirectly through the row ids in kv_indices; the extend
    segment holds the tokens newly computed in this step, whose K/V are stored
    contiguously per token in k_extend / v_extend, with the query in q_extend in
    the same order. For sequence i let E_i = qo_indptr[i+1] - qo_indptr[i]
    (number of extend tokens) and P_i = kv_indptr[i+1] - kv_indptr[i] (number of
    prefix tokens). For the t-th query of the extend segment (global position
    T = P_i + t, row s = qo_indptr[i] + t) and query head h (its KV head is
    g = h // query_group_size, query_group_size = num_q_heads // num_kv_heads):
      K_full = concat(k_buffer[kv_indices[kv_indptr[i]:kv_indptr[i+1]]][:, g, :],
                      k_extend[qo_indptr[i]:qo_indptr[i+1]][:, g, :])   P_i + E_i rows
      V_full likewise (taking the head_dim_v head dimension)
      logits[j] = sm_scale * <q_extend[s, h, :], K_full[j]>
                 (sm_scale defaults to 1/sqrt(head_dim_q))
      visibility: the whole prefix is visible; the extend segment with causal=True
      sees only tokens with global position <= T (that is, the first t+1 extend
      tokens), and with causal=False everything is visible
      p = softmax(logits) (numerically stable online implementation)
      out[s, h, :] = sum_j p[j] * V_full[j]
    Under causal the query's own position is always visible, so the softmax
    denominator is always > 0. K's head dimension equals Q's head dimension
    head_dim_q; V's head dimension head_dim_v may differ from it.

    Input/output specification (evaluation dtype is float16 / bfloat16, all tensors contiguous):
      q_extend   evaluation dtype (total_extend, num_q_heads, head_dim_q)
      k_extend   evaluation dtype (total_extend, num_kv_heads, head_dim_q)
      v_extend   evaluation dtype (total_extend, num_kv_heads, head_dim_v)
      k_buffer   evaluation dtype (num_pool, num_kv_heads, head_dim_q)   prefix K pool
      v_buffer   evaluation dtype (num_pool, num_kv_heads, head_dim_v)   prefix V pool
      qo_indptr  int32 (num_seqs + 1) -- prefix sum of extend lengths (each segment >= 1)
      kv_indptr  int32 (num_seqs + 1) -- prefix sum of prefix lengths (each segment >= 0, may be all 0)
      kv_indices int32 (total_prefix) -- pool row ids of the prefix tokens, any
                 permutation is allowed; num_pool may exceed total_prefix, and the
                 unreferenced pool rows are garbage data that must not take part
                 in the computation
      returns out evaluation dtype (total_extend, num_q_heads, head_dim_v)

    Implementation constraints (violations are scored as failures):
      - The core computation (QK^T / softmax / PV) must be done inside the
        submitted file; calling scaled_dot_product_attention / sdpa / flash_attn
        / aiter / torch.matmul / torch.bmm / torch.einsum / torch.softmax is
        forbidden.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Use float32 for intermediate accumulation; cast the output back to the
        input dtype.
      - The evaluator casts every input tensor to fp32 before passing it to
        forward; the indptrs and pool row ids are small integers that fp32
        represents exactly, so .to(torch.int32) / .to(torch.long) at the entry
        recovers them losslessly.
      - The maximum extend length is not an input: the upper bound of the grid's
        third dimension can be derived inside forward from qo_indptr, or
        total_extend can be used directly as a conservative bound (inside the
        kernel, mask the out-of-range query rows with each sequence's
        cur_seq_len_extend).

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: prefix+extend mixed, variable-length (ragged) batch, GQA,
    causal/non-causal, Q/K/V head-dimension relation as above; no custom mask,
    no sliding window, no attention sink, no KV dequantization scale, the logit
    cap is not enabled (always 0), and the prefix segment gets no separate
    prefill-kernel fusion path -- those variants are left for later standalone
    tasks.
    """

    def __init__(self, causal: bool = True, sm_scale=None):
        super().__init__()
        self.causal = bool(causal)
        self.sm_scale = None if sm_scale is None else float(sm_scale)

    def forward(self, q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices):
        # The evaluator casts every input to fp32; the cumulative lengths and the
        # pool row ids are small integers that fp32 represents exactly, so they
        # are recovered losslessly here
        qo_indptr = qo_indptr.to(torch.int32)
        kv_indptr = kv_indptr.to(torch.int32)
        kv_indices = kv_indices.to(torch.long)

        total_extend, H_Q, Dq = q_extend.shape
        H_KV = k_extend.shape[1]
        Dv = v_extend.shape[-1]
        group = H_Q // H_KV
        assert H_Q == H_KV * group, "num_q_heads must be an integer multiple of num_kv_heads"
        scale = self.sm_scale if self.sm_scale is not None else 1.0 / math.sqrt(Dq)
        B = qo_indptr.shape[0] - 1

        out = torch.empty((total_extend, H_Q, Dv), dtype=q_extend.dtype,
                          device=q_extend.device)
        for i in range(B):
            qs, qe = int(qo_indptr[i]), int(qo_indptr[i + 1])
            ks, ke = int(kv_indptr[i]), int(kv_indptr[i + 1])
            prefix_len = ke - ks
            seq_len = qe - qs

            q = q_extend[qs:qe].to(torch.float32)                     # [E, H_Q, Dq]
            k_prefix = k_buffer[kv_indices[ks:ke]].to(torch.float32)  # [P, H_KV, Dq]
            v_prefix = v_buffer[kv_indices[ks:ke]].to(torch.float32)  # [P, H_KV, Dv]
            k_full = torch.cat([k_prefix, k_extend[qs:qe].to(torch.float32)], dim=0)
            v_full = torch.cat([v_prefix, v_extend[qs:qe].to(torch.float32)], dim=0)

            # GQA: the K/V of one KV head is replicated to every query head in the group
            if group > 1:
                k_full = k_full.repeat_interleave(group, dim=1)       # [P+E, H_Q, Dq]
                v_full = v_full.repeat_interleave(group, dim=1)       # [P+E, H_Q, Dv]

            scores = torch.einsum("qhc,khc->hqk", q, k_full) * scale  # [H_Q, E, P+E]
            if self.causal:
                pos_q = prefix_len + torch.arange(seq_len, device=q_extend.device)
                pos_k = torch.arange(prefix_len + seq_len, device=q_extend.device)
                causal_mask = pos_k.unsqueeze(0) > pos_q.unsqueeze(1)  # [E, P+E]
                scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
            p = torch.softmax(scores, dim=-1)
            out[qs:qe] = torch.einsum("hqk,khd->qhd", p, v_full).to(q_extend.dtype)
        return out


def get_init_inputs():
    return [True]   # causal; sm_scale defaults to 1/sqrt(head_dim_q)


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the evaluator
    # calls this function after set_seed, so multiple correctness trials therefore
    # see diverse inputs
    num_seqs = 4
    num_kv_heads = 4
    query_group_size = 4      # num_q_heads = 16
    head_dim_q = 128
    head_dim_v = 64           # V's head dimension differs from that of Q/K
    max_prefix_len = 256
    max_extend_len = 192

    num_q_heads = num_kv_heads * query_group_size
    extend_lens = torch.randint(1, max_extend_len + 1, (num_seqs,), dtype=torch.int32)
    prefix_lens = torch.randint(1, max_prefix_len + 1, (num_seqs,), dtype=torch.int32)
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32),
                           extend_lens.cumsum(0, dtype=torch.int32)])
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32),
                           prefix_lens.cumsum(0, dtype=torch.int32)])
    total_extend = int(qo_indptr[-1])
    total_prefix = int(kv_indptr[-1])

    # The KV pool is larger than the actual usage: unreferenced rows are garbage
    # data, used to check that the indirect addressing is correct
    num_pool = total_prefix + 8
    kv_indices = torch.randperm(num_pool)[:total_prefix].to(torch.int32)

    q_extend = torch.randn(total_extend, num_q_heads, head_dim_q).to(torch.float16)
    k_extend = torch.randn(total_extend, num_kv_heads, head_dim_q).to(torch.float16)
    v_extend = torch.randn(total_extend, num_kv_heads, head_dim_v).to(torch.float16)
    k_buffer = torch.randn(num_pool, num_kv_heads, head_dim_q).to(torch.float16)
    v_buffer = torch.randn(num_pool, num_kv_heads, head_dim_v).to(torch.float16)
    return [q_extend, k_extend, v_extend, k_buffer, v_buffer,
            qo_indptr, kv_indptr, kv_indices]


def make_inputs(num_seqs: int, num_q_heads: int, num_kv_heads: int,
                head_dim_q: int, head_dim_v: int, max_prefix_len: int,
                max_extend_len: int, dtype: str = "float16", seed: int = 0,
                prefix_lens=None, extend_lens=None, causal: bool = True):
    """Generate inputs from a public/hidden case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device). extend_lens / prefix_lens may be given explicitly (to
    construct boundary cases exactly); by default they are sampled uniformly in
    [1, max_extend_len] / [1, max_prefix_len] (all prefixes are 0 when
    max_prefix_len is 0). causal does not affect the input tensor values, it only
    records the Model init hyper-parameter that belongs to the case. The
    consumption order is fixed: extend_lens -> prefix_lens -> randperm ->
    q_extend -> k_extend -> v_extend -> k_buffer -> v_buffer, all drawn from the
    same torch.Generator(seed), so the same seed reproduces them bit for bit.
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)

    if extend_lens is None:
        extend_lens = torch.randint(1, max_extend_len + 1, (num_seqs,),
                                    generator=gen, dtype=torch.int32)
    else:
        if len(extend_lens) == 1 and num_seqs > 1:   # a single value broadcasts to the whole batch
            extend_lens = list(extend_lens) * num_seqs
        extend_lens = torch.tensor(extend_lens, dtype=torch.int32)
        assert len(extend_lens) == num_seqs and int(extend_lens.min()) >= 1
        assert int(extend_lens.max()) <= max_extend_len

    if prefix_lens is None:
        if max_prefix_len <= 0:
            prefix_lens = torch.zeros(num_seqs, dtype=torch.int32)
        else:
            prefix_lens = torch.randint(1, max_prefix_len + 1, (num_seqs,),
                                        generator=gen, dtype=torch.int32)
    else:
        if len(prefix_lens) == 1 and num_seqs > 1:   # a single value broadcasts to the whole batch
            prefix_lens = list(prefix_lens) * num_seqs
        prefix_lens = torch.tensor(prefix_lens, dtype=torch.int32)
        assert len(prefix_lens) == num_seqs and int(prefix_lens.min()) >= 0
        assert int(prefix_lens.max()) <= max_prefix_len

    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32),
                           extend_lens.cumsum(0, dtype=torch.int32)])
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32),
                           prefix_lens.cumsum(0, dtype=torch.int32)])
    total_extend = int(qo_indptr[-1])
    total_prefix = int(kv_indptr[-1])

    # The KV pool is larger than the actual usage: unreferenced rows are garbage
    # data, and the row ids are randomly permuted (indirect addressing)
    num_pool = total_prefix + 8
    kv_indices = torch.randperm(num_pool, generator=gen)[:total_prefix].to(torch.int32)

    q_extend = torch.randn(total_extend, num_q_heads, head_dim_q, generator=gen).to(dt)
    k_extend = torch.randn(total_extend, num_kv_heads, head_dim_q, generator=gen).to(dt)
    v_extend = torch.randn(total_extend, num_kv_heads, head_dim_v, generator=gen).to(dt)
    k_buffer = torch.randn(num_pool, num_kv_heads, head_dim_q, generator=gen).to(dt)
    v_buffer = torch.randn(num_pool, num_kv_heads, head_dim_v, generator=gen).to(dt)
    return (q_extend, k_extend, v_extend, k_buffer, v_buffer,
            qo_indptr, kv_indptr, kv_indices)
