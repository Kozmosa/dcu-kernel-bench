# 1014_hstu_attention -- HSTU attention forward (jagged variable-length, no
# softmax, KernelBench compatible task file).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=1014, backend=triton.

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """HSTU attention forward (the attention of the generative-recommendation HSTU unit, no softmax / no normalization):
    dense Q/K/V are split by seq_offsets into batch variable-length sequences,
    and each sequence computes independently
    out = (valid * SiLU(alpha*Q*K^T) / N) * V, where N = max_seq_len.

    Mathematical definition (sequence b: s = seq_offsets[b],
    L = seq_offsets[b+1] - seq_offsets[b], g = num_targets[b],
    c = contextual_seq_len, A = max_attn_len; when c = 0, M = L and r[i] = i;
    when c > 0, M = L - c + 1 and r[i] = max(i - c + 1, 0)):
      position renumbering m[i] = min(r[i], M - g)
      d[i,j] = m[i] - m[j]; when causal = False, d[i,j] = abs(d[i,j])
      valid[i,j] = (i == j) or d[i,j] > 0
      when A > 0, valid[i,j] additionally requires d[i,j] <= A
      when c > 0, valid[i,j] is additionally relaxed to true when m[i] == 0 and
      m[j] < M - g
      out[s+i, h, :] = sum_j valid[i,j] * SiLU(alpha*<q[s+i,h,:], k[s+j,h,:]>) / N
                       * v[s+j,h,:]
    where SiLU(x) = x*sigmoid(x). When causal = True, valid is lower triangular
    (including the diagonal); the weights get no softmax and no sum
    normalization, and masked positions contribute exactly 0.
    Boundary behaviour: the last g tokens of a sequence are targets and are
    excluded from the key side by the renumbering truncation; g = 0 is equivalent
    to no target exclusion; when c > 0 the first c tokens (renumbered m = 0)
    additionally see all non-target keys (including each other); an empty
    sequence (L = 0) is legal and produces no output rows; a sequence may be
    shorter than N, and the scaling denominator is always N (not that sequence's
    own length).

    Input/output specification:
      q            [total, num_heads, qk_dim]   float16 / bfloat16
      k            [total, num_heads, qk_dim]   same dtype as q
      v            [total, num_heads, v_dim]    same dtype as q; v_dim may differ from qk_dim
      seq_offsets  [batch + 1] int64, monotonically non-decreasing, first element 0, last element total
      num_targets  [batch] int32, 0 <= num_targets[b] and c + num_targets[b] <= L_b
      out          [total, num_heads, v_dim], same dtype as v
    Global-domain constraints: 0 <= L_b <= max_seq_len; qk_dim / v_dim are powers
    of 2 (of order 64..256).

    Hyper-parameters (__init__): max_seq_len (scaling denominator and upper bound
    on length, N); alpha (logits scale, defaults to 1/sqrt(qk_dim)); causal
    (defaults to True); max_attn_len (defaults to 0 = distance not limited);
    contextual_seq_len (defaults to 0 = disabled).

    Implementation constraints (violations are scored as failures):
      - The core computation (QK^T / SiLU / mask / PV) must be done inside the
        submitted file; calling scaled_dot_product_attention / sdpa / flash_attn
        / torch.matmul / torch.bmm / torch.einsum / torch.softmax /
        torch.nn.functional.silu and their alias forms (F.silu / F.softmax /
        torch.mm / the @ matrix-multiply operator) is forbidden (implement SiLU
        yourself as x*sigmoid(x)).
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Use float32 for intermediate accumulation; cast the output back to v's
        dtype.

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: forward only, non-quantized inputs, no dropout, no
    backpropagation.
    """

    # Note: the __init__/forward signatures must stay single-line (the framework
    # loader extracts the signature from the first line)
    def __init__(self, max_seq_len: int, alpha=None, causal=True, max_attn_len=0, contextual_seq_len=0):
        super().__init__()
        self.max_seq_len = int(max_seq_len)
        self.alpha = float(alpha) if alpha is not None else None
        self.causal = bool(causal)
        self.max_attn_len = int(max_attn_len)
        self.contextual_seq_len = int(contextual_seq_len)

    def forward(self, q, k, v, seq_offsets, num_targets):
        # The KernelBench evaluator casts every input tensor to the evaluation
        # precision (fp32 for the triton backend); the offset table and the target
        # counts are small integers that fp32 represents exactly, so they are
        # recovered losslessly here
        seq_offsets = seq_offsets.to(torch.long)
        num_targets = num_targets.to(torch.int32)

        B = int(seq_offsets.numel()) - 1
        alpha = self.alpha if self.alpha is not None else 1.0 / math.sqrt(q.shape[2])
        N = self.max_seq_len
        c = self.contextual_seq_len

        out = torch.empty_like(v)
        for b in range(B):
            s, e = int(seq_offsets[b]), int(seq_offsets[b + 1])
            L = e - s
            g = int(num_targets[b])
            assert 0 <= g and c + g <= L, "task constraint: 0 <= g and contextual_seq_len + g <= L"

            # float32 intermediate computation; the [H, L, *] view makes per-head batched matmul easy
            q_b = q[s:e].transpose(0, 1).to(torch.float32)            # [H, L, D_qk]
            k_b = k[s:e].transpose(0, 1).to(torch.float32)
            v_b = v[s:e].transpose(0, 1).to(torch.float32)            # [H, L, D_v]

            logits = torch.matmul(q_b, k_b.transpose(-2, -1)) * alpha  # [H, L, L]
            w = logits * torch.sigmoid(logits) / N                     # SiLU(logits)/N

            # valid mask: built from the renumbered relative distance
            ids = torch.arange(L, device=q.device)
            M = L if c == 0 else L - c + 1
            if c > 0:
                ids = torch.clamp(ids - (c - 1), min=0)
            ids = torch.clamp(ids, max=M - g)
            dist = ids[:, None] - ids[None, :]
            if not self.causal:
                dist = torch.where(dist > 0, dist, -dist)
            valid = torch.eye(L, dtype=torch.bool, device=q.device) | (dist > 0)
            if self.max_attn_len > 0:
                valid = valid & (dist <= self.max_attn_len)
            if c > 0:
                valid = valid | ((ids[:, None] == 0) & (ids[None, :] < M - g))

            o_b = torch.matmul(w * valid[None].to(w.dtype), v_b)       # [H, L, D_v]
            out[s:e] = o_b.transpose(0, 1).to(v.dtype)
        return out


def get_init_inputs():
    return [256]  # max_seq_len; alpha defaults to 1/sqrt(qk_dim), causal=True,
    # max_attn_len=0, contextual_seq_len=0


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the evaluator
    # calls this function after set_seed, so multiple correctness trials therefore
    # see diverse inputs
    num_seqs = 8
    num_heads = 4
    qk_dim = 128
    v_dim = 128
    max_seq_len = 256

    lengths = torch.randint(1, max_seq_len + 1, (num_seqs,), dtype=torch.int64)
    num_targets = torch.minimum(
        torch.randint(1, 21, (num_seqs,), dtype=torch.int64), lengths
    ).to(torch.int32)

    seq_offsets = torch.zeros(num_seqs + 1, dtype=torch.int64)
    for i in range(num_seqs):
        seq_offsets[i + 1] = seq_offsets[i] + int(lengths[i])

    total = int(lengths.sum())
    q = torch.randn(total, num_heads, qk_dim).to(torch.bfloat16)
    k = torch.randn(total, num_heads, qk_dim).to(torch.bfloat16)
    v = torch.randn(total, num_heads, v_dim).to(torch.bfloat16)
    return [q, k, v, seq_offsets, num_targets]


def make_inputs(batch_size: int, max_seq_len: int, num_heads: int,
                qk_dim: int, v_dim: int, dtype: str = "bfloat16", seed: int = 0,
                lengths=None, sparsity=None, num_targets=None,
                target_size: int = 20, causal: bool = True, max_attn_len: int = 0,
                contextual_seq_len: int = 0, alpha=None):
    """Generate inputs from a public/hidden/perf case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device). Returns (q, k, v, seq_offsets, num_targets). causal /
    max_attn_len / contextual_seq_len / alpha do not take part in tensor
    generation, they only describe the corresponding Model hyper-parameters (the
    evaluation side builds the Model from them), of which contextual_seq_len
    enters the legality constraints on the lengths and the targets. lengths /
    num_targets may be given explicitly (a single value broadcasts to the whole
    batch); by default lengths ~ U[1, max_seq_len] and
    num_targets ~ U[1, target_size] truncated to the legal domain; sparsity gives
    the sparsity sampling (official convention): 0 -> all lengths 0,
    1 -> all max_seq_len, >= 0.5 -> U[(2s-1)*N, N), < 0.5 -> U[0, 2s*N).
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)

    if lengths is None:
        if sparsity is None:
            lo, hi = 1, max_seq_len + 1
        elif sparsity == 0.0:
            lo, hi = 0, 1
        elif sparsity == 1.0:
            lo, hi = max_seq_len, max_seq_len + 1
        elif sparsity >= 0.5:
            lo, hi = int((2 * sparsity - 1.0) * max_seq_len), max_seq_len
        else:
            lo, hi = 0, int(2 * sparsity * max_seq_len)
        if contextual_seq_len > 0:
            lo = max(lo, contextual_seq_len + 1)
            hi = max(hi, lo + 1)
        lengths = torch.randint(lo, hi, (batch_size,), generator=gen, dtype=torch.int64)
    else:
        if len(lengths) == 1 and batch_size > 1:      # a single value broadcasts to the whole batch
            lengths = lengths * batch_size
        lengths = torch.tensor(lengths, dtype=torch.int64)
    assert lengths.numel() == batch_size
    assert bool((lengths >= 0).all()) and int(lengths.max()) <= max_seq_len
    if contextual_seq_len > 0:
        assert bool((lengths >= contextual_seq_len).all()), "task constraint: c + g <= L"

    cap = torch.clamp(lengths - contextual_seq_len, min=0)
    if num_targets is None:
        nt = torch.randint(1, target_size + 1, (batch_size,), generator=gen, dtype=torch.int64)
        nt = torch.minimum(nt, cap)
    else:
        if len(num_targets) == 1 and batch_size > 1:
            num_targets = num_targets * batch_size
        nt = torch.tensor(num_targets, dtype=torch.int64)
    assert nt.numel() == batch_size
    assert bool((nt >= 0).all()) and bool((nt <= cap).all()), "task constraint: 0 <= g <= L - c"

    seq_offsets = torch.zeros(batch_size + 1, dtype=torch.int64)
    for i in range(batch_size):
        seq_offsets[i + 1] = seq_offsets[i] + int(lengths[i])

    total = int(lengths.sum())
    q = torch.randn(total, num_heads, qk_dim, generator=gen).to(dt)
    k = torch.randn(total, num_heads, qk_dim, generator=gen).to(dt)
    v = torch.randn(total, num_heads, v_dim, generator=gen).to(dt)
    return q, k, v, seq_offsets, nt.to(torch.int32)
