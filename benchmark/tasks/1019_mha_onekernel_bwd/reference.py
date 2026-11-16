# 1019_mha_onekernel_bwd -- FlashAttention backward, one-kernel form (KernelBench
# compatible task file).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=1019, backend=triton.

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """FlashAttention backward (one-kernel form): given the forward output o and the
    per-row log-sum-exp statistics lse, compute the gradients (dq, dk, dv) from
    (q, k, v, o, lse, do); dK/dV and dQ are produced inside the same kernel
    (dK/dV first, dQ after).

    Mathematical definition (let g = num_q_heads // num_kv_heads and the scaled
    score be s[b,h,i,j] = sm_scale * <q[b,i,h,:], k[b,j,h//g,:]>,
    with sm_scale defaulting to 1/sqrt(head_dim)):
      keep[b,i,j] = (j <= i + seqlen_k - seqlen_q) when causal=1 (bottom-right
                    aligned), always true when causal=0
      p[b,h,i,j]   = exp(s[b,h,i,j] - lse[b,h,i]), and masked positions (keep=0) are always 0
      delta[b,h,i] = sum_d o[b,i,h,d] * do[b,i,h,d]   (computed from the given o)
      dp[b,h,i,j]  = <do[b,i,h,:], v[b,j,h//g,:]>
      ds[b,h,i,j]  = p[b,h,i,j] * (dp[b,h,i,j] - delta[b,h,i])
      dv[b,j,h_k,d] = sum_{h = h_k*g .. h_k*g+g-1} sum_i p[b,h,i,j] * do[b,i,h,d]
      dk[b,j,h_k,d] = sm_scale * sum_{h in the same group} sum_i ds[b,h,i,j] * q[b,i,h,d]
      dq[b,i,h,d]   = sm_scale * sum_j ds[b,h,i,j] * k[b,j,h//g,d]
    Note that dK/dQ are multiplied by sm_scale at the end while dV is not; delta
    uses the input o (not the ideal forward output).
    Boundary behaviour: when causal=1 and i < seqlen_q - seqlen_k the row is fully
    masked -- p/ds are always 0, the dq row is all 0, and it contributes nothing to
    dk/dv (the task input convention sets o=0 and lse=0 for those rows).

    Input/output specification:
      q, o, do  [batch, seqlen_q, num_q_heads, head_dim] float16/bfloat16
      k, v      [batch, seqlen_k, num_kv_heads, head_dim] same dtype as q
      lse       [batch, num_q_heads, seqlen_q] float32 -- natural-log log-sum-exp of
                the scaled scores (0 for fully masked rows)
      returns a single 1-D tensor packed (multi-output packing protocol):
      packed = cat([dq flattened, dk flattened, dv flattened]) (each segment is
      first cast back to the input dtype, then converted to float32)
      where dq has the same shape as q and dk/dv the same shape as k; the total
      length of packed is
      (batch*seqlen_q*num_q_heads + 2*batch*seqlen_k*num_kv_heads) * head_dim.
    Global-domain constraints: num_q_heads is an integer multiple of num_kv_heads
      (GQA; equal means MHA); 1 <= head_dim <= 256 (need not be a power of 2);
      1 <= seqlen_q, seqlen_k; o and lse are the self-consistent forward statistics
      of (q, k, v) (guaranteed by the task's input generator).

    Implementation constraints (violations are scored as failures):
      - The core computation (all matrix multiplies and elementwise operations of
        delta / p / ds / dV / dK / dQ) must be done inside the submitted file;
        calling scaled_dot_product_attention / sdpa / flash_attn / torch.matmul /
        torch.bmm / torch.einsum / torch.softmax is forbidden, and so are
        automatic-differentiation shortcuts such as torch.autograd /
        autograd.grad / .backward(.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Use float32 for intermediate accumulation; cast the output back to the
        input dtype.
      - forward must return a single 1-D float32 tensor following the packing
        protocol above (compute dq/dk/dv internally first, then flatten and
        concatenate each).

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: dense fixed-length (non-varlen), non-quantized inputs, no dropout /
    ALiBi / bias / fp8; causal and non-causal are selected by __init__'s causal
    switch.
    """

    def __init__(self, head_dim: int, causal: int, scale=None):
        super().__init__()
        self.head_dim = int(head_dim)
        self.causal = bool(int(causal))
        self.sm_scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)

    def forward(self, q, k, v, o, lse, do):
        # This forward has no integer inputs (no index/table/length tensors), so no lossless recovery is needed
        B, M, H_Q, D = q.shape
        _, N, H_K, _ = k.shape
        group = H_Q // H_K
        assert H_Q == H_K * group, "num_q_heads must be an integer multiple of num_kv_heads"
        causal = bool(self.causal)

        dq = torch.empty_like(q)
        dk = torch.zeros(B, N, H_K, D, dtype=torch.float32, device=q.device)
        dv = torch.zeros_like(dk)

        for b in range(B):
            for h in range(H_Q):
                hk = h // group    # GQA: every query head in the group shares one KV head
                qh = q[b, :, h, :].to(torch.float32)    # [M, D]
                kh = k[b, :, hk, :].to(torch.float32)   # [N, D]
                vh = v[b, :, hk, :].to(torch.float32)
                doh = do[b, :, h, :].to(torch.float32)
                oh = o[b, :, h, :].to(torch.float32)

                scores = torch.matmul(qh, kh.transpose(-2, -1)) * self.sm_scale
                if causal:
                    blocked = torch.triu(
                        torch.ones(M, N, dtype=torch.bool, device=q.device),
                        diagonal=N - M + 1,
                    )
                    scores = scores.masked_fill(blocked, float("-inf"))

                p = torch.exp(scores - lse[b, h].to(torch.float32).unsqueeze(-1))
                delta = torch.sum(oh * doh, dim=-1)                      # [M]
                dp = torch.matmul(doh, vh.transpose(-2, -1))             # [M, N]
                ds = p * (dp - delta.unsqueeze(-1))

                dv[b, :, hk, :] += torch.matmul(p.transpose(-2, -1), doh)
                dk[b, :, hk, :] += (
                    torch.matmul(ds.transpose(-2, -1), qh) * self.sm_scale
                )
                dq[b, :, h, :] = torch.matmul(ds, kh) * self.sm_scale

        # multi-output packing protocol: a single 1-D concatenated tensor (the
        # evaluator only supports single-tensor outputs).
        # Each gradient is first cast back to the input dtype (following the output
        # dtype convention), then converted to float32 for the concatenation
        return torch.cat([
            dq.reshape(-1).to(k.dtype).to(torch.float32),
            dk.reshape(-1).to(k.dtype).to(torch.float32),
            dv.reshape(-1).to(v.dtype).to(torch.float32),
        ])


def _forward_stats(q, k, v, sm_scale, causal):
    """Compute the forward statistics from (q, k, v): the output o (input dtype) and
    lse (float32, natural-log log-sum-exp; o=0 and lse=0 on causal fully masked
    rows). Used by the input generator only.
    """
    B, M, H_Q, D = q.shape
    _, N, H_K, _ = k.shape
    group = H_Q // H_K

    o = torch.empty_like(q)
    lse = torch.zeros(B, H_Q, M, dtype=torch.float32)
    for b in range(B):
        for h in range(H_Q):
            qh = q[b, :, h, :].to(torch.float32)
            kh = k[b, :, h // group, :].to(torch.float32)
            vh = v[b, :, h // group, :].to(torch.float32)

            scores = torch.matmul(qh, kh.transpose(-2, -1)) * sm_scale
            if causal:
                blocked = torch.triu(
                    torch.ones(M, N, dtype=torch.bool), diagonal=N - M + 1
                )
                scores = scores.masked_fill(blocked, float("-inf"))

            lse_h = torch.logsumexp(scores, dim=-1)
            if causal and M > N:
                # Fully masked rows (i < M - N): lse is set to 0 (consistent with
                # the FlashAttention forward convention), after which
                # p=exp(-inf-0)=0 and the o rows are 0
                lse_h = torch.where(
                    torch.arange(M) < M - N, torch.zeros_like(lse_h), lse_h
                )
            p = torch.exp(scores - lse_h.unsqueeze(-1))
            o[b, :, h, :] = torch.matmul(p, vh).to(q.dtype)
            lse[b, h] = lse_h
    return o, lse


def get_init_inputs():
    return [64, 1]  # head_dim, causal; sm_scale defaults to 1/sqrt(head_dim)


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the evaluator
    # calls this function after set_seed, so multiple correctness trials therefore
    # see diverse inputs (causal is 1, consistent with get_init_inputs)
    batch = 2
    num_kv_heads = 2
    query_group_size = 4      # num_q_heads = 8
    head_dim = 64
    seqlen_q = 128
    seqlen_k = 128

    num_q_heads = num_kv_heads * query_group_size
    q = torch.randn(batch, seqlen_q, num_q_heads, head_dim).to(torch.float16)
    k = torch.randn(batch, seqlen_k, num_kv_heads, head_dim).to(torch.float16)
    v = torch.randn(batch, seqlen_k, num_kv_heads, head_dim).to(torch.float16)
    do = torch.randn(batch, seqlen_q, num_q_heads, head_dim).to(torch.float16)
    o, lse = _forward_stats(q, k, v, head_dim ** -0.5, causal=True)
    return [q, k, v, o, lse, do]


def make_inputs(batch: int, num_q_heads: int, num_kv_heads: int, head_dim: int,
                seqlen_q: int, seqlen_k: int, dtype: str = "float16",
                causal: int = 1, seed: int = 0):
    """Generate inputs from a public/hidden/perf case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device). o and lse are computed from (q, k, v) with an exact float32
    forward (o=0 and lse=0 on fully masked rows), which guarantees consistency
    with the task semantics."""
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert num_q_heads % num_kv_heads == 0, "num_q_heads must be an integer multiple of num_kv_heads"

    q = torch.randn(batch, seqlen_q, num_q_heads, head_dim, generator=gen).to(dt)
    k = torch.randn(batch, seqlen_k, num_kv_heads, head_dim, generator=gen).to(dt)
    v = torch.randn(batch, seqlen_k, num_kv_heads, head_dim, generator=gen).to(dt)
    do = torch.randn(batch, seqlen_q, num_q_heads, head_dim, generator=gen).to(dt)
    o, lse = _forward_stats(q, k, v, head_dim ** -0.5, causal=bool(causal))
    return q, k, v, o, lse, do
