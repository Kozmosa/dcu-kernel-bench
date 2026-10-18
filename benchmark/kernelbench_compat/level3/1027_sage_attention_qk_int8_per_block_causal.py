# 1027_sage_attention_qk_int8_per_block_causal -- SageAttention-style causal
# attention forward (KernelBench compatible task file).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=1027, backend=triton.

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """SageAttention2-style causal scaled dot-product attention forward:
    Q/K use per-block INT8 quantization, and P is multiplied against V in
    float16 (the dot within a block outputs float16, the accumulation across
    blocks is float32) in an online softmax implementation (HND layout, MHA/GQA
    in general).

    (1) Operator semantics (the acceptance ground truth): for batch b, query
    head h (its KV head is h // g, g = num_qo_heads // num_kv_heads) and query
    position i (write L = qo_len = kv_len):
      logits[i,j] = sm_scale * <q[b,h,i,:], k[b,h//g,j,:]>,
      sm_scale = 1/sqrt(head_dim)
      causal mask: positions j > i get -inf (query i only attends keys 0..i)
      p[i,:]  = softmax_j(logits[i, 0..i])
      out[b,h,i,:] = sum_{j<=i} p[i,j] * v[b,h//g,j,:]
    The softmax uses a numerically stable (online) implementation, intermediate
    computation is float32, and the output is cast back to the input dtype.
    Boundary behaviour: i = 0 only attends itself (then out[b,h,0,:] =
    v[b,h//g,0,:]); every query row always has a visible key, so the softmax
    denominator is always positive and there is no fully masked row.

    (2) Input/output specification:
      q   [batch, num_qo_heads, qo_len, head_dim]  float16/bfloat16
      k   [batch, num_kv_heads,  qo_len, head_dim] same dtype as q
      v   [batch, num_kv_heads,  qo_len, head_dim] same dtype as q
      out [batch, num_qo_heads, qo_len, head_dim]  same dtype as q
    Global-domain constraints: qo_len == kv_len (causal); num_qo_heads is an
    integer multiple of num_kv_heads (GQA, equality means MHA); head_dim in
    {64, 128}; HND layout, last dimension contiguous.

    (3) Implementation constraints (violations are scored as failures):
      - The core computation (INT8 quantization / QK^T / softmax / PV) must be
        completed inside the submitted file.
      - Implement it as the algorithm in the task statement: quantize Q per block
        of 128 rows and K per block of 64 rows to per-block INT8 -- the
        quantization of Q folds sm_scale*log2(e) into the quantized value, while
        the quantization of K folds no scale in; the per-block scale is
        max|x|/127, and int8 uses round-half-away-from-zero; qk =
        int8_dot(q,k) * (q_scale * k_scale) are the logits in the exp2 domain;
        the online softmax is updated per KV block in the exp2 domain; P is cast
        to float16 and dotted against the float16 V (the output within a block is
        float16), accumulating into a float32 acc; after the last block divide by
        the softmax denominator.
      - K may first have its mean removed along the sequence dimension (smooth-k)
        before quantization: this is mathematically neutral for the exact causal
        attention output (the logits of each query row are reduced by the same
        constant), it only improves the quantization accuracy and does not change
        the acceptance ground truth.
      - The acceptance tolerance atol=rtol=2e-2 already covers the approximation
        error of the quantization scheme above.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Use float32 for intermediate accumulation; the output is cast back to
        the input dtype.

    (4) Forbidden list: scaled_dot_product_attention / sdpa / flash_attn /
      torch.matmul / torch.bmm / torch.einsum / torch.softmax, as well as
      equivalent wrappers of any third-party operator library.
      Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
      Task scope: causal forward, HND layout, no attn_mask, no LSE output, no
      dropout; the non-causal branch and the NHD layout are not part of this
      task.
    """

    def __init__(self, head_dim: int, scale=None):
        super().__init__()
        self.sm_scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)

    def forward(self, q, k, v):
        # All three inputs of this task are floating-point tensors (HND layout);
        # there are no integer index/length inputs, so nothing needs recovering
        B, Hq, L, D = q.shape
        Bk, Hkv, Lk, Dk = k.shape
        assert B == Bk and L == Lk and D == Dk, "task constraint: qo_len == kv_len (causal)"
        group = Hq // Hkv
        assert Hq == Hkv * group, "num_qo_heads must be an integer multiple of num_kv_heads"

        causal_mask = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=q.device), diagonal=1
        )
        out = torch.empty_like(q)
        for b in range(B):
            qf = q[b].to(torch.float32) * self.sm_scale          # [Hq, L, D]
            kf = k[b].to(torch.float32)                           # [Hkv, L, D]
            vf = v[b].to(torch.float32)                           # [Hkv, L, D]
            # GQA: the K/V of one KV head are replicated to every query head in the group
            if group > 1:
                kf = kf.repeat_interleave(group, dim=0)           # [Hq, L, D]
                vf = vf.repeat_interleave(group, dim=0)           # [Hq, L, D]

            scores = torch.matmul(qf, kf.transpose(-2, -1))       # [Hq, L, L]
            scores = scores.masked_fill(causal_mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out[b] = torch.matmul(probs, vf).to(q.dtype)
        return out


def get_init_inputs():
    return [128]  # head_dim; sm_scale defaults to 1/sqrt(head_dim)


def get_inputs():
    # Fixed shape family (except qo_len); the random part consumes the global
    # RNG -- the evaluator calls this function after set_seed, so multiple
    # correctness trials therefore see diverse inputs
    batch = 2
    num_kv_heads = 4
    query_group_size = 4      # num_qo_heads = 16
    head_dim = 128
    max_qo_len = 512

    num_qo_heads = num_kv_heads * query_group_size
    qo_len = int(torch.randint(1, max_qo_len + 1, (1,)))

    q = torch.randn(batch, num_qo_heads, qo_len, head_dim).to(torch.float16)
    k = torch.randn(batch, num_kv_heads, qo_len, head_dim).to(torch.float16)
    v = torch.randn(batch, num_kv_heads, qo_len, head_dim).to(torch.float16)
    return [q, k, v]


def make_inputs(batch: int, num_qo_heads: int, num_kv_heads: int, qo_len: int,
                head_dim: int, dtype: str = "float16", seed: int = 0):
    """Generate inputs from a public/hidden/perf case description. Runs on the evaluation side only (generated on CPU; the evaluator moves them to the device).

    The causal task statement constrains qo_len == kv_len, so the kv tensors share
    the same qo_len as q; GQA requires num_qo_heads to be an integer multiple of
    num_kv_heads.
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert num_qo_heads % num_kv_heads == 0, "num_qo_heads must be an integer multiple of num_kv_heads"

    q = torch.randn(batch, num_qo_heads, qo_len, head_dim, generator=gen).to(dt)
    k = torch.randn(batch, num_kv_heads, qo_len, head_dim, generator=gen).to(dt)
    v = torch.randn(batch, num_kv_heads, qo_len, head_dim, generator=gen).to(dt)
    return q, k, v
