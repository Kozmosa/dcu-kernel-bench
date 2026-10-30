# 1026_sage_attention_qk_int8_per_block -- SageAttention-style non-causal forward
# (Q/K per-block int8 codewords + per-block scale, PV fp16), a model_class
# (KernelBench compatible) task file.
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=1026, backend=triton.

import torch
import torch.nn as nn

# log2(e): the quantization stage folds the softmax scale and log2(e) into
# q_scale together, and the kernel side runs the online softmax in base 2;
# converting the dequantized result back to the natural-log domain requires
# dividing by this constant
_LOG2E = 1.4426950408889634


def _quant_per_block_int8(x, blk):
    """Quantize [b, h, L, d] to int8 symmetric form with absmax/127 per block of blk rows along L.

    Replicates the numeric behaviour of the source quantization kernel: the
    absmax over all block elements is the scale (fp32), then x/scale is offset by
    0.5*sign and truncated toward zero (round-half-away-from-zero). Returns
    (int8 codewords [b,h,L,d], scale [b,h,ceil(L/blk)] fp32).
    """
    b, h, length, d = x.shape
    n_blk = (length + blk - 1) // blk
    pad = n_blk * blk - length
    if pad:
        x = nn.functional.pad(x, (0, 0, 0, pad))
    blocks = x.reshape(b, h, n_blk, blk * d)
    scale = blocks.abs().amax(dim=-1) / 127.0
    t = blocks / scale[..., None]
    t = torch.trunc(t + 0.5 * torch.where(t >= 0, 1.0, -1.0))
    q = t.to(torch.int8).reshape(b, h, n_blk * blk, d)
    if pad:
        q = q[:, :, :length, :]
    return q.contiguous(), scale.contiguous()


def _sage_quantize(q, k, v, sm_scale=None):
    """Turn fp16 q/k/v in HND layout into the inputs of this task's forward (quantization-stage emulation).

    smooth_k is on by default: k first has its mean along the sequence dimension
    subtracted (every row of logits is reduced by the same q dot km, so the
    softmax semantics are unchanged); the quantization of q folds
    sm_scale*log2(e) into the scale, while the quantization of k folds no scale
    in. Q is blocked in 128 rows and K in 64 rows.
    """
    head_dim = q.shape[-1]
    if sm_scale is None:
        sm_scale = head_dim ** -0.5
    km = k.float().mean(dim=2, keepdim=True)          # [b, h_kv, 1, d]
    k_smooth = k.float() - km
    q_int8, q_scale = _quant_per_block_int8(q.float() * (sm_scale * _LOG2E), 128)
    k_int8, k_scale = _quant_per_block_int8(k_smooth, 64)
    return q_int8, k_int8, v, q_scale, k_scale


class Model(nn.Module):
    """SageAttention-style non-causal scaled dot-product attention forward:
    Q/K are already quantized per block to int8 symmetric form (per-block
    absmax/127 scale), while V stays fp16.

    Mathematical definition (g = num_q_heads // num_kv_heads, query head h
    corresponds to KV head h//g; quantization blocking: Q in blocks of 128 rows
    along the sequence direction and K in blocks of 64 rows, with block indices
    i//128 and j//64 respectively, i.e. the index along the last dimension of
    q_scale/k_scale):
      dq[b,h,i,:]    = q_int8[b,h,i,:] * q_scale[b,h,i//128]    (float32 dequant)
      dk[b,hk,j,:]   = k_int8[b,hk,j,:] * k_scale[b,hk,j//64]
      logit[b,h,i,j] = <dq[b,h,i,:], dk[b,h//g,j,:]> / log2(e)
      (divide by log2(e): the quantization stage already folded the softmax scale
         and log2(e) into q_scale, and the softmax
         in base 2, so everything is converted back to the natural-log domain
         here)
      positions where attn_mask[i,j] is False get logit = -inf (bool mask,
        shape [qo_len, kv_len], shared by all batches and heads)
      p[b,h,i,:]   = softmax_j(logit[b,h,i,:])
      out[b,h,i,:] = sum_j p[b,h,i,j] * v[b,h//g,j,:]
    The softmax follows the semantics of a numerically stable online
    implementation; the reference implementation works entirely in float32 (the
    dot product of int8 codewords is exact integer arithmetic in float32), and
    the output is cast back to the dtype of v.
    Numeric scheme (used by the source operator, a submission may reuse it): cast
    p to fp16 and do the matrix multiply against v in fp16, accumulating the
    result of each KV shard into a float32 buffer ("PV fp16 + fp32 shard
    accumulation"), and do the int8 QK^T with a matrix multiply of int8 inputs
    accumulating directly in integers -- whichever numeric scheme is used, it
    must match this reference implementation within the tolerance.
    Boundary behaviour: qo_len and kv_len may differ and need not be integer
    multiples of the block sizes (the tail block takes part in the
    dequantization and the softmax as usual); the task guarantees that every
    query row has at least one unmasked position, so the softmax denominator is
    always positive.

    Input/output specification (tensor_layout="HND"/"NHD" is given by __init__):
      q_int8   int8     HND [batch, num_q_heads, qo_len, head_dim]
                         NHD [batch, qo_len, num_q_heads, head_dim]
      k_int8   int8     HND [batch, num_kv_heads, kv_len, head_dim]
                         NHD [batch, kv_len, num_kv_heads, head_dim]
      v        float16  same layout and same number of KV heads as k_int8
      q_scale  float32  [batch, num_q_heads, ceil(qo_len/128)] (layout independent)
      k_scale  float32  [batch, num_kv_heads, ceil(kv_len/64)] (layout independent)
      attn_mask bool    [qo_len, kv_len] (layout independent)
      out      same shape as q_int8, dtype same as v (single-tensor output)
    Global-domain constraints: num_q_heads is an integer multiple of num_kv_heads
    (GQA; equality means MHA); head_dim in {64, 128}; qo_len, kv_len >= 1.

    Implementation constraints (violations are scored as failures):
      - The core computation (int8 QK^T / dequantization / softmax / PV) must be
        completed inside the submitted file; calling
        scaled_dot_product_attention / sdpa / flash_attn / torch.matmul /
        torch.bmm / torch.mm / torch.addmm / F.linear / torch.einsum /
        torch.softmax is forbidden.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Use float32 for intermediate accumulation; the output is cast back to
        the input dtype.
      - The evaluator casts every input to fp32 before calling forward: the int8
        codewords, the fp16 v and the bool mask are all exactly representable in
        fp32, so the original dtypes are recovered losslessly at the entry.

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: non-causal forward, bool mask, GQA, both the HND and NHD layouts;
    does not include the causal variant, float additive masks, LSE output, or the
    quantization stage itself (the int8 codewords are given directly as inputs).
    """

    def __init__(self, tensor_layout="HND"):
        super().__init__()
        assert tensor_layout in ("HND", "NHD"), "tensor_layout only supports HND / NHD"
        self.tensor_layout = tensor_layout

    def forward(self, q_int8, k_int8, v, q_scale, k_scale, attn_mask):
        # Note: forward uses single-line statements only (conditions are written
        # as conditional expressions), so that the loader extracting the
        # statements one by one into the ModelNew scaffold does not break
        # indentation
        q_int8 = q_int8.to(torch.int8)
        k_int8 = k_int8.to(torch.int8)
        v = v.to(torch.float16)
        q_scale = q_scale.to(torch.float32)
        k_scale = k_scale.to(torch.float32)
        attn_mask = attn_mask.to(torch.bool)
        nhd = self.tensor_layout == "NHD"
        # NHD [b,l,h,d] is uniformly transposed into the HND [b,h,l,d] view for the computation
        q = (q_int8.permute(0, 2, 1, 3) if nhd else q_int8).to(torch.float32)
        k = (k_int8.permute(0, 2, 1, 3) if nhd else k_int8).to(torch.float32)
        vv = (v.permute(0, 2, 1, 3) if nhd else v).to(torch.float32)
        b, h_q, qo_len, head_dim = q.shape
        h_kv, kv_len = k.shape[1], k.shape[2]
        group = h_q // h_kv
        assert h_q == h_kv * group, "num_q_heads must be an integer multiple of num_kv_heads"
        # GQA: the K/V (and k_scale) of one KV head are replicated to every
        # query head in the group
        k = k.repeat_interleave(group, dim=1)
        vv = vv.repeat_interleave(group, dim=1)
        k_scale = k_scale.repeat_interleave(group, dim=1)
        # Expand the per-block scale to the sequence dimension: Q blocks of 128
        # rows, K blocks of 64 rows
        qs = q_scale.repeat_interleave(128, dim=-1)[..., :qo_len]   # [b,h_q,qo_len]
        ks = k_scale.repeat_interleave(64, dim=-1)[..., :kv_len]    # [b,h_q,kv_len]
        # int8 codeword dot product (exact integer arithmetic under fp32), then
        # dequantize per block and divide by log2(e) to get back to the
        # natural-log domain (the constant is inlined so the scaffold stays
        # self-contained)
        logits = torch.matmul(q, k.transpose(-2, -1))
        logits = logits * qs[:, :, :, None] * ks[:, :, None, :] / 1.4426950408889634
        logits = logits.masked_fill(~attn_mask, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        out = torch.matmul(probs, vv)
        out = out.permute(0, 2, 1, 3) if nhd else out
        return out.contiguous().to(v.dtype)


def get_init_inputs():
    return ["HND"]   # tensor_layout


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the
    # evaluator calls this function after set_seed, so multiple correctness
    # trials therefore see diverse inputs
    batch = 2
    num_kv_heads = 4
    query_group_size = 2      # num_q_heads = 8 (GQA)
    qo_len = 256
    kv_len = 512
    head_dim = 64

    num_q_heads = num_kv_heads * query_group_size
    q = torch.randn(batch, num_q_heads, qo_len, head_dim).to(torch.float16)
    k = torch.randn(batch, num_kv_heads, kv_len, head_dim).to(torch.float16)
    v = torch.randn(batch, num_kv_heads, kv_len, head_dim).to(torch.float16)
    q_int8, k_int8, v, q_scale, k_scale = _sage_quantize(q, k, v)

    # bool mask: randomly drop about 15% of the positions, column 0 is always on,
    # guaranteeing at least one unmasked position per row
    attn_mask = torch.rand(qo_len, kv_len) > 0.15
    attn_mask[:, 0] = True
    return [q_int8, k_int8, v, q_scale, k_scale, attn_mask]


def make_inputs(batch: int, num_q_heads: int, num_kv_heads: int, qo_len: int,
                kv_len: int, head_dim: int, tensor_layout: str = "HND",
                mask_mode: str = "all_true", dtype: str = "float16",
                seed: int = 0):
    """Deterministic case generator (CPU, torch.Generator initialized from seed).

    fp16 q/k/v are generated in the HND view first, go through smooth_k and
    per-block int8 quantization (Q blocks of 128 rows, K blocks of 64 rows,
    sm_scale folded into q_scale), and are then transposed to the NHD layout when
    required. mask_mode: "all_true" leaves everything open; "random" randomly
    masks about 15% of the positions and forces column 0 to True. The fields map
    one to one onto the hidden/perf cases.
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)

    group = num_q_heads // num_kv_heads
    assert num_q_heads == num_kv_heads * group, "num_q_heads must be an integer multiple of num_kv_heads"
    assert head_dim in (64, 128), "head_dim only supports 64 / 128"
    assert tensor_layout in ("HND", "NHD"), "tensor_layout only supports HND / NHD"

    q = torch.randn(batch, num_q_heads, qo_len, head_dim, generator=gen).to(dt)
    k = torch.randn(batch, num_kv_heads, kv_len, head_dim, generator=gen).to(dt)
    v = torch.randn(batch, num_kv_heads, kv_len, head_dim, generator=gen).to(dt)
    q_int8, k_int8, v, q_scale, k_scale = _sage_quantize(q, k, v)

    if mask_mode == "random":
        attn_mask = torch.rand(qo_len, kv_len, generator=gen) > 0.15
        attn_mask[:, 0] = True
    else:
        attn_mask = torch.ones(qo_len, kv_len, dtype=torch.bool)

    if tensor_layout == "NHD":
        q_int8 = q_int8.permute(0, 2, 1, 3).contiguous()
        k_int8 = k_int8.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
    return [q_int8, k_int8, v, q_scale, k_scale, attn_mask]
