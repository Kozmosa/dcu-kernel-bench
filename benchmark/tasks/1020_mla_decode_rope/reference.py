# 1020_mla_decode_rope -- DeepSeek MLA decode attention with fused RoPE
# (KernelBench compatible task file).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=1020, backend=triton.

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """DeepSeek MLA decode-phase attention with a single-token query, fusing the RoPE
    rotation, page size = 1.

    Semantics: q is the current decode step's query [B, H, c + r] (c =
    kv_lora_rank compressed nope part first, r = qk_rope_head_dim rope part q_pe
    after it). The KV cache is stored per token (page size 1): kv_cache has
    total_tokens rows, each row = [compressed latent vector (first c dims) | k_pe
    (last r dims)]; the j-th logical token of batch b lives at physical row
    kv_indices[kv_indptr[b] + j] (kv_indptr is the cumulative sum of the sequence
    lengths, starting at 0); unreferenced rows of the row pool are garbage data and
    must not take part in the computation. Value is exactly the first c columns of
    that row: the compressed part of K shares the same latent vector with V (the
    MLA structure -- every query head shares one per-token compressed KV, there is
    no separate KV head).

    For every batch b (let pos = positions[b], rd = rotary_dim, half = rd // 2):
      1. RoPE rotation: take the row [cos(half) | sin(half)] of cos_sin_cache by
         position pos. The first rd dims of q_pe are rotated and the dims after rd
         pass through unchanged; the k_pe of the last logical token
         (j = seq_len_b - 1) is likewise rotated only in its first rd dims (dims
         after rd pass through), while the k_pe of the other tokens is treated as
         already rotated and used as the raw cache value. The rotation style is
         chosen by is_neox_style:
           NEOX (rotate-half): (x1, x2) = (first half dims, last half dims)
           GPT-J (interleaved): (x1, x2) = (even dims, odd dims)
           rotation output = concat(x1*cos - x2*sin, x2*cos + x1*sin) (scattered
           back in their own order).
      2. Attention (softmax over all j = 0..seq_len_b-1):
           s[h, j] = sm_scale * ( <q_nope[b,h,:], k_lat[b,j,:]>
                                + <q_pe'[b,h,:], k_pe'[b,j,:]> )
         where ' denotes after rotation; k_pe' is the rotated value only for the
         last token and the raw cache value for the others.
           p[h, :] = softmax_j(s[h, :])
           attn_out[b, h, :] = sum_j p[h, j] * k_lat[b, j, :]
      3. k_pe_tokens[b, :] = the full rotated k_pe of the last token (r dims, for
         writing back to the cache).
    softmax is numerically stable, intermediate accumulation is float32, and the
    output is cast back to the input dtype.
    Boundary behaviour: seq_len_b >= 1 (when it equals 1 that token is "the last
    token" and its k_pe takes part in the rotation); how the KV reduction is
    organized is only an implementation detail -- any implementation numerically
    equivalent to the full softmax is acceptable; there is no logit cap, no causal
    mask (decode naturally attends the whole cache), and no quantization.

    Input/output specification:
      q              [B, H, c + r] float16/bfloat16
      kv_cache       [total_tokens, c + r] same dtype as q (token-level KV row pool)
      cos_sin_cache  [max_positions, rd] same dtype as q (first half columns cos,
                     last half columns sin; row t corresponds to position
                     t = 0,1,...)
      positions      [B] int32 (sequence position of the current query; values
                     < max_positions, independent of seq_len)
      kv_indptr      [B+1] int32 (cumulative sum of the sequence lengths: first
                     element 0, monotonically non-decreasing)
      kv_indices     [total_kv] int32 (logical token -> physical row id; values
                     < total_tokens; total_kv == kv_indptr[-1])
      out            1-D flat concatenation, length B*H*c + B*r, same dtype as q:
                     the first B*H*c entries = attn_out ([B, H, c] flattened in
                     row-major order), the last B*r entries = k_pe_tokens
                     ([B, r] flattened in row-major order)
    Global-domain constraints: rd is a power of 2 and rd <= r;
    total_tokens >= kv_indptr[-1]; each sequence seq_len_b >= 1; B, H, c, r,
    max_positions are given by the shape family.

    Implementation constraints (violations are scored as failures):
      - The core computation (RoPE rotation / KV gathering / QK^T / softmax / PV)
        must be done inside the submitted file; calling
        scaled_dot_product_attention / sdpa / flash_attn / aiter / torch.matmul /
        torch.bmm / torch.einsum / torch.softmax is forbidden.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Use float32 for intermediate accumulation; cast the output back to the
        input dtype.
      - The 1-D concatenation layout of the output must match the specification
        above (the evaluator compares element by element).

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: single-token decode, use_rope always true, non-quantized
    fp16/bf16, no logit cap, no causal mask / ALiBi / sliding window.
    """

    def __init__(self, kv_lora_rank: int, qk_rope_head_dim: int, rotary_dim: int, sm_scale=None, is_neox_style=True):
        super().__init__()
        self.kv_lora_rank = int(kv_lora_rank)
        self.qk_rope_head_dim = int(qk_rope_head_dim)
        self.rotary_dim = int(rotary_dim)
        self.sm_scale = float(sm_scale) if sm_scale is not None else 1.0 / math.sqrt(self.kv_lora_rank + self.qk_rope_head_dim)
        self.is_neox_style = bool(is_neox_style)

    def forward(self, q, kv_cache, cos_sin_cache, positions, kv_indptr, kv_indices):
        # The KernelBench evaluator casts every input tensor to the evaluation
        # precision (fp32 for the triton backend); row ids / cumulative lengths /
        # positions are small integers that fp32 represents exactly, so they are
        # recovered losslessly here
        positions = positions.to(torch.long).reshape(-1)
        kv_indptr = kv_indptr.to(torch.long)
        kv_indices = kv_indices.to(torch.long)

        c, rd = self.kv_lora_rank, self.rotary_dim
        half = rd // 2
        B, H, D = q.shape
        r = D - c
        assert r == self.qk_rope_head_dim, "q's last dim must equal kv_lora_rank + qk_rope_head_dim"

        def apply_rope(x, cos, sin):
            # rotate the last dim (width r) of the float32 tensor x in place: the first rd dims are rotated, the rest pass through
            rot = x[..., :rd]
            if self.is_neox_style:
                x1, x2 = rot[..., :half], rot[..., half:]
            else:
                x1, x2 = rot[..., ::2], rot[..., 1::2]
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            if self.is_neox_style:
                rot_out = torch.cat((o1, o2), dim=-1)
            else:
                rot_out = torch.stack((o1, o2), dim=-1).flatten(-2)
            return torch.cat((rot_out, x[..., rd:]), dim=-1)

        attn_out = torch.empty(B, H, c, dtype=q.dtype, device=q.device)
        k_pe_tokens = torch.empty(B, r, dtype=q.dtype, device=q.device)
        for b in range(B):
            s0, s1 = int(kv_indptr[b]), int(kv_indptr[b + 1])
            toks = kv_indices[s0:s1]                                 # physical rows of this batch's tokens
            kv_b = kv_cache.index_select(0, toks).to(torch.float32)  # [S, c+r] (a copy)
            k_lat = kv_b[:, :c]                                      # compressed latent vector (shared by K/V)
            k_pe = kv_b[:, c:]

            pos = int(positions[b])
            cos = cos_sin_cache[pos, :half].to(torch.float32)
            sin = cos_sin_cache[pos, half:].to(torch.float32)

            q_b = q[b].to(torch.float32)                             # [H, c+r]
            q_pe = apply_rope(q_b[:, c:], cos, sin)                  # q_pe rotation
            k_pe[-1] = apply_rope(k_pe[-1], cos, sin)                # only the last token's k_pe is rotated
            k_pe_tokens[b] = k_pe[-1].to(q.dtype)

            qk = torch.matmul(q_b[:, :c], k_lat.transpose(0, 1))     # inner product of the nope part
            qk += torch.matmul(q_pe, k_pe.transpose(0, 1))           # inner product of the rope part
            qk *= self.sm_scale
            probs = torch.softmax(qk, dim=-1)                        # [H, S]
            attn_out[b] = torch.matmul(probs, k_lat).to(q.dtype)
        # single-tensor output: attn_out[B,H,c] || k_pe_tokens[B,r] (row-major flat concatenation)
        return torch.cat((attn_out.reshape(-1), k_pe_tokens.reshape(-1)))


def get_init_inputs():
    return [512, 64, 64]   # kv_lora_rank, qk_rope_head_dim, rotary_dim; sm_scale
    # defaults to 1/sqrt(c+r), is_neox_style defaults to True


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the evaluator
    # calls this function after set_seed, so multiple correctness trials therefore
    # see diverse inputs
    batch = 2
    num_heads = 128
    seq_len = 1024
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    rotary_dim = 64
    rope_base = 10.0
    max_positions = 16384

    total_kv = batch * seq_len
    q = torch.randn(batch, num_heads, kv_lora_rank + qk_rope_head_dim).to(torch.bfloat16)
    # The row pool exceeds the actual usage: unreferenced rows are garbage data, used to check that the indirect addressing is correct
    kv_cache = torch.randn(total_kv + 8, kv_lora_rank + qk_rope_head_dim).to(torch.bfloat16)
    kv_indptr = (torch.arange(batch + 1) * seq_len).to(torch.int32)
    kv_indices = torch.randperm(total_kv + 8)[:total_kv].to(torch.int32)
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    t = torch.arange(max_positions, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16)
    # official convention: the current query position == that batch's KV length
    positions = torch.full((batch,), seq_len, dtype=torch.int32)
    return [q, kv_cache, cos_sin_cache, positions, kv_indptr, kv_indices]


def make_inputs(batch: int, num_heads: int, seq_lens, kv_lora_rank: int,
                qk_rope_head_dim: int, rotary_dim: int, dtype: str = "bfloat16",
                is_neox_style: bool = True, positions=None, rope_base: float = 10.0,
                max_positions: int = 16384, extra_tokens: int = 8, seed: int = 0):
    """Generate inputs from a public/hidden/perf case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device). seq_lens gives each batch's KV length explicitly (a single
    value broadcasts to the whole batch; by default 1..1024 random per batch);
    positions defaults to seq_lens (official convention: the current query
    position == sequence length) and may be given explicitly to decouple them;
    is_neox_style is the Model init hyper-parameter passed along with the case (it
    does not affect the input tensor contents, the evaluation side uses it to
    build the Model); when extra_tokens > 0 the physical row pool gets that many
    extra garbage rows appended.
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)

    assert rotary_dim % 2 == 0 and 0 < rotary_dim <= qk_rope_head_dim, \
        "rotary_dim must be a multiple of 2 and <= qk_rope_head_dim"
    if seq_lens is None:
        seq_lens = torch.randint(1, 1025, (batch,), generator=gen).tolist()
    if len(seq_lens) == 1 and batch > 1:      # a single value broadcasts to the whole batch
        seq_lens = seq_lens * batch
    assert len(seq_lens) == batch and all(int(s) >= 1 for s in seq_lens), \
        "seq_lens length must equal batch and every entry must be >= 1"
    if positions is None:
        positions = list(seq_lens)
    positions = [int(p) for p in positions]
    assert len(positions) == batch and max(positions) < max_positions, \
        "positions length must equal batch and every value must be < max_positions"

    total_kv = sum(int(s) for s in seq_lens)
    q = torch.randn(batch, num_heads, kv_lora_rank + qk_rope_head_dim, generator=gen).to(dt)
    kv_cache = torch.randn(total_kv + extra_tokens, kv_lora_rank + qk_rope_head_dim,
                           generator=gen).to(dt)
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(torch.tensor([int(s) for s in seq_lens], dtype=torch.int64),
                                 dim=0).to(torch.int32)
    kv_indices = torch.randperm(total_kv + extra_tokens, generator=gen)[:total_kv].to(torch.int32)

    inv_freq = 1.0 / (rope_base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    t = torch.arange(max_positions, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dt)
    positions_t = torch.tensor(positions, dtype=torch.int32)
    return q, kv_cache, cos_sin_cache, positions_t, kv_indptr, kv_indices
