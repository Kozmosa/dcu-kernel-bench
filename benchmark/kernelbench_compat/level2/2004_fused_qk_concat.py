# 2004_fused_qk_concat -- MLA-style QK preprocessing: RoPE rotation on the pe
# segment plus nope/pe concatenation, as a KernelBench compatible model_class task
# file.
#
# This file must be self-contained: the evaluator loads the task via exec(source),
# so local module imports are forbidden -- the reference implementation
# (Model.forward) and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=2, problem_id=2004, backend=triton.

import torch
import torch.nn as nn


class Model(nn.Module):
    """MLA-style QK preprocessing fused operator: rotary position embedding (RoPE)
    on the pe segment plus nope/pe concatenation.

    (1) Operator semantics (mathematical definition and boundary behaviour):
    The Q side takes q_nope [B, QH, D_nope] and q_pe [B, QH, D_pe]; the K side takes
    k_nope [B, KH, D_nope] and k_pe [B, KH, D_pe]. QH must be an integer multiple of
    KH (GQA; write QH_PER_KH = QH // KH) and D_pe must be even. The output is packed
    into a single tensor out [B, QH + KH, D_nope + D_pe]: along the head dimension
    the first QH rows are the Q result and the last KH rows are the K result (each
    KV head is computed and written back only once, i.e. the GQA deduplicated
    write-back semantics).

    The nope segment is copied verbatim: out[:, :QH, :D_nope] = q_nope,
    out[:, QH:, :D_nope] = k_nope. The pe segment (x means q_pe or k_pe, promoted to
    float32 before the computation):
    - apply_rope=False (pure concatenation mode): out[..., D_nope:] = x copied
      verbatim; pos / cos / sin are ignored;
    - apply_rope=True (rope mode): for each batch b take c = cos[pos[b]] and
      s = sin[pos[b]] (width d_freq; d_freq == D_pe is the full-width table,
      d_freq == D_pe//2 is the first-half reuse table), and expand to width D_pe
      according to the rotation style:
        full width: c~ = c;
        first-half reuse + NEOX: c~ = [c, c] (the whole segment repeated once; both
          halves share the same frequency);
        first-half reuse + GPTJ: c~ = [c0, c0, c1, c1, ...] (every element repeated
          twice back to back).
      rotate_half(x) is defined as:
        NEOX (is_neox=True): rotate_half(x)[i] = -x[i + D_pe//2]
          (i < D_pe//2), = x[i - D_pe//2] (i >= D_pe//2);
        GPTJ (is_neox=False): rotate_half(x)[2j] = -x[2j+1],
          rotate_half(x)[2j+1] = x[2j].
      pe segment output = x * c~ + rotate_half(x) * s~, cast back to the input dtype.
    Boundary behaviour: pos[b] is in [0, max_pos) (out of range is undefined
    behaviour); this operator is a memory-bound preprocessing step and computes no
    attention score; the nope segment and the pe segment in pure concatenation mode
    are verbatim copies and must match bit-for-bit; the whole forward is out-of-place
    and must not overwrite any input in place.

    (2) Input/output specification:
    - q_nope / q_pe / k_nope / k_pe: float16 / bfloat16, contiguous, shapes as above
      (QH == KH * QH_PER_KH, one shared batch for all four tensors);
    - pos: int64, [B], one position index per batch (it is cast to floating point
      under the evaluation precision and restored to integer losslessly at the
      entry);
    - cos / sin: same dtype as the data, [max_pos, d_freq], contiguous, used only
      through the row-style indexing cos[pos[b], d];
    - returns: out [B, QH + KH, D_nope + D_pe] with the same dtype as q_nope;
    - hyper-parameters (__init__): is_neox -- True selects the NEOX rotation style /
      False selects GPTJ; apply_rope -- True rotates the pe segment / False pure
      concatenation.

    (3) Implementation constraints (violations are scored as failures):
    - The core computation (row indexing of cos/sin by position, rotation of the pe
      segment, the nope/pe concatenated write-back, and the K-side deduplication
      under GQA) must be completed inside the submitted file with a self-written
      Triton kernel: a single kernel (one launch) with a single
      read-compute-write pass;
    - The __init__ and forward signatures of ModelNew must not be changed;
    - Use float32 for intermediate computation; cast the output back to the input
      dtype.

    (4) Forbidden list and target hardware:
    - Calling ATen shortcuts such as torch.cat / torch.concat (including the tensor
      method .cat) or index_select to complete this task's computation directly is
      forbidden; the authoritative forbidden list is the task.yaml forbidden list
      (it covers the source operator library and other closed-source acceleration
      libraries; the final static audit does substring matching on the code outside
      docstrings / comments);
    - Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    """

    def __init__(self, is_neox: bool = True, apply_rope: bool = True):
        super().__init__()
        self.is_neox = bool(is_neox)
        self.apply_rope = bool(apply_rope)

    def forward(self, q_nope, q_pe, k_nope, k_pe, pos, cos, sin):
        # The evaluator casts all input tensors to the evaluation precision (fp32 for
        # the triton backend); pos is a position index (integer semantics) and is
        # exactly representable in fp32 as long as max_pos < 2^24, so it is restored
        # to integer losslessly here. Note: forward's statements must stay
        # single-line (the framework scaffold re-indents statement by statement, so
        # multi-line blocks get broken)
        pos = pos.to(torch.long)
        B, QH, D_nope = q_nope.shape
        KH = k_nope.shape[1]
        D_pe = q_pe.shape[-1]
        assert q_pe.shape[0] == B and k_nope.shape[0] == k_pe.shape[0] == B, "the batch dimension must match"
        assert q_pe.shape[1] == QH and k_pe.shape[1] == KH, "the head counts of Q and K must each match"
        assert k_nope.shape[2] == D_nope and k_pe.shape[2] == D_pe, "the D_nope and D_pe of q and k must match"
        assert QH % KH == 0, "QH must be an integer multiple of KH (GQA)"
        assert D_pe % 2 == 0, "D_pe must be even (half-width rotation)"
        out = torch.empty((B, QH + KH, D_nope + D_pe), dtype=q_nope.dtype, device=q_nope.device)
        out[:, :QH, :D_nope] = q_nope
        out[:, QH:, :D_nope] = k_nope
        # rope mode: gather the cos/sin rows by position and expand them to width
        # D_pe (float32); in pure concatenation mode cos_f/sin_f do not take part in
        # the result (the ternary below takes the verbatim pe segment directly)
        d_freq = cos.shape[-1]
        assert d_freq in (D_pe, D_pe // 2), "the cos/sin width must be D_pe or D_pe//2"
        cos_b = cos.to(torch.float32)[pos]
        sin_b = sin.to(torch.float32)[pos]
        cos_f = (torch.cat((cos_b, cos_b), dim=-1) if self.is_neox else cos_b.repeat_interleave(2, dim=-1)) if d_freq == D_pe // 2 else cos_b
        sin_f = (torch.cat((sin_b, sin_b), dim=-1) if self.is_neox else sin_b.repeat_interleave(2, dim=-1)) if d_freq == D_pe // 2 else sin_b
        cos_f = cos_f.unsqueeze(1)                # [B, 1, D_pe] broadcast over the head dimension
        sin_f = sin_f.unsqueeze(1)
        rotate = (lambda x: torch.cat((-x[..., D_pe // 2:], x[..., : D_pe // 2]), dim=-1)) if self.is_neox else (lambda x: torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2))
        q32, k32 = q_pe.to(torch.float32), k_pe.to(torch.float32)
        out[:, :QH, D_nope:] = (q32 * cos_f + rotate(q32) * sin_f).to(q_nope.dtype) if self.apply_rope else q_pe
        out[:, QH:, D_nope:] = (k32 * cos_f + rotate(k32) * sin_f).to(q_nope.dtype) if self.apply_rope else k_pe
        return out


def get_init_inputs():
    return [True, True]   # is_neox=NEOX rotation style; apply_rope=rope+concat (the flagship path)


def get_inputs():
    # Fixed shape family (matching public_cases.json); the random part consumes the
    # global RNG -- the evaluator calls this function after set_seed, so multiple
    # correctness trials therefore see diverse inputs.
    B, KH, qh_per_kh = 8, 4, 4
    QH = KH * qh_per_kh                 # 16
    D_nope, D_pe, d_freq = 512, 64, 32  # first-half reuse table (the common MLA configuration)
    max_pos = 1024
    dt = torch.bfloat16

    q_nope = torch.randn((B, QH, D_nope), dtype=dt)
    q_pe = torch.randn((B, QH, D_pe), dtype=dt)
    k_nope = torch.randn((B, KH, D_nope), dtype=dt)
    k_pe = torch.randn((B, KH, D_pe), dtype=dt)
    pos = torch.randint(0, max_pos, (B,), dtype=torch.int64)
    freqs = torch.randn((max_pos, d_freq), dtype=dt)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return [q_nope, q_pe, k_nope, k_pe, pos, cos, sin]


def make_inputs(B, KH, QH_PER_KH, D_nope, D_pe, max_pos, d_freq=None,
                dtype="bfloat16", seed=0, is_neox=True, apply_rope=True):
    """Deterministic case generator: torch.Generator().manual_seed(seed), generated
    on CPU.

    The fields map one-to-one onto the hidden/perf cases (B / KH / QH_PER_KH /
    D_nope / D_pe / max_pos / d_freq / dtype / seed); is_neox and apply_rope are
    Model initialization settings, recorded together with the case so the evaluation
    side can construct the Model, and they take no part in tensor generation.
    Runs on the evaluation side only (generates
    [q_nope, q_pe, k_nope, k_pe, pos, cos, sin]).
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    QH = KH * QH_PER_KH
    if d_freq is None:
        d_freq = D_pe
    assert D_pe % 2 == 0, "D_pe must be even"
    assert d_freq in (D_pe, D_pe // 2), "the cos/sin width must be D_pe or D_pe//2"

    q_nope = torch.randn((B, QH, D_nope), dtype=dt, generator=gen)
    q_pe = torch.randn((B, QH, D_pe), dtype=dt, generator=gen)
    k_nope = torch.randn((B, KH, D_nope), dtype=dt, generator=gen)
    k_pe = torch.randn((B, KH, D_pe), dtype=dt, generator=gen)
    pos = torch.randint(0, max_pos, (B,), generator=gen)
    freqs = torch.randn((max_pos, d_freq), dtype=dt, generator=gen)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return [q_nope, q_pe, k_nope, k_pe, pos, cos, sin]
