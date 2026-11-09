# 2008_rope -- model_class (KernelBench compatible) task file for the forward pass of
# RoPE (rotary position embedding) in the dense sbhd layout.
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=2, problem_id=2008, backend=triton.

import torch
import torch.nn as nn


class Model(nn.Module):
    """RoPE (Rotary Position Embedding) forward: dense sbhd layout, with the
    rotation angles given directly as a freqs tensor (not a cached cos/sin table);
    the channels of every head vector are rotated as planes according to the
    sequence position.

    Semantics (the trigonometric functions and the multiply-add are all evaluated in
    float32; the output is cast back to the input dtype):
       The input is x (S, B, H, D) together with freqs (S, 1, 1, F); freqs
       broadcasts over the B and H dimensions.
       Rotation width R = F * 2 (reuse_freqs_front_part=True) or R = F (False).
       Split the last dimension (length D) into a "rotation segment" (length R) and
       a "pass-through segment" (length D - R): with nope_first=False the rotation
       segment occupies the first R channels, with True the last R channels.
       Angle row Theta (length R): with reuse=False, Theta = freqs, one angle per
       rotation channel; with reuse=True, F = R/2 angles are expanded to R columns --
       the NEOX style tiles the whole segment ([a, b] -> [a, b, a, b], each half used
       once), the GPTJ style repeats adjacently ([a, b] -> [a, a, b, b], each number
       pair shares one angle).
       For the rotation segment x_rot (length R) compute:
         out_rot = x_rot * cos(Theta) + rotate_half(x_rot) * sin(Theta)
       rotate_half is decided by the __init__ argument rotate_style:
         NEOX (rotate_style=0): rotate_half(x_rot) = concat(-x_rot[R/2:],
           x_rot[:R/2]), i.e. the first half u and the second half v are swapped with
           the u segment negated, which is equivalent to a plane rotation of every
           pair (u_j, v_j).
         GPTJ (rotate_style=1): even-odd channel pairing, rotate_half(x_rot)[2j] =
           -x_rot[2j+1], rotate_half(x_rot)[2j+1] = x_rot[2j].
       The pass-through segment is appended back without any transformation: with
       nope_first=False, out = concat(out_rot, pass-through segment); with True,
       out = concat(pass-through segment, out_rot).
       Boundary behaviour: when R = D the pass-through segment is empty (full-width
       rotation); when R < D the pass-through segment is kept verbatim, bit for bit;
       the last dimension F of freqs has only the three legal values {D, D/2, D/4}
       (their combination with reuse and R is listed below); all other
       configurations are undefined.

    Input/output specification:
        x      (S, B, H, D)  float16 / bfloat16, contiguous; D is a power of two
                              (4..256), S, B, H are arbitrary positive integers
        freqs  (S, 1, 1, F)  same dtype as x, contiguous; legal combinations:
                              full-width rotation reuse=False: F = D;
                              full-width rotation reuse=True: F = D/2;
                              partial rotation (R=D/2) reuse=False: F = D/2;
                              partial rotation (R=D/2) reuse=True: F = D/4
        output out (S, B, H, D), same dtype as x
        (the evaluator casts the input to fp32 before passing it in; fp16/bf16 ->
        fp32 is lossless)
        hyper-parameters (__init__): rotate_style (0=NEOX, 1=GPTJ), nope_first,
        reuse_freqs_front_part.

    Implementation constraints (violations are scored as failures):
        - The core computation (angle expansion, rotate_half, cos/sin and the
          multiply-add, concatenation of the pass-through segment) must be completed
          inside the submitted file with a Triton kernel.
        - The __init__ and forward signatures of ModelNew must not be changed.
        - Use float32 for intermediate computation; cast the output back to the input
          dtype.

    Forbidden: scaled_dot_product_attention / sdpa / flash_attn /
    torch.matmul / torch.bmm / torch.einsum / torch.softmax / torch.polar /
    torch.view_as_complex.
    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: dense sbhd non-cached forward; variable-length thd, cos/sin cache
    addressing, the dual-input GQA case, the backward pass, and 2D image rotation are
    outside this task.
    """

    def __init__(self, rotate_style: int, nope_first: bool, reuse_freqs_front_part: bool):
        super().__init__()
        self.rotate_style = int(rotate_style)
        self.nope_first = bool(nope_first)
        self.reuse_freqs_front_part = bool(reuse_freqs_front_part)

    def forward(self, x, freqs):
        # The evaluator casts all inputs to fp32; fp16/bf16 -> fp32 is lossless, so
        # compute uniformly in float32 and cast the output back to the dtype at the
        # entry (fp32 online, the original dtype offline).
        # Note: do not use statements that carry an else branch inside forward, so
        # that the statement-by-statement scaffold extraction by the loader still
        # yields valid Python.
        dt = x.dtype
        x32 = x.to(torch.float32)
        f32 = freqs.to(torch.float32)

        d = x32.shape[-1]
        rotate_dim = f32.shape[-1] * (2 if self.reuse_freqs_front_part else 1)

        # (1) rotation / pass-through segmentation: with nope_first=True the rotation
        #     segment is at the tail [d-R, d), otherwise at the front [0, R); the
        #     pass-through segment keeps its original order when concatenated (empty
        #     when rotate_dim == D)
        start = (d - rotate_dim) if self.nope_first else 0
        x_rot = x32[..., start : start + rotate_dim]
        x_pass = torch.cat((x32[..., :start], x32[..., start + rotate_dim :]), dim=-1)

        # (2) angle row: reuse mode expands the F=R/2 columns to R columns (NEOX
        #     tiling / GPTJ adjacent repetition)
        theta = f32
        if self.reuse_freqs_front_part and self.rotate_style == 0:
            theta = f32.repeat([1] * (f32.dim() - 1) + [2])
        if self.reuse_freqs_front_part and self.rotate_style == 1:
            theta = f32.repeat_interleave(2, dim=-1)

        # (3) rotate_half + plane rotation, all in float32
        half = rotate_dim // 2
        rot_half = x_rot
        if self.rotate_style == 0:  # NEOX: swap the first half u / second half v, negate the u segment
            x1 = x_rot[..., :half]
            x2 = x_rot[..., half:]
            rot_half = torch.cat((-x2, x1), dim=-1)
        if self.rotate_style == 1:  # GPTJ: even-odd pairing; the even slot takes the negated odd element, the odd slot takes the even element
            x1 = x_rot[..., ::2]
            x2 = x_rot[..., 1::2]
            rot_half = torch.stack((-x2, x1), dim=-1).flatten(-2)
        x_embed = x_rot * torch.cos(theta) + rot_half * torch.sin(theta)

        # (4) append the pass-through segment back (it comes first when nope_first)
        #     and then cast the whole thing back to the input dtype
        pieces = (x_pass, x_embed) if self.nope_first else (x_embed, x_pass)
        out = torch.cat(pieces, dim=-1)
        return out.to(dt)


def get_init_inputs():
    return [0, False, False]  # rotate_style=NEOX, nope_first=False, reuse=False


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the evaluator
    # calls this function after set_seed, so multiple correctness trials therefore
    # see diverse inputs. Consumption order: x / freqs.
    # The configuration must match get_init_inputs: NEOX, pass-through segment last,
    # reuse=False -> full-width rotation, freqs last dimension F = D = 128
    s, b, h, d = 64, 2, 8, 128
    x = torch.randn(s, b, h, d).to(torch.float16)
    freqs = torch.randn(s, 1, 1, d).to(torch.float16)
    return [x, freqs]


def make_inputs(s: int, b: int, h: int, d: int, rotate_style: int, nope_first: bool,
                reuse_freqs_front_part: bool, rotary_percent: float = 1.0,
                dtype: str = "float16", seed: int = 0):
    """Generate inputs from a public/hidden case description. Runs on the
    evaluation side only (generated on CPU; the evaluator moves them to the
    device).

    rotary_percent is the fraction of D occupied by the rotation width (1.0 full
    width / 0.5 partial rotation); together with reuse_freqs_front_part it decides
    the last dimension of freqs, F = int(d * rotary_percent) //
    (2 if reuse else 1); rotate_style / nope_first do not affect the tensor shapes
    and are only used to construct the Model (mapping one-to-one onto the case
    fields). The consumption order is fixed: x (randn), freqs (randn); everything
    comes from the same torch.Generator(seed), so the same seed reproduces inputs
    bit-for-bit.
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    rotate_dim = int(d * rotary_percent)
    assert d % 2 == 0 and rotate_dim % 2 == 0 and d % rotate_dim == 0, \
        "D must be even and the rotation width R must be an even number that divides D"
    freqs_d = rotate_dim // (2 if reuse_freqs_front_part else 1)
    x = torch.randn(s, b, h, d, generator=gen).to(dt)
    freqs = torch.randn(s, 1, 1, freqs_d, generator=gen).to(dt)
    return (x, freqs)
