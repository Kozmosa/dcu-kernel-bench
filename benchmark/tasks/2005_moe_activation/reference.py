# 2005_moe_activation -- model_class (KernelBench compatible) task file for the MoE
# FFN activation variant family (gated SwiGLU variants x chunked / interleaved
# layouts x non-gated element-wise activations).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=2, problem_id=2005, backend=triton.

import torch
import torch.nn as nn


class Model(nn.Module):
    """MoE FFN activation variant family: gated activations (SwiGLU family
    variants) and non-gated element-wise activations, dispatched by the __init__
    parameters activation / alpha / limit.

    All computation runs in float32 and the output is cast back to the input dtype.
    Define
        SiLU(u)      = u * sigmoid(u)
        GELU(u)      = 0.5 * u * (1 + erf(u * 0.7071067811865476))
        GELU_tanh(u) = 0.5 * u * (1 + tanh(0.7978845608028654
                                           * (u + 0.044715 * u^3)))
        ReLU2(u)     = max(u, 0)^2
    sigmoid / erf / tanh are all evaluated as float32 math functions.

    [gated variants] The input x has shape (M, 2N) and the output (M, N). Two data
    layouts:
        chunked (default) : gate = x[:, :N], up = x[:, N:]
        interleaved       : gate = x[:, 0::2] (even columns), up = x[:, 1::2] (odd
                            columns)
    Write the clamped quantities as gate' = min(gate, limit) (upper bound only) and
    up' = clamp(up, -limit, limit) (symmetric clamping); the output of each variant
    is:
        activation="silu", alpha=None, limit=None:
            out = SiLU(gate) * up                        (standard SwiGLU)
        activation="silu", alpha=None, limit=L:
            out = SiLU(gate') * up'                       (clamped SwiGLU)
        activation="silu", alpha=A, limit=L (A and L must be given together):
            out = gate' * sigmoid(A * gate') * (up' + 1)  (scaled clamped gate)
        activation="gelu" (does not accept alpha / limit):
            out = GELU(gate) * up
        activation="gelu_tanh" (does not accept alpha / limit):
            out = GELU_tanh(gate) * up
        activation="swiglustep" (limit defaults to 7.0, does not accept alpha):
            out = clamp(SiLU(gate), max=limit) * up'      (SiLU first, then the
                                                           upper clamp)
        activation="swiglu_interleaved" (interleaved layout; alpha defaults to
        1.702, limit defaults to 7.0):
            out = gate' * sigmoid(alpha * gate') * (up' + 1)

    [non-gated variants] The input x has shape (M, N) and the output (M, N);
    element-wise activation, alpha / limit are ignored:
        activation="silu_no_mul"      : out = SiLU(x)
        activation="gelu_no_mul"      : out = GELU(x)
        activation="gelu_tanh_no_mul" : out = GELU_tanh(x)
        activation="relu2"            : out = ReLU2(x)

    Boundary behaviour: clamping of gate is upper-bound only (nothing is clamped on
    the negative side) while clamping of up is symmetric; the clamping is applied
    before the activation, except for swiglustep, which computes SiLU first and then
    clamps the result on the upper side. Illegal parameter combinations (an unknown
    variant name; gelu / gelu_tanh with alpha or limit; swiglustep with alpha; silu
    given alpha but not limit) are rejected at construction time.

    Input/output specification:
        x    float16 / bfloat16 (M, n), contiguous, M >= 1; the gated variants
             require n to be even (N = n/2 has no alignment constraint: non-powers
             of two and sizes below the common block size are allowed); the
             non-gated variants accept any n >= 1. The evaluator casts the input to
             fp32 before passing it in; fp16/bf16 -> fp32 is lossless.
        out  same dtype as x; shape (M, n/2) for the gated variants and (M, n) for
             the non-gated ones.
        The input value range is the usual randn magnitude (no inf/NaN).

    Implementation constraints (violations are scored as failures):
        - The core computation (chunked / interleaved data selection, clamping,
          activation, gated multiply) must be completed with a Triton kernel inside
          the submitted file; intermediates are always float32.
        - The __init__ and forward signatures of ModelNew must not be changed.
        - Use float32 for intermediate accumulation; cast the output back to the
          input dtype.

    (4) Forbidden list and target hardware:
    - Calling ATen shortcuts such as torch.nn.functional.silu /
      torch.nn.functional.gelu / torch.nn.SiLU / torch.nn.GELU / torch.sigmoid /
      torch.erf / torch.tanh / torch.relu / torch.clamp to complete this task's
      computation directly is forbidden; the authoritative forbidden list is the
      task.yaml forbidden list (it covers the source operator library and other
      closed-source acceleration libraries; the final static audit does substring
      matching on the code outside docstrings / comments);
    - Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    """

    def __init__(self, activation: str = "silu", alpha=None, limit=None):
        super().__init__()
        gated = ("silu", "gelu", "gelu_tanh", "swiglustep", "swiglu_interleaved")
        nomul = ("silu_no_mul", "gelu_no_mul", "gelu_tanh_no_mul", "relu2")
        name = str(activation).lower()
        assert name in gated or name in nomul, (
            "activation must be one of " + "/".join(gated + nomul) + " variants")
        # parameter-combination legality (consistent with the variant definitions;
        # illegal combinations are rejected at construction time)
        if name in ("gelu", "gelu_tanh"):
            assert alpha is None and limit is None, "gelu / gelu_tanh do not accept alpha / limit"
        if name == "swiglustep":
            assert alpha is None, "swiglustep does not accept alpha"
        if name == "silu" and alpha is not None:
            assert limit is not None, "silu must set limit as well when alpha is set"
        # default filling (only the following two variants carry defaults)
        if name == "swiglustep" and limit is None:
            limit = 7.0
        if name == "swiglu_interleaved":
            alpha = 1.702 if alpha is None else alpha
            limit = 7.0 if limit is None else limit
        self.activation = name
        self.is_gated = name in gated
        self.alpha = None if alpha is None else float(alpha)
        self.limit = None if limit is None else float(limit)

    def forward(self, x):
        # The evaluator casts all inputs to fp32; compute uniformly in float32 and
        # cast the output back to the input dtype
        xf = x.to(torch.float32)
        assert xf.dim() == 2, "the input must be a 2D (M, n) tensor"
        if not self.is_gated:
            # non-gated: element-wise activation, shape unchanged (alpha / limit are
            # ignored)
            if self.activation == "silu_no_mul":
                out = xf * torch.sigmoid(xf)
            elif self.activation == "gelu_no_mul":
                out = 0.5 * xf * (1.0 + torch.erf(xf * 0.7071067811865476))
            elif self.activation == "gelu_tanh_no_mul":
                inner = 0.7978845608028654 * (xf + 0.044715 * xf * xf * xf)
                out = 0.5 * xf * (1.0 + torch.tanh(inner))
            else:   # relu2
                act = torch.clamp(xf, min=0.0)
                out = act * act
            return out.to(x.dtype)

        assert xf.shape[-1] % 2 == 0, "the gated variants require the last dimension n to be even"
        half = xf.shape[-1] // 2
        interleaved = self.activation == "swiglu_interleaved"
        gate = xf[:, 0::2] if interleaved else xf[:, :half]   # interleaved / chunked layout
        up = xf[:, 1::2] if interleaved else xf[:, half:]
        if self.activation == "swiglustep":
            # SiLU first, then the upper clamp: clamp(SiLU(gate), max=limit) * up'
            silu_g = gate * torch.sigmoid(gate)
            out = silu_g.clamp(max=self.limit) * up.clamp(min=-self.limit, max=self.limit)
        if self.activation == "silu" and self.alpha is not None:
            # scaled clamped gate: gate' * sigmoid(alpha*gate') * (up' + 1)
            gate_p = gate.clamp(max=self.limit)
            up_p = up.clamp(min=-self.limit, max=self.limit)
            out = gate_p * torch.sigmoid(self.alpha * gate_p) * (up_p + 1.0)
        if self.activation == "silu" and self.alpha is None and self.limit is not None:
            # clamped SwiGLU: SiLU(gate') * up'
            gate_p = gate.clamp(max=self.limit)
            up_p = up.clamp(min=-self.limit, max=self.limit)
            out = gate_p * torch.sigmoid(gate_p) * up_p
        if self.activation == "silu" and self.alpha is None and self.limit is None:
            out = gate * torch.sigmoid(gate) * up             # standard SwiGLU
        if self.activation == "gelu":
            out = 0.5 * gate * (1.0 + torch.erf(gate * 0.7071067811865476)) * up
        if self.activation == "gelu_tanh":
            inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
            out = 0.5 * gate * (1.0 + torch.tanh(inner)) * up
        if self.activation == "swiglu_interleaved":
            # gate' * sigmoid(alpha*gate') * (up' + 1)
            gate_p = gate.clamp(max=self.limit)
            up_p = up.clamp(min=-self.limit, max=self.limit)
            out = gate_p * torch.sigmoid(self.alpha * gate_p) * (up_p + 1.0)
        return out.to(x.dtype)


def get_init_inputs():
    return ["silu", 1.702, 7.0]   # activation / alpha / limit -> the scaled clamped gate variant


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the evaluator
    # calls this function after set_seed, so multiple correctness trials therefore
    # see diverse inputs (M=128, 2N=4096, i.e. N=2048)
    m, n = 128, 4096
    x = torch.randn(m, n).to(torch.bfloat16)
    return [x]


def make_inputs(m: int, n: int, activation: str, alpha=None, limit=None,
                dtype: str = "float16", seed: int = 0, scale: float = 1.0):
    """Generate inputs from a public/hidden case description. Runs on the
    evaluation side only (generated on CPU; the evaluator moves them to the
    device).

    The consumption order is fixed: x = torch.randn(m, n) * scale, then cast to
    dtype; everything comes from the same torch.Generator(seed), so the same seed
    reproduces inputs bit-for-bit; activation / alpha / limit map one-to-one onto
    the case fields and are passed through to the Model constructor. scale only
    controls the input magnitude so that the clamping boundaries are covered; it
    takes no part in the operator semantics.
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert m >= 1 and n >= 1, "requires M >= 1 and n >= 1"
    x = (torch.randn(m, n, generator=gen) * float(scale)).to(dt)
    return (x,)
