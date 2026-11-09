# 6001_fused_adamw -- model_class (KernelBench compatible) task file for the
# fused AdamW optimizer family (standard / Transformer schedule).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=2, problem_id=6001, backend=cuda (HIP).

import torch
import torch.nn as nn


class Model(nn.Module):
    """Fused AdamW optimizer family: a single functional (out-of-place) parameter
    update for standard AdamW and for Transformer-schedule AdamW.

    (1) Operator semantics: the __init__ hyper-parameters lr, beta1, beta2, eps,
    weight_decay are construction constants, and `variant` selects one of two
    update paths. Let `step` be a 0-dimensional float32 scalar tensor (the
    current iteration count). The four state tensors params / grads / exp_avg /
    exp_avg_sq have the same shape and are element-wise aligned; the whole update
    is computed in float32 (inputs are promoted to float32 first):
        1) decoupled weight decay (the AdamW form, applied in both variants --
           equivalent to p = params - lr * weight_decay * params, not L2
           regularization):
               p = params * (1 - lr * weight_decay)
        2) first moment: m = beta1 * exp_avg + (1 - beta1) * grads
        3) second moment: v = beta2 * exp_avg_sq + (1 - beta2) * grads^2
        4) variant="adamw" (standard AdamW, bias correction ON):
               bc1 = 1 - beta1^step; bc2 = 1 - beta2^step
               out = p - lr * (m / bc1) / (sqrt(v / bc2) + eps)
           variant="transformer_adamw" (Transformer schedule, bias correction
           OFF -- the only difference between the two variants):
               out = p - lr * m / (sqrt(v) + eps)
    The two variants are line-by-line identical apart from bias correction; eps is
    added outside the sqrt (denom = sqrt(...) + eps); the power beta^step is a
    float32 scalar power.
    Boundary behaviour: at step=1, bc1 = 1 - beta1 and bc2 = 1 - beta2 take their
    minimum values (the standard AdamW path amplifies the most); for very large
    step the two variants converge numerically; exp_avg_sq is always
    non-negative (second-moment constraint, generated in [0, 0.5)) and
    denom >= eps guarantees no division by zero; this operator has no reduction
    and no indexing -- it is a pure element-wise map.

    (2) Input/output specification:
      - params / grads / exp_avg / exp_avg_sq: float32 or float16 tensors, all
        with the same shape (ndim 1-4, numel >= 1, including non-power-of-two and
        boundary sizes), contiguous;
      - step: a 0-dimensional float32 scalar tensor (numel == 1) whose value is a
        positive-integer-semantics step count (>= 1);
      - returns: the updated parameters, a new tensor with the same shape and
        dtype as params (single-tensor output, out-of-place; no input may be
        modified in place, and no state such as exp_avg / exp_avg_sq may be
        returned).

    (3) Implementation constraints (violations are scored as failures):
      - The core computation (decoupled weight decay, first/second moment update,
        bias-correction power, sqrt, element-wise update) must be completed with a
        HIP kernel inside the submitted file;
      - out-of-place: params / grads / exp_avg / exp_avg_sq must not be modified
        in place;
      - the __init__ and forward signatures of ModelNew must not be changed;
      - intermediate computation uses float32 (fp16 inputs are promoted to
        float32 as well), and the final result is cast back to the dtype of
        params;
      - step arrives as a 0-dimensional tensor and participates in the power
        operation as a scalar inside the kernel; it must not be materialized into
        a temporary tensor of the same shape as params.

    (4) Forbidden list and target hardware:
      - the whole torch.optim family is forbidden (torch.optim.AdamW / Adam /
        SGD / ... plus their _functional / _multi_tensor internal paths);
      - any fused-optimizer library and closed-source entry point is forbidden:
        apex / deepspeed / bitsandbytes / the fused_adam entry-point family /
        hipdnn / miopen;
      - completing this task via an ATen shortcut is forbidden: torch.addcdiv /
        torch.addcmul (including the .addcdiv / .addcdiv_ / .addcmul / .addcmul_
        tensor methods and in-place forms, which finish the parameter update or
        the moment update in a single call); the multi-tensor primitives
        torch._foreach_addcdiv / torch._foreach_addcmul are equally forbidden
        (including their in-place _ forms, which complete the core update step in
        a single call) (the final static audit does substring matching on code
        outside docstrings/comments);
      - target hardware: Hygon DCU (gfx936/gfx938). Implementation language: HIP
        (compiled by hipcc; write inside the CUDA_CU_SRC / CUDA_CPP_SRC anchor
        regions; device code must include <hip/hip_runtime.h> --
        <cuda_runtime.h> is not available on this machine).
    """

    def __init__(self, variant: str = "adamw", lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.01):
        super().__init__()
        assert variant in ("adamw", "transformer_adamw"), \
            "variant must be one of adamw / transformer_adamw"
        self.variant = variant
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)

    def forward(self, params, grads, exp_avg, exp_avg_sq, step):
        # Promote the inputs to float32; `step` is a 0-dimensional scalar tensor
        # (the evaluator may also pass a plain numeric value through).
        p = params.to(torch.float32)
        g = grads.to(torch.float32)
        m = exp_avg.to(torch.float32)
        v = exp_avg_sq.to(torch.float32)
        s = step.to(torch.float32) if torch.is_tensor(step) \
            else torch.tensor(float(step))

        # 1) decoupled weight decay (the AdamW form, not L2)
        p = p * (1.0 - self.lr * self.weight_decay)
        # 2) first moment
        m = self.beta1 * m + (1.0 - self.beta1) * g
        # 3) second moment
        v = self.beta2 * v + (1.0 - self.beta2) * g * g

        if self.variant == "adamw":
            # standard AdamW: bias correction ON
            bc1 = 1.0 - torch.pow(self.beta1, s)
            bc2 = 1.0 - torch.pow(self.beta2, s)
            denom = torch.sqrt(v / bc2) + self.eps
            update = (m / bc1) / denom
        else:
            # transformer_adamw: bias correction OFF (the only difference)
            denom = torch.sqrt(v) + self.eps
            update = m / denom

        return (p - self.lr * update).to(params.dtype)


def get_init_inputs():
    # variant, lr, beta1, beta2, eps, weight_decay (the torch.optim.AdamW defaults family)
    return ["adamw", 1e-3, 0.9, 0.999, 1e-8, 0.01]


def get_inputs():
    # Fixed shape family (consistent with public_cases.json); the random part
    # consumes the global RNG -- the evaluator calls this function after set_seed,
    # so multiple correctness trials therefore see diverse inputs. The consumption
    # order is fixed: params -> grads -> exp_avg -> exp_avg_sq.
    shape = (8, 3, 224, 224)
    params = torch.rand(*shape)
    grads = torch.randn(*shape)
    exp_avg = torch.rand(*shape) * 0.5
    exp_avg_sq = torch.rand(*shape) * 0.5
    step = torch.tensor(10.0)
    return [params, grads, exp_avg, exp_avg_sq, step]


def make_inputs(shape, dtype="float32", step=10.0, variant="adamw",
                lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                weight_decay=0.01, seed=0):
    """Generate inputs from a public/hidden case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device).

    The consumption order is fixed: params -> grads -> exp_avg -> exp_avg_sq
    (all from the same torch.Generator(seed), so the same seed reproduces inputs
    bit-for-bit); `step` is a deterministic scalar tensor and does not consume the
    RNG. variant / lr / beta1 / beta2 / eps / weight_decay are the init
    hyper-parameters used to construct the Model and are only validated here (the
    generated domain does not depend on the hyper-parameters).
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    shape = tuple(shape)
    assert variant in ("adamw", "transformer_adamw"), \
        "variant must be one of adamw / transformer_adamw"
    assert len(shape) >= 1 and min(shape) >= 1, "requires ndim 1-4 and every dimension >= 1"
    params = torch.rand(*shape, generator=gen).to(dt)
    grads = torch.randn(*shape, generator=gen).to(dt)
    exp_avg = (torch.rand(*shape, generator=gen) * 0.5).to(dt)
    exp_avg_sq = (torch.rand(*shape, generator=gen) * 0.5).to(dt)
    step_t = torch.tensor(float(step), dtype=torch.float32)
    return params, grads, exp_avg, exp_avg_sq, step_t
