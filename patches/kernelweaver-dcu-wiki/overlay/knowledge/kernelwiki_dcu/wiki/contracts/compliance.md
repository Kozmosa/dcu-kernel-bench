---
knowledge_scope: public_general
origin: public_documentation
status: reviewed
confidence: inferred
reproducibility: documentation
id: public-compliance
title: Compliance with the task forbidden-operator list
type: contract
architectures:
- gfx936
backends:
- cuda
- triton
sources:
- source-hip-language
- source-torch-ops
tags:
- contract
- compliance
priority: 120
advice:
- Implement the operator's arithmetic yourself; a candidate that reaches torch.matmul, torch.softmax,
  torch.mm, F.linear, conv2d or torch.cumsum anywhere in the compute path is rejected by the static audit
  even when the online check passes. Import aliases are resolved, so nn.functional / F spellings do not hide it.
- Do not carry CUDA runtime calls, CUDA headers or CUDART_* constants into the kernel; only the hip spelling of
  those APIs exists on this target.
- When a hand-written kernel fails twice on the same pattern, change the decomposition instead of retrying
  it, and never revert to a blocked call as the fallback.
- Validate the replacement against the reference before optimizing.
applicability:
- Every model_class task; the task docstring's forbidden list is authoritative when it is stricter.
---

The static audit runs before execution on both tracks; numerics are checked on one shape afterwards, so a
correct candidate can still be rejected. Matching resolves import aliases (`nn.functional.mm`, `F.linear`),
skips docstrings, comments and prose, and inspects only code reachable from the model entry point: an unused
helper is ignored, a called one is not.

| Blocked call | Implement instead |
|---|---|
| `torch.matmul`, `torch.mm`, `a @ b` | tiled product: `tl.dot` in Triton, shared-memory tiling in HIP |
| `torch.softmax`, `F.softmax` | in-kernel row reduction: max, exp, sum, divide |
| `F.linear`, `nn.Linear` | explicit product plus bias |
| `conv2d`, `F.conv2d` | im2col plus product, or a loop over filter taps |
| `torch.cumsum` | block scan (Hillis-Steele or Blelloch) with a carried offset |
| `torch.autograd` | hand-written backward kernel |

Platform symbols are rejected separately: CUDA runtime types and calls (`cudaMalloc`, `cudaStream_t`,
`cudaMemcpy`), CUDA headers such as cuda_runtime.h, CUDA math constants (`CUDART_*`) and PTX inline assembly
do not exist on this target; use the hip equivalents, `INFINITY`, or a libdevice function. On this stack
`at::cuda::`, `torch.cuda.` and ATen's cuda headers are the correct spelling and stay allowed.

**When a hand-written kernel keeps failing, change the plan instead of retrying it.** Two failures on the same
pattern is a signal, not bad luck:

- decompose differently: smaller tiles, one pass per output segment, an explicit loop over blocks or taps;
- if a fused kernel will not compile, split it into two kernels and fuse later;
- re-read the shape envelope: a kernel tuned to block-size multiples fails the hidden shapes;
- never revert to a blocked call: it turns a correct candidate into a rejected one.
