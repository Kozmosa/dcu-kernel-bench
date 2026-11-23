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
- gfx928
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
- The scaffold is built from the reference forward, so it can still carry a blocked call copied from the
  reference (a torch.softmax statement, for example). The reference is allowed to use those APIs, the
  submission is not — replace every such call with your own kernel arithmetic before submitting.
- Validate the replacement against the reference before optimizing.
applicability:
- Every model_class task; the task docstring's forbidden list is authoritative when it is stricter.
---

A static audit inspects the submitted file — offline here, before execution where the evaluation
bridge is installed. Numerics are checked on one shape only, so a correct candidate can still fail. Matching resolves import aliases (`nn.functional.mm`, `F.linear`),
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

Platform symbols are rejected too: CUDA runtime calls (`cudaMalloc`, `cudaMemcpy`, `cudaStream_t`), CUDA
headers, `CUDART_*` constants and PTX assembly do not exist here; use the hip spelling or `INFINITY`.
On this stack `at::cuda::`, `torch.cuda.` and ATen cuda headers are correct and allowed.

**When a hand-written kernel keeps failing, change the plan, not the retry.** Two failures on the same
pattern is a signal:

- decompose differently: smaller tiles, one pass per output segment, an explicit loop over blocks or taps;
- if a fused kernel will not compile, split it into two kernels and fuse later;
- re-read the shape envelope: a kernel tuned to block-size multiples fails the hidden shapes;
- never revert to a blocked call: it turns a correct candidate into a rejected one.
