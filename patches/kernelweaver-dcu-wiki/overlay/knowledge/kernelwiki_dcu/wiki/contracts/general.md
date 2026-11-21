---
id: public-contract
title: Contract checklist for an uncovered operator
type: contract
knowledge_scope: public_general
origin: public_documentation
status: reviewed
confidence: inferred
reproducibility: documentation
architectures:
- gfx928
backends:
- cuda
- triton
tags:
- contract
sources:
- source-hip-language
advice:
- No specialized operator page matched. Derive semantics from the supplied task and
  request additional primary documentation before inventing an operator algorithm.
- Assume nothing about input dtype — the online check may cast to float32 while the hidden
  review uses the declared dtype. Handle the task dtype domain; take tolerances from the task.
---
Check input and output dtypes, strides, broadcasting, state mutation, indexing width, aliasing and synchronization. Validate a minimal candidate before performance work. Measure the full operation with consistent warmup and timing.

Do not assume every input is float32: the online check may cast to float32, while the hidden review supplies the dtype the task declares (float16, bfloat16, integer index tensors). A kernel that only handles float32 fails as soon as the declared dtype differs. When the task states tolerances, use them instead of a fixed 1e-4.

This page supplies a review procedure, not specialized coverage. An uncovered operator must remain visible in coverage reports.
