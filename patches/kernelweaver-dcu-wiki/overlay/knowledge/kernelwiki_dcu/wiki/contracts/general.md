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
- Inputs arrive in their native dtype and shape envelope; do not assume they were cast
  to float32, and take the tolerance from the task contract instead of a fixed 1e-4.
---
Check input and output dtypes, strides, broadcasting, state mutation, indexing width, aliasing and synchronization. Validate a minimal candidate before performance work. Measure the full operation with consistent warmup and timing.

Inputs arrive in their native dtype — float32, float16, bfloat16 and integer index tensors are all supplied as declared — so a kernel that silently assumes float32 fails on the first case whose dtype differs. The tolerance, and any per-segment tolerance, come from the task contract rather than a fixed 1e-4.

This page supplies a review procedure, not specialized coverage. An uncovered operator must remain visible in coverage reports.
