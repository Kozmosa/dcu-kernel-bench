---
id: public-indexing
title: Gather and scatter index semantics
type: kernel
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
operator_families:
- indexing
tags:
- indexing
priority: 75
sources:
- source-torch-ops
applicability:
- Gather, scatter, index-select or index-add over an explicit index tensor.
advice:
- Index tensors are integers. Do not route them through floating-point arithmetic, and do not
  assume they are non-negative or in range unless the contract says so.
- Repeated destinations are the interesting case. Scatter semantics decide whether one value
  wins, they accumulate, or the contract forbids duplicates. Read it, do not assume an
  atomic add.
- Preserve the output shape that follows the index shape rather than the source shape, and
  preserve broadcasting of higher-rank index layouts.
acceptance:
- Check the complete operator contract before timing; retain a change only after measured
  validation.
---

Gather reads through an index, scatter writes through one. Both are memory-bound, so coalesced index order matters more than arithmetic, and both are sensitive to index layout because neighbouring outputs then reach scattered addresses.

When the same destination can be written more than once, the ordering guarantee decides the implementation. Serialising conflicting positions is correct but slow; an atomic update is fast but changes the result when the contract expects one winner.

