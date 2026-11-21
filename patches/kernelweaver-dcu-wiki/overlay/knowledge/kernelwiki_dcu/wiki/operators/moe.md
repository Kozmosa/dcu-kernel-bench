---
id: public-moe
title: Mixture-of-experts dispatch and combine
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
- moe
tags:
- moe
priority: 115
sources:
- source-torch-ops
applicability:
- Expert routing, token permutation, grouped products and weighted combination in one operator or in an end-to-end pipeline.
advice:
- Routing selects tokens, and selection is not a value transformation. Preserve tie-breaking,
  the number of selected experts per token, whether the selection is normalized, and whether
  dropped tokens keep a zero contribution or leave the count unchanged.
- The permutation and its inverse must cancel. Write the inverse from the same index buffer
  rather than recomputing it, and validate that every token lands in exactly one slot.
- Grouped work has ragged sizes. Guard every boundary against the actual per-expert counts,
  and do not assume the padded layout of another implementation.
- Combine weights may be float32 while the payload is lower precision. Accumulate in the
  dtype the contract asks for.
acceptance:
- Check the complete operator contract before timing; retain a change only after measured
  validation.
---

A dispatch-and-combine operator is a gather, a per-group product, and a weighted scatter-add. Each stage is simple, and the usual failures are in the index bookkeeping and the empty-group case rather than in arithmetic.

Alignment and padding appear when groups are processed in fixed tiles. Whether padding rows participate in the reduction is a contract decision. Validate the all-experts-empty and single-token cases explicitly.

