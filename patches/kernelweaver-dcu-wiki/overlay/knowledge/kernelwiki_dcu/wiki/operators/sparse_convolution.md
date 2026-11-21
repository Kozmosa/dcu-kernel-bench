---
id: public-sparse-convolution
title: Sparse convolution and indice pairing
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
- sparse_convolution
tags:
- sparse-conv
priority: 78
sources:
- source-torch-conv
- source-torch-ops
applicability:
- Sparse or submanifold convolution over an explicit coordinate list, including indice-pair construction and transposed sparse convolution.
advice:
- The coordinate rule is the contract. Whether the pairing keeps positions whose neighbours
  are all active, or only positions active in the input, decides the output extent.
- Indice pairs are a gather pattern plus a count. Preserve their ordering and the counted
  total, and keep empty kernel positions as explicit empty ranges rather than dropping them.
- Gather and scatter passes must agree on the same pairing. Validate that the scattered
  result has no unwritten position before adding a bias or an activation.
acceptance:
- Check the complete operator contract before timing; retain a change only after measured
  validation.
---

Sparse convolution replaces dense spatial iteration with a coordinate list and a precomputed pairing. The relevant cost is building and reusing that pairing, not the arithmetic.

Submanifold versus stride-2 sparse convolution differ only in the pairing rule, so a shared kernel with a different builder is usually correct. Validate the empty-input and single-point cases.

