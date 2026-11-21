---
id: public-reduction
title: Reduction axes and layout
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
- reduction
tags:
- reduction
priority: 60
sources:
- source-torch-ops
- source-hip-language
applicability:
- Sum, mean, minimum and maximum reductions; scans require a different algorithm.
advice:
- Preserve reduced axes, retained dimensions and dtype rules. Read actual strides
  before assigning lanes to input or output axes.
acceptance:
- Check the complete operator contract before timing; retain a change only after measured
  validation.
---
Inferred hypothesis: compare coalesced mappings and hierarchical partial reductions. Check identity values, tails and accumulation error. A block barrier cannot coordinate independent blocks.
