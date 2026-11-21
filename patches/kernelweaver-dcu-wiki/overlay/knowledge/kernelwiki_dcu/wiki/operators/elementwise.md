---
id: public-elementwise
title: Elementwise activation expressions
type: kernel
knowledge_scope: public_general
origin: public_documentation
status: reviewed
confidence: inferred
reproducibility: documentation
architectures:
- gfx936
backends:
- cuda
- triton
operator_families:
- elementwise
tags:
- elementwise
priority: 40
sources:
- source-torch-activation
- source-hip-language
applicability:
- Elementwise expressions including activation compositions; inspect broadcasting
  and aliasing.
advice:
- Keep slopes, approximation modes, in-place requirements and dtype conversions. SiLU
  is the input multiplied by its sigmoid.
acceptance:
- Check the complete operator contract before timing; retain a change only after measured
  validation.
---
Inferred hypothesis: evaluate fusion within the permitted expression to avoid intermediate traffic. Coalesced independent outputs are a starting point. Vectorization needs alignment and remainder handling; shared staging needs demonstrable reuse.
