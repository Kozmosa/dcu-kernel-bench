---
id: public-rmsnorm
title: RMS normalization domains
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
- rmsnorm
tags:
- rmsnorm
priority: 80
sources:
- source-torch-norm
applicability:
- RMSNorm with the configured trailing dimensions.
advice:
- Normalize using the root mean square and epsilon, without subtracting the mean.
  Retain optional elementwise weight and dtype-dependent defaults.
acceptance:
- Check the complete operator contract before timing; retain a change only after measured
  validation.
---
Inferred hypothesis: combine the scaling pass with permitted downstream elementwise work. Verify the exact domain and accumulation precision.
