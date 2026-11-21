---
id: public-avgpool
title: Average pooling divisors
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
- avgpool
tags:
- avgpool
priority: 80
sources:
- source-torch-pool
applicability:
- Regular average pooling; adaptive pooling requires different window construction.
advice:
- Preserve ceil_mode, count_include_pad and any divisor_override supported by the
  API. Valid sample count and divisor need not be identical at boundaries.
acceptance:
- Check the complete operator contract before timing; retain a change only after measured
  validation.
---
Inferred hypothesis: optimize interior and boundary work with explicit guards. Check both output extent and denominator before changing the reduction mapping.
