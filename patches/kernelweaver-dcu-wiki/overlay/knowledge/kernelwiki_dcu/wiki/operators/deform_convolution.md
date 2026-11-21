---
id: public-deform-convolution
title: Deformable convolution and interpolation
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
- deform_convolution
tags:
- deform-conv
priority: 90
sources:
- source-torch-conv
- source-torch-ops
applicability:
- Convolution or attention whose sampling coordinates come from learned offsets, with or without a mask.
advice:
- Bilinear interpolation weights and the out-of-bounds policy define the operator. Preserve
  the zero padding rule, the clamp rule and the fractional coordinate convention.
- Offsets may be packed per kernel position per channel. Read the packing order rather than
  assuming an interleaving.
- The backward pass sums gradient contributions through the same interpolation weights.
  Reusing the forward coordinate buffer is usually cheaper than recomputing them.
acceptance:
- Check the complete operator contract before timing; retain a change only after measured
  validation.
---

A deformable convolution turns each output position into a small set of fractional source coordinates, gathers four neighbours with bilinear weights and accumulates the filter taps. Throughput is limited by the irregular gathers, so coordinate reuse and vectorised neighbour loads matter more than the arithmetic.

The mask, when present, multiplies each tap. Whether it is applied before or after the interpolation changes nothing mathematically only when the mask is per position.

