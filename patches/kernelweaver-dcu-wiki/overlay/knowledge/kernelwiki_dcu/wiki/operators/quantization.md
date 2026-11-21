---
id: public-quantization
title: Quantization scales, rounding and bit packing
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
- quantization
tags:
- quantization
priority: 102
sources:
- source-torch-ops
applicability:
- Quantize, dequantize or quantized products with per-tensor, per-channel or per-block scales and packed low-bit dtypes.
advice:
- Scale granularity is part of the contract. Per-tensor, per-row, per-block and per-group
  differ by which elements share a scale, and a wrong grouping is only visible on comparison.
- Preserve rounding mode, saturation behaviour and the signed or unsigned range. Rounding
  half to even is not rounding half away from zero.
- Bit packing order and nibble order are layout, not arithmetic. Read the packing from the
  task and write the same order back.
- Accumulate block-scaled products in the wider dtype the contract asks for. Scaling after a
  low-precision accumulation changes the result.
acceptance:
- Check the complete operator contract before timing; retain a change only after measured
  validation.
---

Quantization is a scale and a rounding applied elementwise, plus a packing layout when values are stored below byte granularity. Dequantization resurrects a scale per group, and a quantized product moves the scale out of the inner loop.

Two facts decide most implementations: which elements share a scale, and where the scale multiply happens. Deriving either from another library's layout is a common source of silent mismatch.

