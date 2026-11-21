---
id: public-normalization
title: Normalization statistics and state
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
- normalization
tags:
- normalization
priority: 96
sources:
- source-torch-norm
- source-torch-batchnorm
applicability:
- Batch, layer, group, instance, RMS or softmax-style normalization expressed as a statistic followed by a rescale.
advice:
- Establish which axes the statistic reduces over, whether it is mean and variance,
  mean square only, or a softmax denominator, and which tensors carry state.
- Training-mode batch statistics update running buffers; inference reads them. Preserve
  momentum, epsilon, affine parameters and the update itself, and do not silently drop a
  buffer mutation because the forward output happened to match.
- Group and instance normalization map work to per-sample, per-group domains. A
  batch-wide statistic mixes independent samples and changes the operator.
- Accumulation order changes the result. Validate the variance construction and the
  rescale against the reference before changing either.
acceptance:
- Check the complete operator contract before timing; retain a change only after measured
  validation.
---

The statistic and the rescale are two passes unless the whole reduced domain fits a
block. When it does not fit, either reduce per block and combine partials with an
explicit second stage, or keep the reduced domain inside one block and stream the other
axes. Both are workload decisions, not correctness guarantees.

RMS-style normalization has no mean subtraction and no variance; reusing a variance
pipeline for it changes the result. Instance and group normalization differ only in which
axes belong to one reduction domain.

