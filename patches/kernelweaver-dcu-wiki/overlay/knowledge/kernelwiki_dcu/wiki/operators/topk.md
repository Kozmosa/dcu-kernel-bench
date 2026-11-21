---
id: public-topk
title: Top-k selection and stable ordering
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
- topk
tags:
- topk
priority: 80
sources:
- source-torch-ops
applicability:
- Top-k selection, sorting, ranking or bucketing of a per-row score vector.
advice:
- Ties are the hard part. Preserve the tie-break rule and whether the output order is sorted by
  value, by index, or unspecified.
- Preserve output shapes, the sorted flag, and whether the returned indices are global or
  row-local.
- Fractional k or k above the row length may be legal in context. Derive the effective k from
  the task instead of clamping silently.
acceptance:
- Check the complete operator contract before timing; retain a change only after measured
  validation.
---

Selection over short rows is cheaper with a per-row pass that keeps a running threshold. Selection over long rows favours a partial order or a radix pass on the value representation.

Downstream bucketing or grouping usually needs counts as well as indices. Whether counts include empty buckets is a contract decision, and an off-by-one there corrupts every consumer.

