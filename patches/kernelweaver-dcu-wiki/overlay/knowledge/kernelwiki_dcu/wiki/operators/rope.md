---
id: public-rope
title: Rotary position embedding and pairing convention
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
- rope
tags:
- rope
priority: 108
sources:
- source-torch-ops
applicability:
- Rotary embedding applied to query and key tensors, including decode-time and concatenated query-key layouts.
advice:
- The pairing convention is part of the operator contract. Adjacent pairs, interleaved halves
  and duplicated halves give different results for the same table, so read it from the task rather
  than from memory of another implementation.
- Position and angle tables may arrive precomputed, may be indexed by position ids, or may
  be built inside the operator. Preserve which of these the task asks for, and preserve
  frequency scaling and offset exactly.
- Interleaved output layouts require the write pattern to match the read pattern; a wrong
  permutation is numerically plausible and fails only on comparison.
acceptance:
- Check the complete operator contract before timing; retain a change only after measured
  validation.
---

Rotary embedding is an elementwise rotation of paired channels plus a write back to the same layout. The general cost is memory traffic, not arithmetic, so reuse of the angle table and coalesced reads dominate.

When query and key are concatenated or stacked, apply the convention per tensor and keep the segment boundaries of the packed output aligned with the contract.

