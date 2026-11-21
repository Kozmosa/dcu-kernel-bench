---
id: public-ctc-loss
title: Connectionist temporal classification loss
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
- ctc_loss
tags:
- ctc-loss
priority: 84
sources:
- source-pytorch-functional
- source-torch-loss
applicability:
- CTC or similar alignment-free sequence loss with an integer label tensor and per-sample input lengths.
advice:
- Blank label placement, label length and the target-length convention decide the alignment
  lattice. Read all three from the contract, and keep them out of the inner loop.
- The log-domain recurrence must stay in the log domain. Summing probabilities and then
  taking a logarithm underflows on long sequences.
- Preserve zero-infinity handling for impossible paths, the reduced loss form the contract
  asks for, and the gradient convention when a backward is required.
acceptance:
- Check the complete operator contract before timing; retain a change only after measured
  validation.
---

The forward recursion over the expanded target lattice is a small dynamic program per sample, with a log-sum-exp combine. The work is negligible compared with the arithmetic of a matmul, so the usual failure is indexing the lattice state, not throughput.

Validate with an all-blank target, a target longer than the input and a repeated-label target, all of which stress the blank rule.

