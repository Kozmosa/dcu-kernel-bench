# 7001_ctc_loss -- CTC forward loss (hand-written alpha/beta forward-backward
# dynamic programming), a KernelBench-compatible task file.
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=7001, backend=triton.

import torch
import torch.nn as nn


class Model(nn.Module):
    """CTC (Connectionist Temporal Classification) forward loss: one forward pass
    produces both the per-sequence CTC loss and the gradient with respect to the
    input probabilities. The input probs holds **probabilities**, not logits
    (equivalent to the production library's apply_softmax=False convention: take
    the log of probs directly, no softmax), positive every frame along the class
    dimension and summing to 1; the blank class id is given by a hyperparameter
    (always 0 in the generation domain).

    Layout and input contract (time-major inputs):
      probs          (T, N, C) floating-point probability tensor, T=max_time,
                     N=batch, C=num_classes
      labels         (sum(label_lengths),) integer, the flattened label sequences
                     concatenated in batch order; label values lie in [0, C-1]
                     and are not equal to blank ([1, C-1] in the generation
                     domain)
      label_lengths  (N,) integer, target length of each sequence
      input_lengths  (N,) integer, number of valid input frames of each sequence
                     (1 <= input_lengths[n] <= T)
    Mathematical definition (for each sequence n, write Tb=input_lengths[n], the
    target l_1..l_L (L=label_lengths[n]) and the extended state sequence
    ext = [b, l_1, b, l_2, ..., b, l_L, b] (blank inserted before and after and
    between every two labels), S = 2L+1 states, s = 0..S-1; every recurrence runs
    in log space in float32, with -inf meaning unreachable):
      forward variable alpha (shape (Tb, S)):
        alpha[0,0] = log p[0, blank]; alpha[0,1] = log p[0, ext[1]] (S>1);
        everything else -inf
        alpha[t,s] = log p[t, ext[s]] + LSE( alpha[t-1,s], alpha[t-1,s-1],
                     alpha[t-1,s-2] (only when ext[s] != blank and
                     ext[s] != ext[s-2]) )
      backward variable beta (symmetric; beta does not include the emission of
      the current frame, so that alpha+beta counts every emission factor exactly
      once along each complete alignment path through state s):
        beta[Tb-1,S-1] = beta[Tb-1,S-2] = 0 (S>1), everything else -inf
        beta[t,s] = LSE( log p[t+1,ext[s]] + beta[t+1,s],
                         log p[t+1,ext[s+1]] + beta[t+1,s+1],
                         log p[t+1,ext[s+2]] + beta[t+1,s+2]
                         (the last term only when ext[s] != blank and
                         ext[s] != ext[s+2]) )
      losses[n] = -log Z, Z = LSE( alpha[Tb-1,S-1], alpha[Tb-1,S-2] )
        (the probability sum over all legal alignment paths; the reduction is the
        per-sequence 'none')
      gamma[t,s] = exp( alpha[t,s] + beta[t,s] - log Z )
      gradients[n,t,k] = d losses[n] / d probs[n,t,k]
        = -(1 / p[t,k]) * sum_{s: ext[s]=k} gamma[t,s], t < Tb; 0 for t >= Tb
    forward returns **a single 1-D tensor** out (dtype same as probs, the
    single-tensor packing protocol):
      out[0:N]         = losses (N entries)
      out[N : N+N*T*C] = gradients flattened in (N, T, C) row-major order
    Boundary behaviour:
      - the generation domain guarantees probs strictly positive (the softmax
        convention), labels without consecutive repeats and L <= Tb, so Z is
        finite with no inf/NaN; losses are on the order of <= T*ln(C), which fp16
        can represent.
      - the alpha/beta of unreachable states are -inf and the matching gamma is 0.
      - the gradient is defined with respect to the probability input (not the
        log probability): the chain factor 1/p[t,k] is already part of the
        definition.

    Input/output specification:
      probs          float32 / float16 / bfloat16 (T, N, C), contiguous
      labels / label_lengths / input_lengths   integer (...,), contiguous
      out            a single 1-D tensor, same dtype as probs, length N + N*T*C
      The evaluator casts all inputs to fp32 before passing them in (integer
      tensors must be restored to long by the implementation; their integer
      values are losslessly representable in fp32).

    Implementation constraints (violations are scored as failures):
      - Core computation (log probabilities, the alpha/beta forward-backward
        recurrence, gradient assembly) must be done with a Triton kernel inside
        the submitted file.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Intermediate computation happens in float32 (cast probs to fp32 first,
        then take the log); the output is cast back to the probs dtype and then
        concatenated into a single tensor per the packing protocol above.
      - forward returns the single tensor out (the evaluator compares a single
        output).

    Disallowed list: aiter / torch.nn.functional.ctc_loss / torch.ctc_loss /
    F.ctc_loss / torch.nn.CTCLoss / torch.ops.aten.ctc_loss /
    torch.ops.aten._ctc_loss / aten.ctc_loss / .ctc_loss( (including every
    library-level one-step CTC primitive) / warpctc / warp_ctc / torch.autograd
    and .backward() (automatic differentiation must not be used to obtain the
    gradient).
    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: single blank, reduction='none' (per-sequence loss plus the full
    gradient), the input is already a probability (no apply_softmax branch), and
    there are no empty target sequences.
    """

    def __init__(self, blank: int = 0):
        super().__init__()
        self.blank = int(blank)

    def _ctc_one(self, logp_b, p_b, target, tb, num_classes, max_time):
        # Single-sequence log-space forward-backward: logp_b / p_b are the (T, C)
        # fp32 log probabilities / probabilities, target is (L,) long, tb is the
        # number of valid frames; returns (loss_n, grad_n) with grad_n of shape
        # (max_time, C), always 0 outside the valid frames
        C = num_classes
        L = target.numel()
        S = 2 * L + 1
        neg_inf = float("-inf")
        # device=dev is required: every intermediate tensor built inside this
        # function must live on the same device as logp_b/p_b, which are already
        # on it; omitting device raises "Expected all tensors to be on the same
        # device" on CUDA/DCU (it works on CPU, so local unit tests cannot catch
        # it)
        dev = logp_b.device

        ext = torch.empty(S, dtype=torch.long, device=dev)
        ext[0::2] = self.blank
        ext[1::2] = target

        lp_ext = logp_b[:tb, ext]                      # (tb, S)
        alpha = torch.full((tb, S), neg_inf, dtype=torch.float32, device=dev)
        beta = torch.full((tb, S), neg_inf, dtype=torch.float32, device=dev)

        # Forward recurrence (stepping over the time dimension, the state
        # dimension vectorized as a whole)
        alpha[0, 0] = lp_ext[0, 0]
        alpha[0, 1] = lp_ext[0, 1]
        for t in range(1, tb):
            prev = alpha[t - 1]
            left = torch.full((S,), neg_inf, dtype=torch.float32, device=dev)
            left[1:] = prev[:-1]
            jump = torch.full((S,), neg_inf, dtype=torch.float32, device=dev)
            jump[2:] = prev[:-2]
            ok = (ext[2:] != self.blank) & (ext[2:] != ext[:-2])
            jump[2:] = torch.where(ok, jump[2:], torch.full((S - 2,), neg_inf, dtype=torch.float32, device=dev))
            alpha[t] = lp_ext[t] + torch.logsumexp(torch.stack((prev, left, jump)), dim=0)

        # Backward recurrence (beta excludes the current frame's emission:
        # alpha+beta counts every emission factor exactly once along a complete
        # path through state s, so gamma is the state posterior)
        beta[tb - 1, S - 1] = 0.0
        beta[tb - 1, S - 2] = 0.0
        for t in range(tb - 2, -1, -1):
            nxt = beta[t + 1] + lp_ext[t + 1]
            right = torch.full((S,), neg_inf, dtype=torch.float32, device=dev)
            right[:-1] = nxt[1:]
            jump = torch.full((S,), neg_inf, dtype=torch.float32, device=dev)
            jump[:-2] = nxt[2:]
            ok = (ext[:-2] != self.blank) & (ext[:-2] != ext[2:])
            jump[:-2] = torch.where(ok, jump[:-2], torch.full((S - 2,), neg_inf, dtype=torch.float32, device=dev))
            beta[t] = torch.logsumexp(torch.stack((nxt, right, jump)), dim=0)

        log_z = torch.logsumexp(alpha[tb - 1] + beta[tb - 1], dim=0)
        gamma = torch.exp(alpha + beta - log_z)        # (tb, S) state posterior

        # dL/dp[t,k] = -(1/p[t,k]) * sum_{s: ext[s]=k} gamma[t,s]
        grad = torch.zeros(tb, C, dtype=torch.float32, device=dev)
        grad.index_add_(1, ext, -gamma)
        grad = grad / p_b[:tb]                         # divide by p (probs is strictly positive)

        grad_full = torch.zeros(max_time, C, dtype=torch.float32, device=dev)
        grad_full[:tb] = grad
        return -log_z, grad_full

    def forward(self, probs, labels, label_lengths, input_lengths):
        # The online evaluator casts the integer inputs to fp32; the integer
        # values restore losslessly
        labels = labels.to(torch.long)
        label_lengths = label_lengths.to(torch.long)
        input_lengths = input_lengths.to(torch.long)

        pf = probs.to(torch.float32)
        max_time, batch, num_classes = pf.shape
        logp = torch.log(pf)

        losses = torch.empty(batch, dtype=torch.float32, device=pf.device)
        grads = torch.zeros(batch, max_time, num_classes, dtype=torch.float32, device=pf.device)
        offset = 0
        for n in range(batch):
            tb = int(input_lengths[n])
            L = int(label_lengths[n])
            target = labels[offset:offset + L]
            offset += L
            loss_n, grad_n = self._ctc_one(logp[:, n, :], pf[:, n, :], target, tb,
                                           num_classes, max_time)
            losses[n] = loss_n
            grads[n] = grad_n

        # Single-tensor packing: losses (N,) first, then gradients (N, T, C)
        # flattened in row-major order; both cast back to the probs dtype and
        # concatenated into one 1-D tensor
        losses = losses.to(probs.dtype).reshape(-1)
        grads = grads.to(probs.dtype).reshape(-1)
        return torch.cat((losses, grads))


def get_init_inputs():
    return [0]   # blank (always 0 in the generation domain, consistent with the production library's blank_label_id=0)


def get_inputs():
    # Fixed shape family (consistent with the production sample call contract);
    # the random part consumes the global RNG -- probs comes from randn passed
    # through softmax to give legal probabilities (summing to 1 along the class
    # dimension and strictly positive). The evaluator calls this function after
    # set_seed, so multiple correctness trials see diverse inputs
    max_time, batch, num_classes = 500, 4, 5
    input_lengths = [4, 100, 100, 200]
    label_lengths = [1, 2, 1, 3]
    labels = [1, 2, 3, 4, 2, 3, 2]

    probs = torch.softmax(torch.randn(max_time, batch, num_classes), dim=-1)
    return [
        probs,
        torch.tensor(labels, dtype=torch.int64),
        torch.tensor(label_lengths, dtype=torch.int64),
        torch.tensor(input_lengths, dtype=torch.int64),
    ]


def make_inputs(max_time, batch, num_classes, input_lengths, label_lengths,
                labels, blank=0, dtype="float32", seed=0):
    """Generate inputs from a public/hidden/perf case description. Runs on the
    evaluation side only (generated on CPU; the evaluator moves them to the
    device).

    The consumption order is fixed: probs (randn then softmax normalized along
    the class dimension, then cast to dtype) is the only random quantity, while
    labels / label_lengths / input_lengths are deterministic values given by the
    case and consume no randomness; bit-reproducible under the same seed. blank
    is also the init hyperparameter used to build Model, so the label domain and
    feasibility are validated here (labels != blank, no consecutive repeats,
    1 <= L <= valid frames <= max_time, num_classes >= 2).
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert max_time >= 1 and batch >= 1, "max_time >= 1 and batch >= 1 are required"
    assert num_classes >= 2, "at least blank plus 1 non-blank class are needed (num_classes >= 2)"
    assert 0 <= blank < num_classes, "blank must be a legal class id"
    assert len(input_lengths) == batch and len(label_lengths) == batch, "the length vectors must match batch"
    assert len(labels) == sum(label_lengths), "the length of labels must equal sum(label_lengths)"
    for il, ll in zip(input_lengths, label_lengths):
        assert 1 <= ll <= il <= max_time, "1 <= label_length <= input_length <= max_time is required"
    pos = 0
    for ll in label_lengths:
        seg = labels[pos:pos + ll]
        pos += ll
        for a, b in zip(seg, seg[1:]):
            assert a != b, "the generation domain guarantees labels without consecutive repeats"
        assert all(v != blank for v in seg), "labels must not equal blank"

    probs = torch.softmax(torch.randn(max_time, batch, num_classes, generator=gen), dim=-1).to(dt)
    return (
        probs,
        torch.tensor(labels, dtype=torch.int64),
        torch.tensor(label_lengths, dtype=torch.int64),
        torch.tensor(input_lengths, dtype=torch.int64),
    )
