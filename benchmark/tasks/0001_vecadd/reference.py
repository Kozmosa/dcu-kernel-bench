"""0001_vecadd -- standalone PyTorch reference implementation (the sole
semantic authority).

The evaluator uses the output produced by this file as `expected`.
"""


def reference(a, b):
    """out = a + b, an element-wise addition."""
    return a + b


def make_inputs(n: int, dtype: str = "float32", seed: int = 0):
    """Generate inputs from a public/hidden case description.

    Runs on the evaluation side only.
    """
    import torch

    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    a = torch.randn(n, generator=gen).to(dt)
    b = torch.randn(n, generator=gen).to(dt)
    return a, b
