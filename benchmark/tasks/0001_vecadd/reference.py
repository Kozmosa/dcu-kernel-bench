"""0001_vecadd — 独立 PyTorch 参考实现（语义唯一依据）。

评测器以本文件生成的输出为 expected。
"""


def reference(a, b):
    """out = a + b，逐元素加法。"""
    return a + b


def make_inputs(n: int, dtype: str = "float32", seed: int = 0):
    """按公开/隐藏案例描述生成输入。仅在评测端运行。"""
    import torch

    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    a = torch.randn(n, generator=gen).to(dt)
    b = torch.randn(n, generator=gen).to(dt)
    return a, b
