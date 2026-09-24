"""0001_vecadd — reference 语义与输入生成器测试。"""
import torch

from conftest import load_reference

ref = load_reference("0001_vecadd")


def test_reference_adds_elementwise():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([10.0, 20.0, 30.0])
    out = ref.reference(a, b)
    assert torch.equal(out, torch.tensor([11.0, 22.0, 33.0]))


def test_make_inputs_shape_dtype_and_determinism():
    a1, b1 = ref.make_inputs(1024, "float16", seed=7)
    a2, b2 = ref.make_inputs(1024, "float16", seed=7)
    assert a1.shape == b1.shape == (1024,)
    assert a1.dtype == b1.dtype == torch.float16
    assert torch.equal(a1, a2) and torch.equal(b1, b2)


def test_make_inputs_seed_changes_data():
    a1, _ = ref.make_inputs(64, "float32", seed=1)
    a2, _ = ref.make_inputs(64, "float32", seed=2)
    assert not torch.equal(a1, a2)
