"""The sparse index oracle must return its packed output on the input device."""
import pytest
import torch
from conftest import load_reference


@pytest.mark.parametrize('device', ['cpu', 'meta'] + (['cuda'] if torch.cuda.is_available() else []))
def test_packed_indices_follow_input_device(device):
    ref = load_reference('5007_spconv_indice_pairs')
    # tolist() needs concrete indices; the output allocation is tested on meta
    # even on hosts without a DCU/GPU.
    class Indices:
        pass
    indices = Indices()
    indices.device = torch.device(device)
    indices.to = lambda dtype: torch.tensor([[0, 0, 0, 0]], dtype=dtype)
    model = ref.Model(1, [2, 2, 2], [1, 1, 1], [1, 1, 1], [0, 0, 0], [1, 1, 1])
    output = model(indices)
    assert output.device.type == device
    assert output.dtype == torch.int32
    if device == 'cpu':
        assert output.tolist() == [0, 0, 0, 0, 0, 0, 1]
