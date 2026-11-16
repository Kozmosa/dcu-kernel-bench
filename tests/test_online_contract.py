"""CPU probes of the real public-evaluation path, no GPU or model calls."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark/evaluator'))
from dcukb_online import evaluate, prepare_task, public_trials
from audit_model_class import compare_outputs

REFERENCE = '''import torch
class Model(torch.nn.Module):
    def forward(self, x, indices):
        assert x.dtype == torch.float16
        assert indices.dtype == torch.int32
        return x + 1

def get_init_inputs(): return []
def get_inputs():
    return [torch.randn(8, dtype=torch.float16), torch.arange(8, dtype=torch.int32)]
'''
CANDIDATE = '''import torch
class ModelNew(torch.nn.Module):
    def forward(self, x, indices):
        assert x.dtype == torch.float16
        assert indices.dtype == torch.int32
        return x + 1
'''


def factories():
    def success(runtime, **kwargs):
        return SimpleNamespace(correct=True, runtime=runtime, logs=[], **kwargs)
    def failure(stage, kind, *logs):
        return SimpleNamespace(correct=False, failure_stage=stage, failure_type=kind, logs=list(logs))
    return success, failure


@pytest.fixture
def contract(tmp_path, monkeypatch):
    task_dir = tmp_path / 'benchmark/tasks/9000_probe'
    task_dir.mkdir(parents=True)
    (task_dir / 'task.yaml').write_text(yaml.safe_dump({
        'entry': 'model_class', 'io': {'inputs': [{'dtype': ['float16', 'int32']}]},
        'tolerance': {'float16': {'atol': 0.01, 'rtol': 0.01}},
        'forbidden': ['torch.matmul'],
    }))
    monkeypatch.setenv('DCUKB_ROOT', str(tmp_path))
    task = SimpleNamespace(source_origin='/mirror/level1/9000_probe.py', problem_id=9000,
                           source_code='# scaffold', reference_code=REFERENCE, description='probe')
    config = SimpleNamespace(seed=7, num_correct_trials=2, paper_num_warmup=0, num_perf_trials=2)
    return task, config, task_dir


def test_native_public_dtypes_and_rules(contract):
    task, config, _ = contract
    prepare_task(task)
    assert 'NOT uniformly converted to fp32' in task.description
    assert 'torch.matmul' in task.description
    assert 'private' not in task.description
    result = evaluate(task, CANDIDATE, config, *factories(), device='cpu')
    assert result.correct, result.logs
    report = json.loads(result.logs[-1])
    assert report['trials'][0]['input_dtypes'] == ['torch.float16', 'torch.int32']
    assert [t['seed'] for t in report['trials']] == [7, 8]


def test_online_static_check_precedes_candidate_execution(contract):
    task, config, _ = contract
    code = CANDIDATE.replace('return x + 1', 'return torch.matmul(x, x)')
    code += '\nraise AssertionError("must not execute")\n'
    result = evaluate(task, code, config, *factories(), device='cpu')
    assert not result.correct and result.failure_type == 'static_audit_failed'


def test_wrong_output_dtype_fails_even_when_values_match(contract):
    task, config, _ = contract
    result = evaluate(task, CANDIDATE.replace('return x + 1', 'return (x + 1).float()'),
                      config, *factories(), device='cpu')
    assert not result.correct
    assert 'shape-dtype-device' in result.logs[0]


def test_task_tolerance_replaces_hardcoded_1e4(contract):
    task, config, _ = contract
    result = evaluate(task, CANDIDATE.replace('return x + 1', 'return x + 1 + 0.005'),
                      config, *factories(), device='cpu')
    assert result.correct, result.logs


def test_no_hidden_assets_are_read(contract, monkeypatch):
    original = Path.read_text
    def read(path, *args, **kwargs):
        assert 'private' not in path.parts
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    task, config, _ = contract
    assert evaluate(task, CANDIDATE, config, *factories(), device='cpu').correct


def test_oracle_and_candidate_receive_independent_input_buffers(contract):
    task, config, _ = contract
    task.reference_code = REFERENCE.replace('return x + 1', 'x.add_(1)\n        return x')
    assert evaluate(task, CANDIDATE, config, *factories(), device='cpu').correct


def test_integer_contract_does_not_round_away_errors():
    expected = torch.tensor([2**24], dtype=torch.int32)
    actual = torch.tensor([2**24+1], dtype=torch.int32)
    ok, _ = compare_outputs(actual, expected, {'int32': {'atol': 0, 'rtol': 0}}, 'int32')
    assert not ok


def test_nan_is_not_covered_by_mismatch_budget():
    expected = torch.zeros(1000)
    actual = expected.clone()
    actual[0] = float('nan')
    ok, detail = compare_outputs(actual, expected,
                                {'float32': {'atol': 0, 'rtol': 0, 'max_mismatch_frac': 0.01}}, 'float32')
    assert not ok and detail['mode'] == 'nonfinite'
