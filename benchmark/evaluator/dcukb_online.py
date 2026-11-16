"""DCUKB public evaluation bridge for KernelWeaver and PyramidKernel.

Only public reference/get_inputs and task.yaml are read here. Hidden cases stay
in the offline evaluator. Install the matching framework patch and set
DCUKB_ROOT to this checkout; an explicit contract never silently falls back to
the upstream fp32/1e-4 evaluator.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
import threading

import torch
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from audit_model_class import check_close, clone_inputs, compare_outputs, load_module, load_tolerances
from static_audit import GLOBAL_FORBIDDEN, audit_file, load_task_forbidden


def task_directory(task) -> Path:
    root = Path(os.environ['DCUKB_ROOT']).resolve()
    # Names stay stable in the HIP/Triton mirror trees. Require an exact task
    # match as well as the public model_class declaration.
    stem = Path(task.source_origin).stem
    directory = root / 'benchmark' / 'tasks' / stem
    spec = yaml.safe_load((directory / 'task.yaml').read_text(encoding='utf-8'))
    if spec.get('entry') != 'model_class' or int(stem.split('_', 1)[0]) != task.problem_id:
        raise ValueError(f'Invalid DCUKB public task: {stem}')
    return directory


def prompt_contract(task) -> str:
    spec = yaml.safe_load((task_directory(task) / 'task.yaml').read_text(encoding='utf-8'))
    public = {key: spec[key] for key in ('io', 'tolerance', 'tolerance_segments', 'forbidden') if key in spec}
    return ('DCUKB evaluation contract (authoritative): preserve native input dtypes; '
            'inputs are NOT uniformly converted to fp32. Output shape, dtype and device '
            'must match Model. Use the task tolerances below, NOT a fixed 1e-4. '
            'Support the declared dtype/shape domain. Core computation must use your '
            'own kernels; static compliance is checked before execution.\n'
            + json.dumps(public, ensure_ascii=True, sort_keys=True))


def prepare_task(task):
    """Add only public rules to both structured-agent and direct-code prompts."""
    contract = prompt_contract(task)
    task.description += '\n' + contract
    # A comment is preserved in scaffolds and direct source prompts; it does not
    # expose hidden cases or change code execution.
    task.source_code = '# ' + contract.replace('\n', '\n# ') + '\n' + task.source_code
    return task


def _load_source(code: str, directory: Path, name: str):
    path = directory / (name + '.py')
    path.write_text(code, encoding='utf-8')
    return load_module(path, name)


def public_trials(task_dir, ref_mod, cand_mod, seed, count, device):
    """Shared by online evaluation and offline --public cross-checks."""
    tol = load_tolerances(task_dir)
    failures, records = [], []
    for index in range(max(1, count)):
        trial_seed = seed + index
        try:
            torch.manual_seed(trial_seed)
            init = ref_mod.get_init_inputs()
            ref_model = ref_mod.Model(*init).to(device)
            torch.manual_seed(trial_seed)
            cand_model = cand_mod.ModelNew(*init).to(device)
            # Preserve reference parameters when a candidate retains nn layers.
            state = cand_model.state_dict()
            shared = {k: v for k, v in ref_model.state_dict().items()
                      if k in state and v.shape == state[k].shape}
            cand_model.load_state_dict(shared, strict=False)
            torch.manual_seed(trial_seed)
            inputs = clone_inputs(ref_mod.get_inputs(), device=device)
            # Snapshot diagnostics before either implementation can mutate data.
            input_dtypes = [str(x.dtype) if isinstance(x, torch.Tensor) else type(x).__name__ for x in inputs]
            case_dtype = next((str(x.dtype).replace('torch.', '') for x in inputs
                               if isinstance(x, torch.Tensor) and x.is_floating_point()), '')
            expected = ref_model(*clone_inputs(inputs))
            actual = cand_model(*clone_inputs(inputs))
            if torch.device(device).type == 'cuda':
                torch.cuda.synchronize()
            names = [p for p in __import__('inspect').signature(ref_mod.Model.__init__).parameters if p != 'self']
            kwargs = dict(zip(names, init))
            ok, detail = compare_outputs(actual, expected, tol, case_dtype,
                                         ref_mod=ref_mod, init_kwargs=kwargs,
                                         task_dir=task_dir, inputs=inputs)
            record = {'seed': trial_seed, 'input_dtypes': input_dtypes, 'detail': detail}
            records.append(record)
            if not ok:
                failures.append(record)
        except Exception as exc:
            record = {'seed': trial_seed, 'error_type': type(exc).__name__, 'error': str(exc)}
            records.append(record)
            failures.append(record)
    return {'passed': not failures, 'case_count': max(1, count), 'failures': failures, 'trials': records}


def _runtime(model, inputs, config, device):
    warmup = max(0, int(getattr(config, 'paper_num_warmup', 5)))
    count = max(1, int(getattr(config, 'num_perf_trials', 20)))
    # Inputs are cloned outside the timed region: an in-place optimizer must
    # see the same initial values for every measurement.
    for _ in range(warmup):
        model(*clone_inputs(inputs))
    times = []
    for _ in range(count):
        args = clone_inputs(inputs)
        if torch.device(device).type == 'cuda':
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            model(*args)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / 1000.0)
        else:
            start = time.perf_counter()
            model(*args)
            times.append(time.perf_counter() - start)
    return statistics.median(times)


_EVALUATION_LOCK = threading.RLock()


def evaluate(task, code, config, success, failure, *, device=None):
    with _EVALUATION_LOCK:
        return _evaluate(task, code, config, success, failure, device=device)


def _evaluate(task, code, config, success, failure, *, device=None):
    """Return the caller framework's EvaluationResult via its own factories."""
    try:
        task_dir = task_directory(task)
    except Exception as exc:
        return failure('compile', 'dcukb_contract_error', str(exc))
    with tempfile.TemporaryDirectory(prefix='dcukb_eval_') as tmp:
        tmp = Path(tmp)
        path = tmp / 'candidate.py'
        path.write_text(code, encoding='utf-8')
        # The scaffold is measured only as an oracle/baseline. The root patch
        # separately forbids it from becoming a deliverable candidate.
        is_oracle = code == task.source_code
        if not is_oracle:
            patterns = sorted(set(GLOBAL_FORBIDDEN) | set(load_task_forbidden(task_dir)))
            hits = audit_file(path, patterns)
            if hits:
                return failure('compile', 'static_audit_failed', json.dumps(hits, ensure_ascii=True))
        if device is None:
            if not torch.cuda.is_available():
                return failure('runtime', 'runtime_error', 'DCUKB evaluation requires a CUDA/HIP device')
            device = torch.device(f'cuda:{torch.cuda.current_device()}')
        try:
            with torch.no_grad():
                ref_mod = _load_source(task.reference_code, tmp, 'dcukb_reference')
                cand_mod = _load_source(code, tmp, 'dcukb_candidate')
                report = public_trials(task_dir, ref_mod, cand_mod, config.seed,
                                       config.num_correct_trials, device)
                if not report['passed']:
                    return failure('correctness', 'dcukb_public_correctness', json.dumps(report, ensure_ascii=True))
                torch.manual_seed(config.seed)
                init = ref_mod.get_init_inputs()
                ref_model = ref_mod.Model(*init).to(device)
                torch.manual_seed(config.seed)
                cand_model = cand_mod.ModelNew(*init).to(device)
                state = cand_model.state_dict()
                cand_model.load_state_dict({k: v for k, v in ref_model.state_dict().items()
                                            if k in state and v.shape == state[k].shape}, strict=False)
                inputs = clone_inputs(ref_mod.get_inputs(), device=device)
                runtime = _runtime(cand_model, inputs, config, device)
                reference = _runtime(ref_model, inputs, config, device)
                result = success(runtime, reference_runtime=reference, speedup=reference / runtime,
                                 reference_runtimes={'torch_eager': reference},
                                 speedups={'torch_eager': reference / runtime}, primary_reference='torch_eager')
                result.logs.extend(['dcukb_contract=native_v1', 'dcukb_role=' + ('oracle' if is_oracle else 'candidate'),
                                    json.dumps(report, ensure_ascii=True)])
                return result
        except Exception as exc:
            return failure('runtime', 'dcukb_execution_error', f'{type(exc).__name__}: {exc}')
