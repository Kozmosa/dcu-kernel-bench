import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark/evaluator'))
from audit_batch import run_one, cache_signature


def test_no_candidate_and_missing_task_stay_failures(tmp_path):
    missing = run_one('1002_paged_attention', tmp_path / 'best_code.py', 1, True)
    assert not missing['ok'] and missing['failure_kind'] == 'missing_submission'
    (tmp_path / 'run.json').write_text(json.dumps({'best_node_id': None, 'nodes': {}}))
    no_candidate = run_one('1002_paged_attention', tmp_path / 'best_code.py', 1, True)
    assert not no_candidate['ok'] and no_candidate['failure_kind'] == 'no_valid_candidate'


def test_root_cannot_pass_offline_even_if_code_is_numerically_correct(tmp_path):
    (tmp_path / 'best_code.py').write_text('class ModelNew: pass')
    (tmp_path / 'run.json').write_text(json.dumps({'best_node_id': 'root', 'nodes': {'root': {'correct': True}}}))
    result = run_one('1002_paged_attention', tmp_path / 'best_code.py', 1, True)
    assert not result['ok'] and result['failure_kind'] == 'no_valid_candidate'


def test_cache_invalidated_when_selected_candidate_changes(tmp_path):
    code = tmp_path / 'best_code.py'
    code.write_text('old')
    before = cache_signature(code)
    code.write_text('new')
    assert cache_signature(code) != before


def test_all_missing_manifest_tasks_remain_in_denominator(tmp_path):
    manifest = tmp_path / 'manifest.yaml'
    manifest.write_text('tasks:\n  - alias: 1002_paged_attention\n    level: 3\n    problem_id: 1002\n')
    root = tmp_path / 'runs'
    root.mkdir()
    script = Path(__file__).resolve().parents[1] / 'benchmark/evaluator/audit_batch.py'
    result = subprocess.run([sys.executable, str(script), '--run-root', str(root),
                             '--manifest', str(manifest), '--skip-perf'], capture_output=True, text=True)
    assert result.returncode == 1, result.stdout + result.stderr
    summary = json.loads((root / 'offline_audit.json').read_text())
    assert summary['audited'] == 1 and summary['aggregates']['correctness_passed'] == 0
    assert summary['aggregates']['missing_submission'] == 1


def test_cached_failure_still_returns_failure_exit_code(tmp_path):
    manifest = tmp_path / 'manifest.yaml'
    manifest.write_text('tasks:\n  - alias: 1002_paged_attention\n    level: 3\n    problem_id: 1002\n')
    root = tmp_path / 'runs'
    root.mkdir()
    script = Path(__file__).resolve().parents[1] / 'benchmark/evaluator/audit_batch.py'
    command = [sys.executable, str(script), '--run-root', str(root), '--manifest', str(manifest), '--skip-perf']
    for _ in range(2):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 1, result.stdout + result.stderr
