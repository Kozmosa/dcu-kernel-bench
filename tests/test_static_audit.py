"""static_audit.py 回归：starter 必须通过；作弊提交必须被抓。"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "benchmark" / "evaluator" / "static_audit.py"


def run_audit(task: str, files: list[Path]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(AUDIT), str(REPO / "benchmark" / "tasks" / task), *map(str, files)],
        capture_output=True, text=True,
    )


def test_starters_pass_audit():
    for task in ("0001_vecadd", "1001_paged_attention"):
        starter = next((REPO / "benchmark" / "tasks" / task / "starter").glob("*"))
        r = run_audit(task, [starter])
        assert r.returncode == 0, r.stdout + r.stderr


def test_forbidden_call_is_rejected(tmp_path: Path):
    cheat = tmp_path / "cheat.py"
    cheat.write_text("def f(x):\n    return torch.matmul(x, x)\n", encoding="utf-8")
    r = run_audit("1001_paged_attention", [cheat])
    assert r.returncode == 1
    assert "torch.matmul" in r.stdout


def test_commented_lib_is_not_flagged(tmp_path: Path):
    ok = tmp_path / "ok.py"
    ok.write_text("# 我们不用 rocblas\nvalue = 1\n", encoding="utf-8")
    r = run_audit("1001_paged_attention", [ok])
    assert r.returncode == 0, r.stdout


def test_docstring_mentioning_forbidden_is_not_flagged(tmp_path: Path):
    # model_class scaffold 的 docstring 逐字列出禁用项（规则声明），不得误杀
    src = tmp_path / "scaffold_like.py"
    src.write_text(
        'class ModelNew:\n'
        '    """约束：禁止调用 torch.matmul / torch.einsum / aiter。"""\n'
        "    def forward(self, x):\n"
        "        return x + 1\n",
        encoding="utf-8",
    )
    r = run_audit("1002_paged_attention", [src])
    assert r.returncode == 0, r.stdout


def test_real_call_inside_docstringed_file_still_flagged(tmp_path: Path):
    src = tmp_path / "mixed.py"
    src.write_text(
        '"""docstring 提到 torch.matmul 仅说明规则"""\n'
        "def f(x, y):\n"
        "    return torch.matmul(x, y)\n",
        encoding="utf-8",
    )
    r = run_audit("1002_paged_attention", [src])
    assert r.returncode == 1
    assert "torch.matmul" in r.stdout


def test_user_kernel_name_is_not_a_library_call(tmp_path):
    src = tmp_path / 'adamw.py'
    src.write_text('def _fused_adamw_kernel(x):\n    return x\n'
                   'class ModelNew:\n    def forward(self, x):\n        return _fused_adamw_kernel(x)\n')
    assert run_audit('6001_fused_adamw', [src]).returncode == 0


def test_dead_reference_helpers_are_not_candidates(tmp_path):
    src = tmp_path / 'candidate.py'
    src.write_text('import torch\n'
                   'def reference(x):\n    return torch.softmax(x, -1)\n'
                   'class ModelNew:\n    def unused(self, x):\n        return reference(x)\n'
                   '    def forward(self, x):\n        return x\n')
    assert run_audit('1002_paged_attention', [src]).returncode == 0


def test_reachable_helpers_and_import_aliases_are_rejected(tmp_path):
    src = tmp_path / 'candidate.py'
    src.write_text('from torch import matmul as mm\n'
                   'def helper(x):\n    return mm(x, x)\n'
                   'class ModelNew:\n    def work(self, x):\n        return helper(x)\n'
                   '    def forward(self, x):\n        return self.work(x)\n')
    r = run_audit('1002_paged_attention', [src])
    assert r.returncode == 1 and 'torch.matmul' in r.stdout


def test_method_forbidden_pattern_and_alias(tmp_path):
    src = tmp_path / 'candidate.py'
    src.write_text('import torch as th\nclass ModelNew:\n'
                   '    def forward(self, x):\n        return th.cat([x, x])\n')
    assert run_audit('2004_fused_qk_concat', [src]).returncode == 1


def test_embedded_hip_library_call_is_rejected(tmp_path):
    src = tmp_path / 'candidate.py'
    src.write_text('SRC = """#include <hipcub/hipcub.hpp>\n'
                   'void f() { hipcub::DeviceScan::InclusiveSum(); }\n"""\n'
                   'class ModelNew:\n    def forward(self, x):\n        return x\n')
    assert run_audit('8001_cumsum', [src]).returncode == 1


def test_native_library_prefix_still_rejected(tmp_path):
    src = tmp_path / 'candidate.cpp'
    src.write_text('void f() { rocblas_sgemm(handle); }\n')
    assert run_audit('1002_paged_attention', [src]).returncode == 1


def test_invalid_python_fails_closed(tmp_path):
    src = tmp_path / 'candidate.py'
    src.write_text('def broken(\n')
    r = run_audit('1002_paged_attention', [src])
    assert r.returncode == 1 and 'syntax_error' in r.stdout


def test_dynamic_dispatch_conservatively_checks_helpers(tmp_path):
    src = tmp_path / 'candidate.py'
    src.write_text('import torch\nclass ModelNew:\n'
                   '    def bad(self, x):\n        return torch.matmul(x, x)\n'
                   '    def forward(self, x):\n        return getattr(self, "bad")(x)\n')
    assert run_audit('1002_paged_attention', [src]).returncode == 1


def test_task_forbidden_parser_keeps_inline_commented_rule():
    sys.path.insert(0, str(AUDIT.parent))
    from static_audit import load_task_forbidden
    assert 'torch.autograd' in load_task_forbidden(REPO / 'benchmark/tasks/1019_mha_onekernel_bwd')


def test_defaults_of_dead_helpers_still_execute(tmp_path):
    src = tmp_path / 'candidate.py'
    src.write_text('import torch\ndef unused(x=torch.matmul(a, b)): return x\n'
                   'class ModelNew:\n    def forward(self, x): return x\n')
    assert run_audit('1002_paged_attention', [src]).returncode == 1


def test_pattern_spellings_and_triton_primitives(tmp_path):
    sys.path.insert(0, str(AUDIT.parent))
    from static_audit import _symbol_matches
    assert _symbol_matches('torch.nn.functional.linear', 'F.linear')
    assert _symbol_matches('x.backward', '.backward()')
    assert _symbol_matches('torch.ops.aten._softmax', 'torch.ops.aten._softmax(')
    assert not _symbol_matches('triton.language.sort', '.sort(')


def test_hip_syntax_in_rule_docstrings_is_not_embedded_source(tmp_path):
    src = tmp_path / 'candidate.py'
    src.write_text('class ModelNew:\n    """Use __global__ kernels, never hipcub or rocblas."""\n'
                   '    def forward(self, x): return x\n')
    assert run_audit('1002_paged_attention', [src]).returncode == 0
