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
