"""测试辅助：从任务目录动态加载 reference.py，保持任务包自包含。"""
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = REPO_ROOT / "benchmark" / "tasks"


def load_reference(task_id: str):
    path = TASKS / task_id / "reference.py"
    spec = importlib.util.spec_from_file_location(f"reference_{task_id}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
