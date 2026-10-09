"""kernelbench_compat 镜像一致性回归。

镜像规则：`tasks/<id>/reference.py`（`entry: model_class`）必须与
`kernelbench_compat/level<N>/<id>.py` **字节一致**，level 由 task.yaml 的
difficulty 映射：basic=1 / medium=2 / hard=3。

改动必须落在 tasks 侧，再用
`python benchmark/tools/dcukb.py mirror --all --write` 同步；本测试用于抓漂移。

此前该检查只在 tests/test_1002_model_class.py 里写死 1002；这里做成通用版，
以后新增 model_class 任务会自动纳入。
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "benchmark" / "tasks"
COMPAT = REPO / "benchmark" / "kernelbench_compat"
DIFFICULTY_TO_LEVEL = {"basic": 1, "medium": 2, "hard": 3}
SYNC_HINT = "运行 python benchmark/tools/dcukb.py mirror --all --write"


def _task_scalars(task_dir: Path) -> dict:
    """读 task.yaml 的顶格标量 entry / difficulty（最小解析，需处理行尾注释）。"""
    text = (task_dir / "task.yaml").read_text(encoding="utf-8")
    found = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z_]\w*):\s*(.*?)\s*$", line)
        if not m or m.group(1) not in ("entry", "difficulty"):
            continue
        raw = m.group(2)
        if raw[:1] in ("'", '"'):
            quote = raw[0]
            end = raw.find(quote, 1)
            found[m.group(1)] = raw[1:end] if end > 0 else raw.strip(quote)
        else:
            found[m.group(1)] = raw.split("#", 1)[0].strip()
    return found


def _model_class_tasks() -> list:
    tasks = []
    for task_dir in sorted(TASKS.iterdir()):
        if task_dir.is_dir() and (task_dir / "task.yaml").exists():
            if _task_scalars(task_dir).get("entry") == "model_class":
                tasks.append(task_dir)
    return tasks


def _level_of(task_dir: Path) -> int:
    difficulty = _task_scalars(task_dir).get("difficulty")
    assert difficulty in DIFFICULTY_TO_LEVEL, (
        f"{task_dir.name}: difficulty={difficulty!r} 无法映射 level（应为 basic|medium|hard）"
    )
    return DIFFICULTY_TO_LEVEL[difficulty]


def test_there_is_at_least_one_model_class_task():
    assert _model_class_tasks(), "没有任何 entry: model_class 的任务，镜像规则无从校验"


def test_compat_mirrors_are_byte_identical():
    for task_dir in _model_class_tasks():
        mirror = COMPAT / f"level{_level_of(task_dir)}" / f"{task_dir.name}.py"
        assert mirror.exists(), (
            f"{task_dir.name}: 缺少镜像 {mirror.relative_to(REPO)}；{SYNC_HINT}"
        )
        assert mirror.read_bytes() == (task_dir / "reference.py").read_bytes(), (
            f"{task_dir.name}: 镜像与 tasks 侧不一致。改动请落在 tasks 侧，{SYNC_HINT}"
        )


def test_no_mirror_in_wrong_level():
    """difficulty 改过但旧镜像没删，会让 loader 在两个 level 下都找到同一题。"""
    expected = {d.name: COMPAT / f"level{_level_of(d)}" / f"{d.name}.py" for d in _model_class_tasks()}
    for level_dir in sorted(COMPAT.glob("level*")):
        for mirror in sorted(level_dir.glob("*.py")):
            want = expected.get(mirror.stem)
            if want is not None and mirror.resolve() != want.resolve():
                raise AssertionError(
                    f"level 不匹配的残留镜像：{mirror.relative_to(REPO)}"
                    f"（应位于 {want.relative_to(REPO)}）"
                )
