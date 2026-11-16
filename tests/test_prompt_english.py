"""题面必须是纯英文（给生成 Agent 的上下文不允许出现中文）。

为什么有这条硬约束：题面的唯一载体是 `benchmark/tasks/<id>/reference.py` 里
`Model` 类的 docstring（PyramidKernel loader 用 `ast.get_docstring(model_class)`
取出，作为 `task_description` 塞进请求体），镜像树里那份与之逐字节一致。
题面里残留中文会让模型在产物 docstring / assert 里跟着写中文，也让"全英文运行"
这件事无法用机器验证。因此这里把"零 CJK"做成回归：

  1. `test_model_docstring_has_no_cjk`：Agent 真正看到的题面（Model docstring；
     callable 形态退化为模块 docstring）零 CJK。
  2. `test_task_sources_have_no_cjk`：题目包全部 Python 源（reference.py 与
     starter/*.py）零 CJK —— 含注释与 assert 消息，一次报全部违规文件，避免逐个试。

`task.yaml` / `public_cases.json` 不在此列：`task.yaml` 不进 Agent 上下文，
JSON 里的说明字段同理。
"""
import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "benchmark" / "tasks"

# 汉字 + 中文标点（全角标点同样不该出现在英文题面里）
CJK = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\u3400-\u4dbf\uff00-\uffef]")


def _task_dirs() -> list:
    return sorted(d for d in TASKS.iterdir() if d.is_dir() and (d / "task.yaml").exists())


def _model_docstring(path: Path) -> str:
    """`Model` 类 docstring —— 这就是 Agent 可见的题面全文。"""
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            return ast.get_docstring(node) or ""
    return ""


def test_model_docstring_has_no_cjk():
    offenders = []
    for task_dir in _task_dirs():
        ref = task_dir / "reference.py"
        assert ref.is_file(), f"{task_dir.name}: 缺 reference.py"
        doc = _model_docstring(ref) or (ast.get_docstring(ast.parse(ref.read_text(encoding="utf-8"))) or "")
        hits = CJK.findall(doc)
        if hits:
            offenders.append(f"{task_dir.name}({len(hits)}): {''.join(hits[:24])}")
    assert not offenders, (
        "题面仍有中文，PyramidKernel 拿到的就不是全英文上下文：\n  " + "\n  ".join(offenders)
    )


def test_task_sources_have_no_cjk():
    offenders = []
    for task_dir in _task_dirs():
        sources = [task_dir / "reference.py"] + sorted((task_dir / "starter").glob("*.py")) \
            if (task_dir / "starter").is_dir() else [task_dir / "reference.py"]
        for src in sources:
            hits = CJK.findall(src.read_text(encoding="utf-8"))
            if hits:
                offenders.append(f"{src.relative_to(REPO)} ({len(hits)}): {''.join(hits[:24])}")
    assert not offenders, "题目源码仍有中文（注释/assert 消息也算）：\n  " + "\n  ".join(offenders)


def test_cjk_pattern_sanity():
    """防止正则写宽/写窄导致上面两条恒真。"""
    assert CJK.search("目标硬件")
    assert CJK.search("target hardware；")
    assert not CJK.search("Target hardware: Hygon DCU (gfx936).")
