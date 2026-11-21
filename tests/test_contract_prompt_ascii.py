"""注入生成上下文的在线契约摘要必须是纯 ASCII，且保留容差数值（题面零 CJK 铁律）。

`benchmark/evaluator/dcukb_online.py` 的 `prompt_contract()` 会把 task.yaml 的
`io` / `tolerance` / `tolerance_segments` / `forbidden` 摘要贴到 agent 可见的
description 与 source_code 顶部。task.yaml 的 `tolerance` 里夹着中文 `compare` /
`note` 散文，直接 dump 会把 CJK 带进题面，所以这里有回归守卫。
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmark" / "evaluator"))
from dcukb_online import contract_payload, prompt_contract  # noqa: E402


def model_class_tasks():
    for task_dir in sorted((REPO / "benchmark" / "tasks").iterdir()):
        spec_path = task_dir / "task.yaml"
        if not spec_path.exists():
            continue
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        if spec.get("entry") != "model_class":
            continue
        yield task_dir, spec


def fake_task(task_dir: Path):
    stem = task_dir.name
    return SimpleNamespace(source_origin=str(task_dir / "reference.py"),
                           problem_id=int(stem.split("_", 1)[0]))


def test_contract_is_ascii_and_keeps_numbers():
    checked = 0
    for task_dir, spec in model_class_tasks():
        contract = prompt_contract(fake_task(task_dir))
        assert contract.isascii(), f"{task_dir.name}: 注入的契约里还有非 ASCII 字符"
        assert "native input dtypes" in contract
        assert "NOT a fixed 1e-4" in contract
        payload = json.loads(contract.splitlines()[-1])
        if "tolerance" in spec:
            assert "tolerance" in payload, f"{task_dir.name}: 容差数值被清洗掉了"
            assert any(isinstance(v, dict) and ("atol" in v or "rtol" in v)
                       for v in payload["tolerance"].values()), task_dir.name
        if spec.get("forbidden"):
            assert payload.get("forbidden"), f"{task_dir.name}: 禁用清单被清洗掉了"
            for item in payload["forbidden"]:
                assert isinstance(item, str) and item.isascii()
        checked += 1
    assert checked >= 60, f"只检查到 {checked} 道 model_class 题"


def test_prose_keys_are_dropped():
    """task.yaml 里 tolerance 的散文键不进上下文，但判定仍读原文。"""
    task_dir = REPO / "benchmark" / "tasks" / "2006_norm"
    spec = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    assert any(isinstance(v, str) for v in spec["tolerance"].values()), \
        "这道题的 tolerance 里应当有散文键，本用例才有意义"
    payload = contract_payload(fake_task(task_dir))
    for value in payload["tolerance"].values():
        assert not isinstance(value, str), "散文键应当被丢弃"
        assert all(k in {"atol", "rtol", "max_mismatch_frac"} for k in value), value
