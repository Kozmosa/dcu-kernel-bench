"""case JSON 与 YAML 资产的一致性回归测试。

覆盖此前准入流程中靠人工脚本做的检查：shape 约束、seq_lens 上界、
GQA 整除、dtype 白名单、catalog/sources/task 三处元数据一致。
"""
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "benchmark" / "tasks"
PRIV = REPO / "benchmark" / "private"

CASE_FILES = [
    TASKS / "0001_vecadd" / "public_cases.json",
    PRIV / "0001_vecadd" / "hidden_cases.json",
    PRIV / "0001_vecadd" / "perf_cases.json",
    TASKS / "1001_paged_attention" / "public_cases.json",
    PRIV / "1001_paged_attention" / "hidden_cases.json",
    PRIV / "1001_paged_attention" / "perf_cases.json",
]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_all_case_json_parse_and_have_ids():
    for path in CASE_FILES:
        data = load(path)
        assert data["task_id"] == path.parent.name
        assert len(data["cases"]) > 0


def test_vecadd_cases_bounds():
    for path in CASE_FILES[:3]:
        for c in load(path)["cases"]:
            assert isinstance(c["n"], int) and 1 <= c["n"] <= 67108864
            assert c["dtype"] in ("float32", "float16")


def test_paged_attention_cases_constraints():
    for path in CASE_FILES[3:]:
        for c in load(path)["cases"]:
            n, hq, hkv = c["num_seqs"], c["num_q_heads"], c["num_kv_heads"]
            assert hq % hkv == 0 and 1 <= hq // hkv <= 16
            assert 32 <= c["head_size"] <= 128
            assert 1 <= c["block_size"] <= 512
            assert 1 <= c["max_seq_len"] <= 8192          # v1 调度域
            assert c["dtype"] in ("float16", "bfloat16")
            if (sl := c.get("seq_lens")) is not None:
                if len(sl) == 1:
                    sl = sl * n
                assert len(sl) == n, c["name"]
                assert max(sl) <= c["max_seq_len"], c["name"]


def test_paged_attention_perf_timing_config():
    perf = load(PRIV / "1001_paged_attention" / "perf_cases.json")
    t = perf["timing"]
    assert t["warmup_iters"] >= 3 and t["repeat_iters"] >= 10
    assert t["reduction"] == "median"


def _load_yaml_assets():
    files = [
        REPO / "operator_catalog.yaml",
        REPO / "benchmark" / "sources" / "1001_paged_attention.yaml",
        TASKS / "0001_vecadd" / "task.yaml",
        TASKS / "1001_paged_attention" / "task.yaml",
    ]
    docs = {}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            docs[f.name] = yaml.safe_load(fh)
    return docs


def test_1001_metadata_consistent_across_layers():
    docs = _load_yaml_assets()
    catalog = next(o for o in docs["operator_catalog.yaml"]["operators"] if o["id"] == "1001_paged_attention")
    source = docs["1001_paged_attention.yaml"]
    task = docs["task.yaml"]

    assert catalog["status"] == "admitted"
    assert source["admission"]["decision"] == "admitted"
    commit = catalog["source"]["commit"]
    assert commit == source["source"]["commit"] == task["source"]["commit"]
    assert catalog["source"]["path"] == task["source"]["path"]
    assert source["admission"]["core_compute_in_source"] is True
    assert source["admission"]["calls_closed_libs"] is False
    assert catalog["impl_lang"] == task["impl_lang"] == "triton"
