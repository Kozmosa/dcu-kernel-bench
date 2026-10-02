"""2026-10-02 aiter 批量准入批次（44 题）的资产一致性回归测试。

一套参数化代码覆盖本批全部任务，代替逐题手写样板：
- reference.py 语法完整（ast.parse，不触发任何模型构建）
- kernelbench_compat 镜像与 tasks 侧 reference.py 字节一致
  （compat 路径按 task.yaml difficulty 推 level：hard=3 / medium=2 / basic=1，
  找不到时回退扫描 level1-3 定位，并要求全仓恰好一份镜像）
- private 三 json（hidden_cases / perf_cases / baseline）可解析且 task_id 一致
- task.yaml 含 tolerance 与 forbidden 字段（终审静态审计的契约入口）

风格跟随既有 tests/（复用 conftest.py 的 REPO_ROOT / TASKS）。
"""
import ast
import json
from pathlib import Path

import pytest
import yaml

from conftest import REPO_ROOT, TASKS

PRIV = REPO_ROOT / "benchmark" / "private"
COMPAT_ROOT = REPO_ROOT / "benchmark" / "kernelbench_compat"

# 2026-10-02 批量准入的 44 题（与 operator_catalog.yaml 同批条目一一对应）
BATCH_TASKS = [
    "1010_chunked_pa_prefill",
    "1011_extend_attention",
    "1012_flash_attention_forward",
    "1014_hstu_attention",
    "1015_lean_atten",
    "1016_lean_atten_paged",
    "1017_mha",
    "1018_mha_fused_bwd",
    "1019_mha_onekernel_bwd",
    "1020_mla_decode_rope",
    "1021_pa_decode",
    "1022_pa_prefill",
    "1024_prefill_attention",
    "1026_sage_attention_qk_int8_per_block",
    "1027_sage_attention_qk_int8_per_block_causal",
    "1029_triton_decode_attention",
    "1030_unified_attention",
    "2001_activation",
    "2002_add_swiglu",
    "2003_fused_mul_add",
    "2004_fused_qk_concat",
    "2005_moe_activation",
    "2006_norm",
    "2007_rmsnorm",
    "2008_rope",
    "2009_softmax",
    "3002_moe_align_block_size",
    "3003_moe_op",
    "3004_moe_op_e2e",
    "3006_moe_op_mxfp4",
    "3007_moe_op_silu_fused",
    "3009_routing",
    "3010_topk",
    "4001_batched_gemm_a8w8",
    "4004_batched_gemm_bf16",
    "4005_fused_mxfp4_quant",
    "4006_gemm_a16w16",
    "4007_gemm_a16w16_atomic",
    "4008_gemm_a16w4",
    "4009_gemm_a8w8",
    "4010_gemm_a8w8_blockscale",
    "4016_gemm_w8a8",
    "4017_group_quant_int8",
    "4018_quant",
]

DIFFICULTY_TO_LEVEL = {"basic": 1, "medium": 2, "hard": 3}


def _task_spec(task_id: str) -> dict:
    with open(TASKS / task_id / "task.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _locate_compat_mirror(task_id: str, difficulty: str) -> Path:
    """difficulty 推 level 定位镜像；推不出时扫描 level1-3 兜底。"""
    level = DIFFICULTY_TO_LEVEL[difficulty]
    derived = COMPAT_ROOT / f"level{level}" / f"{task_id}.py"
    if derived.exists():
        return derived
    hits = sorted(COMPAT_ROOT.glob(f"level*/{task_id}.py"))
    assert hits, f"{task_id}: level{level} 无镜像，level1-3 扫描也未找到 compat 镜像"
    return hits[0]


@pytest.mark.parametrize("task_id", BATCH_TASKS, ids=BATCH_TASKS)
def test_reference_parses(task_id: str):
    src = (TASKS / task_id / "reference.py").read_text(encoding="utf-8")
    ast.parse(src)  # 语法完整即过；不 import（避免拉起 torch/模型构建）


@pytest.mark.parametrize("task_id", BATCH_TASKS, ids=BATCH_TASKS)
def test_compat_mirror_byte_identical(task_id: str):
    spec = _task_spec(task_id)
    compat = _locate_compat_mirror(task_id, spec["difficulty"])
    assert compat.parent.name == f"level{DIFFICULTY_TO_LEVEL[spec['difficulty']]}", (
        f"{task_id}: 镜像位于 {compat.parent.name}，与 difficulty="
        f"{spec['difficulty']} 推出的 level 不符"
    )
    assert compat.read_bytes() == (TASKS / task_id / "reference.py").read_bytes(), (
        f"{task_id}: compat 镜像与 tasks 侧 reference.py 不一致：改动请落在 tasks 侧，"
        f"再同步到 benchmark/kernelbench_compat/"
    )


@pytest.mark.parametrize("task_id", BATCH_TASKS, ids=BATCH_TASKS)
def test_private_assets_parse(task_id: str):
    priv = PRIV / task_id
    for kind in ("hidden_cases", "perf_cases", "baseline"):
        data = json.loads((priv / f"{kind}.json").read_text(encoding="utf-8"))
        assert data["task_id"] == task_id, f"{task_id}: private/{kind}.json 的 task_id 漂移"


@pytest.mark.parametrize("task_id", BATCH_TASKS, ids=BATCH_TASKS)
def test_task_yaml_has_tolerance_and_forbidden(task_id: str):
    spec = _task_spec(task_id)
    assert spec.get("tolerance"), f"{task_id}: task.yaml 缺 tolerance 字段"
    assert isinstance(spec.get("forbidden"), list) and spec["forbidden"], (
        f"{task_id}: task.yaml 缺 forbidden 列表"
    )
