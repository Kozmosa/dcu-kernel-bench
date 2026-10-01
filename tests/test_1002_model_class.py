# 1002（1001 的 model_class 变体）的一致性与可加载性测试。
#
# 覆盖：
# - compat 镜像与 tasks 侧 reference.py 字节一致（单一事实源，防漂移）
# - PyramidKernel loader 能从 compat 树构建 model_class scaffold（锚点齐全）
# - 生成的 scaffold 通过框架静态守卫（callable 路线在此处被拦的回归点）
# - Model.forward 与 1001 的 reference 在相同输入下输出一致（重现正确性）
# - KernelBench 评测器的 precision cast（全部张量转 fp32）下 forward 仍正确
# - get_inputs 的 shape 族与 public_cases.json 契约一致

import importlib.util
import json
import math
from pathlib import Path

import torch

from conftest import load_reference

REPO = Path(__file__).resolve().parents[1]
TASK_DIR = REPO / "benchmark" / "tasks" / "1002_paged_attention"
COMPAT_ROOT = REPO / "benchmark" / "kernelbench_compat"
COMPAT_FILE = COMPAT_ROOT / "level3" / "1002_paged_attention.py"


def _load_problem_module():
    spec = importlib.util.spec_from_file_location("kb_problem_1002", COMPAT_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_task_spec():
    from pyramidkernel.core.loader import KernelBenchLoader

    return KernelBenchLoader().load_official_problem(COMPAT_ROOT, 3, 1002, backend="triton")


def test_compat_mirror_in_sync():
    assert COMPAT_FILE.read_bytes() == (TASK_DIR / "reference.py").read_bytes(), (
        "compat 镜像与 tasks 侧 reference.py 不一致：改动请落在 tasks 侧，"
        "再 cp 到 benchmark/kernelbench_compat/level3/"
    )


def test_loader_builds_model_class_scaffold():
    spec = _load_task_spec()
    assert spec.function_name == "ModelNew"
    assert spec.entry_kind == "model_class"
    assert spec.backend == "triton"
    assert "class ModelNew" in spec.source_code
    anchors = {region.anchor_name for region in spec.grounded_regions}
    assert "helpers" in anchors
    assert "init_body" in anchors
    assert any(name.startswith("forward_stmt_") for name in anchors)
    compile(spec.source_code, "<scaffold>", "exec")  # 语法有效性
    # 源码隔离：agent 可见的 scaffold 与描述不得含溯源信息
    # （forbidden 列表里的 "aiter" 是禁用项声明，属合法内容，不在检查范围）
    for provenance in ("pa_decode", "opendas", "c39fff8c", "third_party", "private/", "sources/"):
        assert provenance not in spec.source_code.lower()
        assert provenance not in spec.description.lower()
    assert "paged attention" in spec.description.lower()


def test_scaffold_passes_static_guard():
    from pyramidkernel.core.static_check import check_candidate_static

    result = check_candidate_static(_load_task_spec().source_code, backend="triton")
    assert result.ok, result.logs


def test_model_reproduces_1001_reference():
    mod = _load_problem_module()
    ref1001 = load_reference("1001_paged_attention")

    torch.manual_seed(7)
    query, key_cache, value_cache, block_tables, seq_lens = mod.get_inputs()
    model = mod.Model(*mod.get_init_inputs())
    out_model = model(query, key_cache, value_cache, block_tables, seq_lens)
    out_fn = ref1001.reference(query, key_cache, value_cache, block_tables, seq_lens)

    assert out_model.dtype == query.dtype
    assert out_model.shape == query.shape
    assert torch.equal(out_model, out_fn), "model_class 变体与 1001 函数式 reference 输出不一致"


def test_forward_survives_evaluator_precision_cast():
    # 模拟 KernelBenchPaperEvaluator 的 _process_input_tensor：全部输入张量
    # cast 成评测 precision（triton 后端为 fp32），含 int32 的 block_tables/seq_lens
    mod = _load_problem_module()

    torch.manual_seed(11)
    query, key_cache, value_cache, block_tables, seq_lens = mod.get_inputs()
    model = mod.Model(*mod.get_init_inputs())
    expected = model(query, key_cache, value_cache, block_tables, seq_lens)

    casted = [t.float() for t in (query, key_cache, value_cache, block_tables, seq_lens)]
    out_cast = model(*casted)
    assert out_cast.dtype == torch.float32
    # fp16 输入的输出再上 float，与直接 fp32 计算的输出应在 fp16 舍入误差内一致
    assert torch.allclose(out_cast, expected.float(), atol=2.0e-2, rtol=2.0e-2)


def test_get_inputs_matches_public_cases_contract():
    mod = _load_problem_module()
    contract = json.loads((TASK_DIR / "public_cases.json").read_text(encoding="utf-8"))["shape_family"]

    torch.manual_seed(3)
    query, key_cache, value_cache, block_tables, seq_lens = mod.get_inputs()

    assert query.shape == (contract["num_seqs"], contract["num_q_heads"], contract["head_size"])
    assert key_cache.shape == value_cache.shape == (
        contract["num_blocks"], contract["num_kv_heads"], contract["block_size"], contract["head_size"],
    )
    assert block_tables.shape == (contract["num_seqs"], contract["max_blocks_per_seq"])
    assert seq_lens.shape == (contract["num_seqs"],)
    assert query.dtype == torch.float16
    assert block_tables.dtype == torch.int32 and seq_lens.dtype == torch.int32
    assert int(seq_lens.max()) <= contract["max_seq_len"] and int(seq_lens.min()) >= 1
    # scale 契约：init 传 head_size，Model 内部 1/sqrt(head_size)
    assert mod.Model(*mod.get_init_inputs()).scale == 1.0 / math.sqrt(contract["head_size"])
