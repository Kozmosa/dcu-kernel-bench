"""case 契约回归：每个 model_class 任务的 hidden/perf case 都必须能真正跑起来。

本模块是本分支相对上游 main 的增量之一；迁移到上游基座后做了两处适配：

- 容差按**输出 dtype** 组织（上游 4xxx 量化题输出是 int8/fp8/int32，不能沿用
  输入 dtype 的半精度容差），因此不再断言"case 的 dtype 必须是 tolerance 的键"，
  改为断言 `resolve_tolerance_limit` 能解析出容差块。
- 构造参数改为**关键字**绑定（`case_init_kwargs`）。位置式对稀疏参数不安全：
  实测 1020_mla_decode_rope 的 `get_init_inputs()=[512,64,64]` 只覆盖前 3 个
  参数的默认值，case 里的 `is_neox_style` 一旦按位置 append 就会落到
  `sm_scale` 的槽位上——静默算错；而 reference 与候选都用同一个错值，
  allclose 照样通过。

覆盖的契约：
  1. case 字段能被任务的 make_inputs 接受（混入注解字段会在这里变成 TypeError）
  2. 构造参数能按 `io.init_inputs`（或 Model 签名）解析，且 forward 能跑通
  3. forward 返回**单个** Tensor，形状/dtype 自洽，数值有限
  4. task.yaml 的 tolerance 能覆盖该 case 的输入与输出 dtype
  5. `io.init_inputs` 声明与 Model 签名前缀一致，且 case 字段确实生效

不覆盖：reference 的数值语义是否正确（自比恒真）。那需要真机 allclose 或独立 oracle。
"""
import json
import pathlib
import sys

import torch
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmark"
TASKS = BENCH / "tasks"
PRIV = BENCH / "private"
EVALUATOR = BENCH / "evaluator"

sys.path.insert(0, str(EVALUATOR))

from audit_model_class import (  # noqa: E402
    case_init_kwargs,
    case_inputs,
    load_init_names,
    load_module,
    resolve_case_file,
    resolve_tolerance_limit,
)

# 大于该元素数的 case 在本地测试里跳过前向（真机 audit_model_class.py 仍跑全量）
MAX_ELEMS = 1 << 24  # 16M 元素

# 已知违反单张量契约的任务白名单。当前为空：上游原本有 `1017_mha`（返回 out, lse）
# 与 `1019_mha_onekernel_bwd`（返回 dq, dk, dv）两例，KernelBench 的
# `output.shape` 与终审的 `actual.shape` 都会 AttributeError，真机跑不了。
# 已在本分支修成"打包成单张量"（1017 按最后一维加 1 列；1019 三段展平后沿
# 第 0 维拼接），并同步更新了 task.yaml 的 io.outputs 与 Model docstring。
# 若将来再出现同类任务，把 task id 加进这个集合即可让套件保持绿，但**不要**
# 当成"修好了"——白名单只记录，不解决问题。
KNOWN_SINGLE_TENSOR_VIOLATIONS: set = set()

_counter = [0]


def _fresh_load(path, tag):
    _counter[0] += 1
    return load_module(path, f"contract_{tag}_{_counter[0]}")


def _model_class_tasks():
    out = []
    for task_dir in sorted(TASKS.iterdir()):
        spec_path = task_dir / "task.yaml"
        if task_dir.is_dir() and spec_path.exists():
            if yaml.safe_load(spec_path.read_text(encoding="utf-8")).get("entry") == "model_class":
                out.append(task_dir)
    return out


def _spec(task_dir):
    return yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))


def _cases_of(task_id, kind):
    case_path, gen_dir = resolve_case_file(task_id, kind)
    data = json.loads(case_path.read_text(encoding="utf-8"))
    return data["cases"], gen_dir, case_path


def _concrete_output_dtypes(task_dir):
    outs = ((_spec(task_dir).get("io") or {}).get("outputs") or [])
    result = []
    for entry in outs:
        dtype = entry.get("dtype") if isinstance(entry, dict) else None
        if isinstance(dtype, str):
            result.append(dtype.replace("torch.", ""))
    return result


def _tensors(values):
    return [v for v in values if isinstance(v, torch.Tensor)]


def test_there_are_model_class_tasks():
    assert _model_class_tasks(), "没有任何 entry: model_class 的任务"


def test_case_dtypes_are_resolvable_against_tolerance():
    """每个 case 的输入 dtype 与任务声明的输出 dtype 都必须能解析出容差块。"""
    problems = []
    for task_dir in _model_class_tasks():
        tol = _spec(task_dir).get("tolerance") or {}
        candidates = set(_concrete_output_dtypes(task_dir))
        for kind in ("hidden", "perf"):
            cases, _, _ = _cases_of(task_dir.name, kind)
            for case in cases:
                dts = set(candidates)
                if case.get("dtype"):
                    dts.add(str(case["dtype"]))
                for dt in sorted(dts):
                    try:
                        resolve_tolerance_limit(tol, str(case.get("dtype", "")), dt)
                    except KeyError as exc:
                        problems.append(
                            f"{task_dir.name} {kind}/{case['name']}: dtype {dt!r} 无法解析容差（{exc}）"
                        )
    assert not problems, "容差解析失败：\n" + "\n".join(problems)


def test_every_case_builds_inputs_and_runs_reference():
    """核心回归：每个 case 都要能生成输入、构造 Model 并跑通 forward。"""
    torch.manual_seed(0)
    problems, skipped, dtype_notes = [], [], []
    for task_dir in _model_class_tasks():
        ref_mod = _fresh_load(task_dir / "reference.py", task_dir.name)
        if not hasattr(ref_mod, "Model"):
            continue
        for kind in ("hidden", "perf"):
            cases, gen_dir, _ = _cases_of(task_dir.name, kind)
            gen_mod = _fresh_load(gen_dir / "reference.py", f"{task_dir.name}_{kind}")
            for case in cases:
                label = f"{task_dir.name} {kind}/{case['name']}"
                try:
                    inputs = list(case_inputs(gen_mod, case))
                except TypeError as exc:
                    problems.append(f"{label}: make_inputs 不接受 case 字段: {exc}")
                    continue
                except Exception as exc:  # noqa: BLE001
                    problems.append(f"{label}: 生成输入失败 {type(exc).__name__}: {exc}")
                    continue

                try:
                    init_kwargs = case_init_kwargs(task_dir, ref_mod, case)
                except Exception as exc:  # noqa: BLE001
                    problems.append(f"{label}: 解析构造参数失败 {type(exc).__name__}: {exc}")
                    continue

                n_elems = sum(t.numel() for t in _tensors(inputs))
                if n_elems > MAX_ELEMS:
                    skipped.append(f"{label}（{n_elems} 元素超本地上限）")
                    continue

                try:
                    out = ref_mod.Model(**init_kwargs)(*inputs)
                except Exception as exc:  # noqa: BLE001
                    shapes = [
                        tuple(t.shape) if isinstance(t, torch.Tensor) else type(t).__name__ for t in inputs
                    ]
                    problems.append(
                        f"{label}: forward 失败 {type(exc).__name__}: {exc}"
                        f"（init_kwargs={init_kwargs}，输入={shapes}）"
                    )
                    continue

                if not isinstance(out, torch.Tensor):
                    if task_dir.name in KNOWN_SINGLE_TENSOR_VIOLATIONS:
                        skipped.append(f"{label}（白名单记录：forward 返回 {type(out).__name__}）")
                        continue
                    problems.append(
                        f"{label}: forward 返回 {type(out).__name__}，model_class 契约要求单个 Tensor"
                    )
                    continue
                if not torch.isfinite(out.float()).all():
                    problems.append(f"{label}: 输出含 NaN/Inf")
                # 输出 dtype 只做信息性核对：上游任务异构（int8 输入→fp16 输出、
                # fp16 输入→uint8 输出、int64 输入→fp32 输出 都有），不能一律
                # 要求等于首个输入 dtype；真实口径由 task.yaml 的 tolerance 按
                # 输出 dtype 决定（见 test_case_dtypes_are_resolvable_against_tolerance）。
                first = next(iter(_tensors(inputs)), None)
                if first is not None and out.dtype != first.dtype:
                    dtype_notes.append(f"{label}: 输出 {out.dtype} vs 首输入 {first.dtype}")

    if skipped:
        print(f"[contract] 本地跳过 {len(skipped)} 个大 case: {skipped}")
    if dtype_notes:
        print(f"[contract] 输出 dtype 与首输入不同（信息性，共 {len(dtype_notes)} 条，非失败）")
    assert not problems, "case 契约失败：\n" + "\n".join(problems)


def test_init_inputs_declaration_matches_model_signature():
    """task.yaml `io.init_inputs` 的声明必须与 `Model.__init__` 签名前缀一致。

    这是上游 main 引入的 schema 契约（`audit_model_class.load_init_names` 按名
    取参依赖它）：声明成 `[head_size]` 就必须对应签名的第 0 个参数。声明与实际
    签名错位会让"按名取参"落到错误的参数上，比不声明更危险。
    `io.init_inputs: []` 表示无超参任务，此时 Model 不得有必需参数。
    """
    import inspect

    problems = []
    for task_dir in _model_class_tasks():
        names = load_init_names(task_dir)
        if names is None:
            continue                                     # 未声明则改用签名，不强制
        ref_mod = _fresh_load(task_dir / "reference.py", task_dir.name)
        if not hasattr(ref_mod, "Model"):
            continue
        params = inspect.signature(ref_mod.Model.__init__).parameters
        ordered = [p for p in params if p != "self"]
        if names == []:
            required = [p for p, v in params.items() if p != "self" and v.default is inspect.Parameter.empty]
            if required:
                problems.append(
                    f"{task_dir.name}: 声明 io.init_inputs=[]（无超参）但 Model.__init__ 有必需参数 {required}"
                )
            continue
        if ordered[: len(names)] != names:
            problems.append(
                f"{task_dir.name}: io.init_inputs={names} 与 Model.__init__ 参数表 {ordered} 前缀不一致"
            )
    assert not problems, "io.init_inputs 声明不一致：\n" + "\n".join(problems)


def test_case_init_kwargs_are_effective():
    """case 里与构造参数同名的字段，必须实际出现在解析出的 kwargs 中。

    回归背景——"case 声明了构造参数，但没传到 Model 上"这一类缺陷会**静默**
    测错语义（reference 与候选都用同一个错默认值，allclose 照样通过）：
      - 上游 1020_mla_decode_rope：case 里的 `is_neox_style` 若用位置式 append，
        会落到 `sm_scale` 的槽位（`get_init_inputs()` 只给 3 个默认值）。
      - 1014_hstu_attention：同类（`max_attn_len` / `contextual_seq_len` / `causal`）。
      - 本分支在 dev 线上另发现 1004 / 1002 / 1008 三例，随任务收敛到上游编号后
        由同一套断言继续守住。
    """
    import inspect

    problems = []
    for task_dir in _model_class_tasks():
        declared = load_init_names(task_dir)
        if declared == []:
            continue
        ref_mod = _fresh_load(task_dir / "reference.py", task_dir.name)
        if not hasattr(ref_mod, "Model"):
            continue
        sig = [p for p in inspect.signature(ref_mod.Model.__init__).parameters if p != "self"]
        authoritative = declared if declared is not None else sig
        for kind in ("hidden", "perf"):
            cases, _, _ = _cases_of(task_dir.name, kind)
            for case in cases:
                kwargs = case_init_kwargs(task_dir, ref_mod, case)
                for name in authoritative:
                    if name not in case:
                        continue
                    if name not in kwargs:
                        problems.append(
                            f"{task_dir.name} {kind}/{case['name']}: case 字段 {name}={case[name]!r} "
                            f"未出现在构造 kwargs={kwargs}"
                        )
                    elif kwargs[name] != case[name]:
                        problems.append(
                            f"{task_dir.name} {kind}/{case['name']}: kwargs[{name}]={kwargs[name]!r} "
                            f"!= case 字段 {case[name]!r}"
                        )
    assert not problems, "构造参数未生效：\n" + "\n".join(problems)
