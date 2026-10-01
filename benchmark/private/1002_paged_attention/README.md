# 1002_paged_attention 私有评测资产

1002 与 1001 的算子语义完全一致（同源同 commit），case 资产**复用 1001**：

- `hidden_cases.json` / `perf_cases.json`：`{"inherit": "1001_paged_attention"}`，
  由 `benchmark/evaluator/audit_model_class.py` 解析；输入生成器为
  `benchmark/tasks/1001_paged_attention/reference.py::make_inputs`。
- `baseline.json`：aiter 基线占位（与 1001 共用，待真机构建 aiter 后测量）；
  附 2026-10-01 同机实测参考值（口径为 get_inputs 固定 shape 族，非 perf_cases）。

不在此复制 1001 的 case 内容，避免双份漂移；边界/极值 case 的语义变更只需改 1001 一处。
