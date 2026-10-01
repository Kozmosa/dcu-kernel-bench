# kernelbench_compat — model_class 兼容题库

PyramidKernel 原生 model_class 路径的题目树。`--kernelbench-root` 指向本目录，
loader 按 `level{N}/{problem_id}_*.py` 寻题。level 映射 catalog difficulty：
basic=1 / medium=2 / hard=3。

题目文件是 `benchmark/tasks/<id>/reference.py` 的逐字节镜像
（`tests/test_1002_model_class.py` 校验一致性；改动题目请改 tasks 侧再镜像
过来，漂移会被测试拦截）。

运行示例（quick 冒烟，mock provider）：

```bash
pyramidkernel run-kernelbench \
  --experiment quick \
  --kernelbench-root <本目录绝对路径> \
  --level 3 --problem-id 1002 \
  --backend triton \
  --output-dir runs/1002_smoke
```

| 文件 | 对应任务 |
|---|---|
| `level3/1002_paged_attention.py` | 1002_paged_attention（1001 的 model_class 变体） |
