# 1116 修复与验证

工作分支 `1116-fix`，基座为 `main` 的 `de3f151`。讨论文档中的原因属于待核对的归因，本轮按代码与回归结果修复；没有把历史数据的成功率当成重跑结果。

## 静态审计

按 Python AST 标识符/命名空间与从 `ModelNew.__init__/forward` 可达的辅助代码审计，避免 `_fused_adamw_kernel` 和未调用的参考函数误伤。函数值传递、导入别名、继承辅助方法、内嵌 HIP/C++ 源码仍检查；动态 `getattr/eval/exec` 保守检查全部定义。无 `ModelNew` 的 callable 提交仍扫描全部函数。语法错误判失败。这是静态合规检查，不是任意 Python 的安全沙箱或完整程序证明。

## 5007 参考实现

三个打包段的 `torch.tensor` 指定 `device=indices.device`，并同步 KernelBench 镜像。CPU 验证数值、meta 设备探针验证分配设备，CUDA/HIP 可用时同一个测试会加入真实设备检查。

## 统一评测契约

`benchmark/evaluator/dcukb_online.py`：在线保留 `get_inputs()` 的原生 dtype（包括整数索引），用 `task.yaml` 的输出 dtype 容差、分段容差、失配预算；候选执行前完成与终审相同的静态审计。`ModelNew` 输出 shape/dtype/device 必须与参考一致。参考与候选使用输入副本，防止原地更新相互污染。整数比较不会先转 fp32 丢失大于 2^24 的差异，NaN/Inf 不受失配预算豁免。

在线只读取公开题面和 task.yaml，不读 private 资产。隐藏形状、隐藏 case 仍只用于终审。**统一的是 dtype/容差/合规/输出判据，在线与隐藏测试的样本集合仍不同。** 在线记录每次 seed、输入 dtype、判定细节；用相同 seed 的 `--public` 终审模式交叉重测。

只支持此次固定的 eager 参考分母与 cuda event 中位数，不能和历史 KernelBench 的 fp32/固定 1e-4 数据直接拼接。报告只输出 `torch_eager`，不把它冒充 torch.compile/aiter 分母。重跑后需重新采集基线，不能继续引用旧输入契约下的加速比。

### 部署框架桥接补丁

框架源码不放进 DCUKB。补丁分别针对可访问的仓库快照：

| 仓库 | 起点 | 补丁目录 |
|---|---|---|
| Kozmosa/KernelWeaver | `c11ab7f` | `patches/kernelweaver/` |
| Kozmosa/PyramidKernel | `8352750` | `patches/pyramidkernel-1116/` |

两个目录是两个独立序列，各自按文件名顺序应用，**不要与旧 `patches/pyramidkernel/0001..0006` 混用**。旧目录保留历史部署补丁。脚本检查基座祖先、完整补丁适用性，并在失败时停止，不模糊套用补丁。

```bash
python benchmark/tools/apply_framework_fixes.py kernelweaver /path/to/KernelWeaver --check
python benchmark/tools/apply_framework_fixes.py kernelweaver /path/to/KernelWeaver
# 或 pyramidkernel /path/to/PyramidKernel
export DCUKB_ROOT=/absolute/path/to/dcu-kernel-bench
# 再从对应框架启动原来的 run-kernelbench / run-kernelbench-batch 命令。
```

`DCUKB_ROOT` 激活公开任务契约。明确配置但任务/路径不匹配时失败，不悄悄退回旧评测。Loader 只向生成上下文补充公开 IO/容差/禁用规则，评测桥接代码不进入模型上下文。未设置这个变量的普通 KernelBench 任务保留原评测行为。

原先仅归档在 `evidence/deploy_scripts.tar.gz` 的单次生成脚本已提升为 `benchmark/tools/pure_bench.py`。它要求相同 `DCUKB_ROOT`，提示词不再声称输入全为 fp32 或固定 1e-4。历史证据压缩包不改写。

```bash
# 同一候选、同一公开输入契约复核（DCU 环境）：
python benchmark/evaluator/audit_model_class.py --task 1002_paged_attention \
  --submission /path/to/best_code.py --public --seed 0 --trials 3 --skip-perf
# 随后按原命令跑隐藏 case 终审，不向生成侧开放隐藏样本。
```

## root 不作为候选交付

第二个框架补丁同时处理排行榜、最终选优、两阶段合并、导出、CLI、重放显示和 shadow validation；PyramidKernel 的 feedback/champion 也排除参考根节点。root 仍保留为 oracle/展开起点，`phase2_root` 仍保留为第二阶段的种子快照，二者都不能作为提交获胜。原样返回 oracle scaffold 的“生成”在候选评测前判失败。

合法候选只与生成候选比较，所以速度即使比 root 慢 12.6 倍也不会被参考答案覆盖。没有合法候选时 `best_node_id=null`、`stats.search_status=no_valid_candidate`，所有节点/日志仍保存在 `run.json`，不导出 `best_code.py`；复用输出目录时会删除旧成功文件。CLI 的单题运行返回非零，批量摘要记录失败而不冒充成功。

单次生成脚本保留失败的 `candidate_code.py` 与 `run.json`，不再用 root 作为失败占位。失败候选不会写成 `best_code.py`。旧实验归档保持原样。

## 失败分母与缓存

批量终审拒绝 `root/phase2_root` 和没有在线合法候选的 run；清单里的缺文件任务也计失败分母。缓存绑定候选代码、run.json 的选优、审计实现、任务定义、参考、隐藏/性能 case 与基线（含继承源），旧缓存不会在规则变化后继续冒充新结果。缓存失败仍返回失败退出码。

## 本轮验证记录

环境：Linux x86_64、Python 3.12、PyTorch 2.15.0+cpu、pytest 9.2.0；无 CUDA/HIP/DCU 设备。

- DCUKB：`PYTHONPATH=../verify-pyramidkernel python -m pytest -q -k 'not test_every_case_builds_inputs_and_runs_reference'`：326 passed，1 deselected。
- 被排除项遍历全量隐藏/性能 shape 并执行参考计算；首次完整 CPU 运行在此项被系统终止（exit 137），因此不能声称全套通过。首次运行还因未配置 PyramidKernel 导入路径出现两项导入失败；补好 PYTHONPATH 后这两项已通过。
- 从 `c11ab7f` 干净检出应用两个 KernelWeaver 补丁后：整个框架 `tests/` 23 passed。
- 从 `8352750` 干净检出应用两个 PyramidKernel 补丁后：oracle 边界、两阶段工作流、阶段转换、feedback、direct baseline、CLI、持久化/loader 共 31 passed。
- 两个干净框架的实际 loader 都能注入公开契约；实际 evaluator 在无 GPU 条件下也先拦下禁用调用，没有退回上游 fp32 评测。
- `git diff --check`、Python 编译检查通过。

未验证：DCU 真机编译/运行、HIP/Triton 数值与计时、真实模型生成、历史候选全量重测。原先发布的评测数字未重新计算。下一步在 DCU 上按部署节应用对应补丁、设置 DCUKB_ROOT，先用固定候选执行公开输入交叉复核，再跑隐藏 case 与新基线；确认后再启动新的生成实验。
