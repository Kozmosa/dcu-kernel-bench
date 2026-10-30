# hipDNN 作为性能基线的可行性探针（2026-10-15，DCU BW / gfx936 / DTK 26.04）

## 结论（先说）

**在当前机器上，hipDNN 做不了基线。** 阻塞点在 vendor 自己的引擎插件（DTK 26.04 的
`libmiopen_legacy_plugin.so`），不在我们的工具链。官方样例本身也要求 **DTK ≥ 26.10**。

**但有个更好的消息**：`torch_eager` 对 conv 族题目**已经就是 vendor 库（MIOpen）基线** ——
见下文第 4 节。所以这些题并不缺「vendor 分母」。

## 1. 背景

2026-10-12 新准入的 21 道题（14 道 5xxx–8xxx + 1031/1032 + 2010–2014）源自
`opendtk/hipdnn_samples`。AGENTS.md 写明「闭源 hipDNN 库只作性能基线，不作题源」，
所以自然想给这批题补上 hipDNN 分母（现在只有 `torch_eager`）。

## 2. 探针做了什么

写了自包含的 `ConvFpropBench.cpp`（只用 `hipdnn_frontend` + 裸 `hipMalloc`，带 hipEvent
计时），对应 `tasks/5001_conv2d_fwd` 的 shape 口径。**能编译、能链接、能跑到图构建成功**，
卡在执行计划的创建。

## 3. 探针查出的真实约束（每条都由日志坐实，非推测）

DTK 26.04 的 hipDNN 与官方样例（按 ≥26.10 写）有多处 API 差异，逐条踩过：

| # | 约束 | 证据 |
|---|---|---|
| 1 | **引擎插件不会自动发现**，必须在 `hipdnnCreate` **之前**调 `hipdnnSetEnginePluginPaths_ext`（带 active handle 调会返回 `NOT_SUPPORTED`） | 不调：`HIPDNN_ATTR_ENGINEHEUR_RESULTS elementCount=0`；调用后日志出现 `Plugin loaded successfully` + `numEngines=1` |
| 2 | **节点属性**要显式 `set_compute_data_type(FLOAT)`，图的 io type 不会下发 | 报 `MIOpen only supports nodes with an fp32 compute_data_type` |
| 3 | **每个张量**要显式 `set_data_type` | 不设：conv plan builder 报泛化的 "not applicable"（该分支无详细子消息） |
| 4 | 26.04 用**三步构建**：`build_operation_graph` → `create_execution_plans` → `build_plans`（样例的 `graph->build(handle)` 是 ≥26.10 的便利方法） | 26.04 的 `Graph` 无 `build` 成员 |
| 5 | SDK 布局/命名空间是 `hipdnn/sdk/hipdnn_sdk/...` + `hipdnn_sdk`，**不是** `hipdnn_data_sdk` | 样例 `#include <hipdnn_data_sdk/utilities/Tensor.hpp>` 编译失败 |
| 6 | 句柄用 C API `hipdnnCreate/hipdnnDestroy`（`hipdnn_frontend::detail::hipdnnBackend()` 在 26.04 不导出） | `'detail' has not been declared` |
| 7 | 本机唯一引擎是 `libmiopen_legacy_plugin.so`，即 hipDNN 的执行后端就是 MIOpen | `ls lib/hipdnn_plugins/engines/` |

## 4. 真正的阻塞点

满足 1–3 之后，图进入插件、逐个 plan builder 试：

```
Batchnorm/SDPA/Layernorm/Groupnorm/Rmsnorm ... not applicable
ConvolutionPointwise / GroupnormPointwise ... 需要 >= 2 个节点
Deformable convolution ... not applicable
=> Convolution plan builder 内部 segfault（core dumped）
```

不设 data_type 时是 `Convolution plan builder is not applicable for this graph`（NCHW 与
NHWC 布局都试过）。也就是说 **vendor 这个 legacy 插件的卷积路径在本机不可用**。

另外：样例 README 要求的 Python 绑定也没随 DTK 26.04 发布 ——
`pip install ${ROCM_PATH}/share/hipdnn/wheels/hipdnn-*.whl`，而 `share/hipdnn/wheels/`
不存在，全盘无 `hipdnn*.whl`，三个 hipdnn `.so` 的 `PyInit` 计数均为 0。

## 5. 顺带查清：`torch_eager` 对 conv 题已等价于 MIOpen 基线

```
ldd libtorch_hip.so          -> libMIOpen.so.1  ✓
torch.backends.cudnn.is_available() = True (version 2018000)   # DCU 上即 MIOpen 后端
profile F.conv2d(16,16,16,16 / 16,16,3,3) -> winograd_fwd_2d_fp32_3x3_s1_16x16x4
```

最后一条是 **MIOpen 的 Winograd 内核**。所以对 conv 族（5001/5002/5008/5009 等），
`baseline.json` 里的 `torch_eager` **就是 vendor 库分母**，不是「弱基线」。

对 torch 没有 MIOpen 路径的算子（deform_conv2d、spconv、ctc_loss、cumsum、gather、
scatter_nd、adamw），`torch_eager` 才是纯 ATen 实现。

## 6. 若要继续推进

按代价从低到高：

1. **不动**（推荐）：conv 族已有 MIOpen 分母；其余题保留 `torch_eager` 分母。
2. 换到带 **DTK ≥ 26.10** 的环境（或 FlagTree HCU 镜像）再试 hipDNN —— 属于基础设施变更，
   且需注意铁律 4：整套基线要重采。
3. 在本机继续逆向 `libmiopen_legacy_plugin.so` 的卷积前置条件（盲试代价高，不建议）。

## 7. 复现物

- 基准源码：`.dcu_runs/deploy/hipdnn_bench/{ConvFpropBench.cpp,CMakeLists.txt}`
- 构建脚本：`.dcu_runs/deploy/hipdnn_build.sh`
- 官方样例快照：`.dcu_runs/hipdnn-src/`（commit `b67a3f96`）
- 日志：服务器 `/root/dcudeploy/hipdnn_build*.log`，诊断脚本 `check_plugin*.sh` / `dump_api*.sh`
