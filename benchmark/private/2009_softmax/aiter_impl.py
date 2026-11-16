# aiter_impl.py — 2009_softmax 的 aiter 官方实现适配器
#
# 来源（aiter pinned 检出 .dcu_runs/aiter_pinned，commit c39fff8c77df4e80617649e92fa3c2615f2c43d1）：
#   aiter/ops/triton/softmax.py
#     sha256 8579f5f4da0a01369603b26170537af16b8b5b63fd79f7cd67c8a668a2d6a229
#     （与 benchmark/sources/2009_softmax.yaml 记录的准入 sha256 一致）
#   公开算子入口（文件:行号）：
#     softmax(x)                                  softmax.py:55
#     _softmax_kernel_online(output_ptr, input_ptr, input_row_stride,
#                            output_row_stride, n_rows, n_cols, BLOCK_SIZE)  softmax.py:7
#   官方语义权威（同 commit）：
#     op_tests/triton_tests/test_softmax.py:25-38
#       sha256 f39678ad822d0e2ce0c3d58128a82a693102e09633bdf04ccf7f3071a29d9549
#       —— x = randn(M, N, dtype, device="cuda"); y = softmax(x);
#          与 torch.softmax(x, axis=1) 比对，fp16/bf16 atol=rtol=1e-2。
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
#
# 布局与语义对齐（题面 reference.py::Model.forward ↔ aiter softmax）：
#   - 题面 x 是 (M, N) contiguous、沿最后一维（dim=-1）逐行归一化；
#     aiter softmax(x) 正是 2D 行级、沿最后一维归一化（softmax.py:69 取
#     n_rows, n_cols = x.shape；kernel 按 input_row_stride = x.stride(0) 逐行扫描）。
#     两者语义、形状完全一致，**无需任何 permute/reshape**。
#   - 返回值：aiter 返回单张量 y = torch.empty_like(x)（softmax.py:73），
#     shape (M, N)、dtype 与 x 相同 → 直接满足题面「与 reference 输出同形、同
#     dtype 的单个张量」契约（题面 Model 也是 x.dtype 输出，reference.py:68）。
#     题面 reference 内部先 to(float32) 再算、最后 cast 回入口 dtype；aiter 的
#     kernel 直接在入口 dtype 上做 max/exp/求和（row_sum 累计为 fp32，块内
#     tl.sum 为入口 dtype）。两者差异由官方测试的 1e-2 容差覆盖，本题
#     task.yaml 的 fp16/bf16 容差为 2e-2（宽于官方），不额外 cast，
#     保持官方 fp16/bf16 路径的真实性能。
#   - 无需 autotune：入口内 BLOCK_SIZE = min(65536 // element_size,
#     next_power_of_2(n_cols)) 就地算，num_warps=8 / num_stages=2 /
#     waves_per_eu=2 为源码硬编码常量，不读 AITER_TRITON_CONFIGS_PATH，
#     也不带 @triton.autotune。
#   - 行内分块循环（tl.range(0, n_cols, BLOCK_SIZE)）自带掩码
#     （col_offsets < n_cols，other=-inf），N=1、非 2 的幂、N 超出单块上限
#     （fp16/bf16 为 65536/2 = 32768）都由官方路径自行处理。

import torch


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方实现。

    inputs     : [x]，x 为 (M, N) fp16/bf16/fp32、contiguous、已在 device 上
    init_kwargs: {}（本题 softmax 前向无超参数，task.yaml io.init_inputs 为空）

    返回 (out, ctx)；out 为 (M, N) 单张量，dtype 与 x 相同。
    """
    # 函数内 import：aiter 顶层 import 很重，真机部署走最小导入垫片
    from aiter.ops.triton.softmax import softmax

    if len(inputs) != 1:
        raise ValueError(f"2009_softmax 只接受 1 个输入张量，收到 {len(inputs)} 个")
    x = inputs[0]

    # 无超参数：任何传入的构造参数都说明口径不符，直接失败而不是静默忽略
    unexpected = set(init_kwargs or {})
    if unexpected:
        raise ValueError(
            f"2009_softmax 的 Model.__init__ 无参数（get_init_inputs()=[], "
            f"task.yaml io.init_inputs=[]），却收到 init_kwargs={sorted(unexpected)}"
        )

    if x.dim() != 2:
        raise ValueError(f"aiter softmax 要求 2D 输入 (M, N)，收到 {tuple(x.shape)}")
    if not x.is_cuda:
        raise ValueError("aiter softmax 要求输入位于 GPU（DCU）上")
    m, n = int(x.shape[0]), int(x.shape[1])
    if m < 1 or n < 1:
        raise ValueError(f"需 M >= 1 且 N >= 1，收到 (M, N)=({m}, {n})")

    # aiter 只按 x.stride(0) 逐行寻址，行内需连续；题面输入本就是 contiguous，
    # 非连续时显式转一次（仅 layout 转换，不改语义）
    if not x.is_contiguous():
        x = x.contiguous()

    out = softmax(x)

    if out.shape != x.shape or out.dtype != x.dtype:
        raise RuntimeError(
            f"aiter softmax 返回 ({tuple(out.shape)}, {out.dtype})，"
            f"与输入 ({tuple(x.shape)}, {x.dtype}) 不一致"
        )

    torch.cuda.synchronize()
    return out, {
        "path": "aiter.ops.triton.softmax.softmax -> _softmax_kernel_online",
        "layout_in": "bshd_n/a: 2D (M, N) row-wise, dim=-1",
        "M": m,
        "N": n,
        "dtype": str(x.dtype),
        "block_size": min(65536 // x.element_size(), _next_power_of_2(n)),
        "num_warps": 8,
        "num_stages": 2,
        "waves_per_eu": 2,
        "num_programs": m,
    }


def _next_power_of_2(n: int) -> int:
    """与 triton.next_power_of_2 等价（仅用于 ctx 记录，不参与计算）。"""
    return max(1, 1 << (int(n) - 1).bit_length())
