# 4010_gemm_a8w8_blockscale — 2D 分块 scale 反量化的 8bit GEMM（fp8 激活 ×
# fp8 权重，block scale 沿 K/N 两维分块）的 model_class（KernelBench 兼容）
# 题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=4010, backend=triton。

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """2D 分块 scale 反量化的 8bit GEMM（fp8 激活 × fp8 权重）。

    数学定义：给定 fp8(e4m3) 码字矩阵 xq (M, K)、wq (N, K) 与 float32 分块
    scale x_scale (M, ceil(K/gk))、w_scale (ceil(N/gn), ceil(K/gk))，输出
      dq_x[m, k] = xq[m, k] * x_scale[m, k // gk]
      dq_w[n, k] = wq[n, k] * w_scale[n // gn, k // gk]
      acc[m, n]  = sum_k dq_x[m, k] * dq_w[n, k]     （float32 累加）
      out[m, n]  = cast(acc[m, n], out_dtype)
    即每个元素先按它所属的 (m, k//gk) / (n//gn, k//gk) 块 scale 反量化，再做
    float32 GEMM（等价于 xq @ wq^T 的转置 GEMM，wq 为行主 (N, K)）。scale
    归属按元素逐个判定，与 K 方向 tile 划分无关；gk、gn 为 2 的幂（1 ~ 1024），
    K 不整除 gk / N 不整除 gn 时尾块内元素照常使用所在块的 scale（scale 个数
    为 ceil 除法结果）；M/N/K >= 1，允许非 2 次幂。反量化乘法与 GEMM 累加全程
    float32，输出最后 cast 到 out_dtype。

    输入输出规格：
      x       float8_e4m3fn (M, K)            激活码字（行主，TN 布局）
      w       float8_e4m3fn (N, K)            权重码字（行主，转置 GEMM）
      x_scale float32 (M, ceil(K/gk))         激活 scale：M 逐行、K 按 gk 分块
      w_scale float32 (ceil(N/gn), ceil(K/gk)) 权重 2D 分块 scale（行 = N 块，列 = K 块）
      out     out_dtype (M, N)                bfloat16 / float16，由 __init__ 指定
    在线评测器会把全部输入 cast 成 fp32 后传入 forward；fp8(e4m3) 的有限值在
    fp32 中精确表示，入口 .to(torch.float8_e4m3fn) 无损恢复码字。

    终审兼容形态：离线终审 harness 以单参 Model(case["head_size"]) 构造模型
    （case["head_size"] = [group_k, group_n, out_dtype]），故 __init__ 首参亦
    接受 [group_k, group_n, out_dtype] 序列并按序解包；ModelNew 须保持同样的
    双形态兼容（三参与单参序列均可构造）。

    实现约束（违规判负）：
      - 核心计算（8bit GEMM 累加 + 分块 scale 反量化）必须在提交文件内完成，
        禁止调用任何 ATen 矩阵乘捷径：torch.matmul / torch.bmm / torch.mm /
        torch.addmm / torch.einsum / torch._int_mm / torch._scaled_mm /
        F.linear / functional.linear，及其 torch.ops.aten.* 全限定名与张量
        方法形式（torch.ops.aten.mm、.mm( / .bmm( / .matmul( / .addmm( /
        .einsum( / ._int_mm( / ._scaled_mm( 等）、矩阵乘运算符 @ 与张量转置
        方法 .t()（x @ w.t() 即典型伪装写法；转置应在 kernel 内用 stride
        处理）。终审静态审计按上述字面串扫描提交文件（docstring 除外）。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间累加用 float32；输出 cast 到 out_dtype。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：单矩阵（无 batch 维）、无 bias、无 splitK；码字与 scale 值域沿用
    题面生成器（码字为 [0, 0.1) 的 fp8 值、scale 为 [0, 1) 的 float32）。
    """

    def __init__(self, group_k=128, group_n=128, out_dtype="bfloat16"):
        super().__init__()
        # 离线终审 harness 以单参构造（Model(case["head_size"])）；首参为
        # list/tuple 时按 [group_k, group_n, out_dtype] 解包
        if isinstance(group_k, (list, tuple)):
            group_k, group_n, out_dtype = group_k
        self.group_k = int(group_k)
        self.group_n = int(group_n)
        self.out_dtype = getattr(torch, out_dtype)

    def forward(self, x, w, x_scale, w_scale):
        # 评测器把全部输入 cast 成 fp32；fp8(e4m3) 有限值在 fp32 精确表示，
        # 入口 round-trip 无损恢复码字（离线路径直接传 fp8 张量时为 no-op）
        x = x.to(torch.float8_e4m3fn)
        w = w.to(torch.float8_e4m3fn)

        gk, gn = self.group_k, self.group_n
        M, K = x.shape
        N = w.shape[0]
        assert K == w.shape[1], "x 与 w 的 K 维必须一致"
        scale_k = (K + gk - 1) // gk
        scale_n = (N + gn - 1) // gn
        assert x_scale.shape == (M, scale_k), \
            f"x_scale 形状应为 ({M}, {scale_k})，实际 {tuple(x_scale.shape)}"
        assert w_scale.shape == (scale_n, scale_k), \
            f"w_scale 形状应为 ({scale_n}, {scale_k})，实际 {tuple(w_scale.shape)}"

        # 每个元素所属 scale 块：k -> k // gk、n -> n // gn（与 tile 划分无关，
        # 尾块部分元素照常取所在块 scale）
        k_idx = torch.arange(K, device=x.device) // gk      # (K,)
        n_idx = torch.arange(N, device=x.device) // gn      # (N,)
        xs = x_scale[:, k_idx]                              # (M, K) float32
        ws = w_scale[n_idx[:, None], k_idx[None, :]]        # (N, K) float32

        # 反量化与 GEMM 累加全程 float32
        dq_x = x.to(torch.float32) * xs
        dq_w = w.to(torch.float32) * ws
        out = F.linear(dq_x, dq_w)                          # (M, N) float32
        return out.to(self.out_dtype)


def get_init_inputs():
    return [128, 128, "bfloat16"]   # group_k；group_n；out_dtype


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性。码字与 scale 值域同 make_inputs。
    m, n, k = 128, 512, 1024
    group_k, group_n = 128, 128
    scale_k = (k + group_k - 1) // group_k
    scale_n = (n + group_n - 1) // group_n

    x = (torch.rand((m, k), dtype=torch.float16) / 10).to(torch.float8_e4m3fn)
    w = (torch.rand((n, k), dtype=torch.float16) / 10).to(torch.float8_e4m3fn)
    x_scale = torch.rand((m, scale_k), dtype=torch.float32)
    w_scale = torch.rand((scale_n, scale_k), dtype=torch.float32)
    return [x, w, x_scale, w_scale]


def make_inputs(m: int, n: int, k: int, group_k: int = 128, group_n: int = 128,
                out_dtype: str = "bfloat16", seed: int = 0,
                seq_lens=None, head_size=None, dtype=None):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    out_dtype 选择该 case 的输出 dtype（Model 构造参数），不改变输入张量本身。
    消费顺序固定：x -> w -> x_scale -> w_scale，全部来自同一
    torch.Generator(seed)，同 seed 下逐位可复现。

    离线终审 harness 兼容形参（audit_model_class 以 make_inputs(seq_lens=None,
    **case_fields) 统一调用，case 的额外字段一并转发）：seq_lens 本题不存在，
    须为 None；head_size/dtype 转发自 case，须与显式参数一致（一致性断言）。
    """
    assert seq_lens is None, "本题没有 seq_lens 维度"
    if head_size is not None:
        assert list(head_size) == [group_k, group_n, out_dtype], \
            f"head_size {head_size} 与 (group_k, group_n, out_dtype) 不一致"
    if dtype is not None:
        assert dtype == out_dtype, f"dtype {dtype} 与 out_dtype {out_dtype} 不一致"

    gen = torch.Generator().manual_seed(seed)

    scale_k = (k + group_k - 1) // group_k
    scale_n = (n + group_n - 1) // group_n

    x = (torch.rand((m, k), generator=gen, dtype=torch.float16) / 10).to(torch.float8_e4m3fn)
    w = (torch.rand((n, k), generator=gen, dtype=torch.float16) / 10).to(torch.float8_e4m3fn)
    x_scale = torch.rand((m, scale_k), generator=gen, dtype=torch.float32)
    w_scale = torch.rand((scale_n, scale_k), generator=gen, dtype=torch.float32)
    return x, w, x_scale, w_scale
