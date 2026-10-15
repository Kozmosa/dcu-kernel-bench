# 7001_ctc_loss —— CTC 前向损失（手写 alpha/beta 前向-后向动态规划）的
# KernelBench 兼容题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=7001, backend=triton。

import torch
import torch.nn as nn


class Model(nn.Module):
    """CTC（Connectionist Temporal Classification）前向损失：一次前向同时算出
    逐序列 CTC 损失与对输入概率的梯度。输入 probs 是**概率**而不是 logits
    （等价于生产库 apply_softmax=False 口径：直接对 probs 取 log，不做
    softmax），沿类别维每帧为正数且和为 1；blank 类别 id 由超参给出
    （生成域恒为 0）。

    布局与输入契约（time-major 输入）：
      probs          (T, N, C) 浮点概率张量，T=max_time、N=batch、C=num_classes
      labels         (sum(label_lengths),) 整型，按 batch 顺序拼接的扁平标签
                     序列，标签取值 [0, C-1] 且不等于 blank（生成域为 [1, C-1]）
      label_lengths  (N,) 整型，各序列目标长度
      input_lengths  (N,) 整型，各序列有效输入帧数（1 <= input_lengths[n] <= T）
    数学定义（对每个序列 n，记 Tb=input_lengths[n]、目标 l_1..l_L
    （L=label_lengths[n]），扩展状态序列 ext = [b, l_1, b, l_2, ..., b, l_L, b]
    （前后及每两个标签之间插入 blank），S = 2L+1 个状态，s = 0..S-1；
    全部递推在 log 空间、float32 中进行，-inf 表示不可达）：
      前向变量 alpha（尺寸 (Tb, S)）：
        alpha[0,0] = log p[0, blank]；alpha[0,1] = log p[0, ext[1]]（S>1）；
        其余 -inf
        alpha[t,s] = log p[t, ext[s]] + LSE( alpha[t-1,s], alpha[t-1,s-1],
                     alpha[t-1,s-2]（仅当 ext[s] != blank 且 ext[s] != ext[s-2]） )
      后向变量 beta（对称；beta 不含本帧发射，使 alpha+beta 对每条过状态 s
      的完整对齐路径恰好把每个发射因子计一次）：
        beta[Tb-1,S-1] = beta[Tb-1,S-2] = 0（S>1），其余 -inf
        beta[t,s] = LSE( log p[t+1,ext[s]] + beta[t+1,s],
                         log p[t+1,ext[s+1]] + beta[t+1,s+1],
                         log p[t+1,ext[s+2]] + beta[t+1,s+2]
                         （最后一项仅当 ext[s] != blank 且 ext[s] != ext[s+2]） )
      losses[n] = -log Z，Z = LSE( alpha[Tb-1,S-1], alpha[Tb-1,S-2] )
        （全部合法对齐路径的概率和；reduction 为逐序列 'none'）
      gamma[t,s] = exp( alpha[t,s] + beta[t,s] - log Z )
      gradients[n,t,k] = d losses[n] / d probs[n,t,k]
        = -(1 / p[t,k]) * sum_{s: ext[s]=k} gamma[t,s]，t < Tb；t >= Tb 处为 0
    forward 返回**单个一维张量** out（dtype 同 probs，单张量打包协议）：
      out[0:N]         = losses（N 项）
      out[N : N+N*T*C] = gradients 按 (N, T, C) 行主序平铺
    边界行为：
      - 生成域保证 probs 严格为正（softmax 口径）、标签无连续重复且
        L <= Tb，故 Z 有限，无 inf/NaN；losses 量级 <= T*ln(C)，fp16 可表示。
      - 不可达状态的 alpha/beta 为 -inf，gamma 对应为 0。
      - 梯度对概率输入定义（非对 log 概率）：链式因子 1/p[t,k] 已含在定义中。

    输入输出规格：
      probs          float32 / float16 / bfloat16 (T, N, C)，contiguous
      labels / label_lengths / input_lengths   整型 (…,)，contiguous
      out            单张量、一维、同 probs dtype、长度 N + N*T*C
      评测器会把全部输入 cast 成 fp32 传入（整型张量需自行恢复为 long，
      整数值在 fp32 下可无损表示）。

    实现约束（违规判负）：
      - 核心计算（log 概率、alpha/beta 前向-后向递推、梯度组装）必须在提交
        文件内以 Triton kernel 完成。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间计算在 float32 中进行（先把 probs cast 到 fp32 再取 log），输出
        cast 回 probs dtype 后按上述打包协议拼接为单张量。
      - forward 返回单张量 out（评测器按单输出比较）。

    禁用列表：aiter / torch.nn.functional.ctc_loss / torch.ctc_loss /
    F.ctc_loss / torch.nn.CTCLoss / torch.ops.aten.ctc_loss /
    torch.ops.aten._ctc_loss / aten.ctc_loss / .ctc_loss(（含一切库级一步
    CTC 原语） / warpctc / warp_ctc / torch.autograd 与 .backward()（禁止
    以自动微分求梯度）。
    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：单 blank、reduction='none'（逐序列损失 + 全量梯度）、
    输入已是概率（无 apply_softmax 分支）、无空目标序列。
    """

    def __init__(self, blank: int = 0):
        super().__init__()
        self.blank = int(blank)

    def _ctc_one(self, logp_b, p_b, target, tb, num_classes, max_time):
        # 单序列 log 空间前向-后向：logp_b / p_b 为 (T, C) fp32 的 log 概率 /
        # 概率，target (L,) long，tb 为有效帧数；返回 (loss_n, grad_n)，
        # grad_n 为 (max_time, C)、有效帧数之外恒 0
        C = num_classes
        L = target.numel()
        S = 2 * L + 1
        neg_inf = float("-inf")
        # device=dev 必需：本函数内构造的所有中间张量都要与已在设备上的 logp_b/p_b
        # 同设备；漏 device 会在 CUDA/DCU 上抛 "Expected all tensors to be on the
        # same device"（CPU 上跑得通，所以本地单测发现不了）
        dev = logp_b.device

        ext = torch.empty(S, dtype=torch.long, device=dev)
        ext[0::2] = self.blank
        ext[1::2] = target

        lp_ext = logp_b[:tb, ext]                      # (tb, S)
        alpha = torch.full((tb, S), neg_inf, dtype=torch.float32, device=dev)
        beta = torch.full((tb, S), neg_inf, dtype=torch.float32, device=dev)

        # 前向递推（时间维步进，状态维整体向量化）
        alpha[0, 0] = lp_ext[0, 0]
        alpha[0, 1] = lp_ext[0, 1]
        for t in range(1, tb):
            prev = alpha[t - 1]
            left = torch.full((S,), neg_inf, dtype=torch.float32, device=dev)
            left[1:] = prev[:-1]
            jump = torch.full((S,), neg_inf, dtype=torch.float32, device=dev)
            jump[2:] = prev[:-2]
            ok = (ext[2:] != self.blank) & (ext[2:] != ext[:-2])
            jump[2:] = torch.where(ok, jump[2:], torch.full((S - 2,), neg_inf, dtype=torch.float32, device=dev))
            alpha[t] = lp_ext[t] + torch.logsumexp(torch.stack((prev, left, jump)), dim=0)

        # 后向递推（beta 不含本帧发射：alpha+beta 对过状态 s 的完整路径
        # 恰好计一次每个发射因子，gamma 即状态后验）
        beta[tb - 1, S - 1] = 0.0
        beta[tb - 1, S - 2] = 0.0
        for t in range(tb - 2, -1, -1):
            nxt = beta[t + 1] + lp_ext[t + 1]
            right = torch.full((S,), neg_inf, dtype=torch.float32, device=dev)
            right[:-1] = nxt[1:]
            jump = torch.full((S,), neg_inf, dtype=torch.float32, device=dev)
            jump[:-2] = nxt[2:]
            ok = (ext[:-2] != self.blank) & (ext[:-2] != ext[2:])
            jump[:-2] = torch.where(ok, jump[:-2], torch.full((S - 2,), neg_inf, dtype=torch.float32, device=dev))
            beta[t] = torch.logsumexp(torch.stack((nxt, right, jump)), dim=0)

        log_z = torch.logsumexp(alpha[tb - 1] + beta[tb - 1], dim=0)
        gamma = torch.exp(alpha + beta - log_z)        # (tb, S) 状态后验

        # dL/dp[t,k] = -(1/p[t,k]) * sum_{s: ext[s]=k} gamma[t,s]
        grad = torch.zeros(tb, C, dtype=torch.float32, device=dev)
        grad.index_add_(1, ext, -gamma)
        grad = grad / p_b[:tb]                         # 除以 p（probs 严格为正）

        grad_full = torch.zeros(max_time, C, dtype=torch.float32, device=dev)
        grad_full[:tb] = grad
        return -log_z, grad_full

    def forward(self, probs, labels, label_lengths, input_lengths):
        # 在线评测器会把整型输入 cast 成 fp32；整数值可无损恢复
        labels = labels.to(torch.long)
        label_lengths = label_lengths.to(torch.long)
        input_lengths = input_lengths.to(torch.long)

        pf = probs.to(torch.float32)
        max_time, batch, num_classes = pf.shape
        logp = torch.log(pf)

        losses = torch.empty(batch, dtype=torch.float32, device=pf.device)
        grads = torch.zeros(batch, max_time, num_classes, dtype=torch.float32, device=pf.device)
        offset = 0
        for n in range(batch):
            tb = int(input_lengths[n])
            L = int(label_lengths[n])
            target = labels[offset:offset + L]
            offset += L
            loss_n, grad_n = self._ctc_one(logp[:, n, :], pf[:, n, :], target, tb,
                                           num_classes, max_time)
            losses[n] = loss_n
            grads[n] = grad_n

        # 单张量打包：losses (N,) 在前，gradients (N, T, C) 行主序平铺在后，
        # cast 回 probs dtype 后拼接为单个一维张量
        losses = losses.to(probs.dtype).reshape(-1)
        grads = grads.to(probs.dtype).reshape(-1)
        return torch.cat((losses, grads))


def get_init_inputs():
    return [0]   # blank（生成域恒为 0，与生产库 blank_label_id=0 一致）


def get_inputs():
    # 固定 shape 族（与生产样例调用契约一致）；随机部分消费全局 RNG——probs 由
    # randn 经 softmax 得到合法概率（沿类别维和为 1、严格为正）。评测器在
    # set_seed 后调用本函数，多轮 correctness trial 因此获得输入多样性
    max_time, batch, num_classes = 500, 4, 5
    input_lengths = [4, 100, 100, 200]
    label_lengths = [1, 2, 1, 3]
    labels = [1, 2, 3, 4, 2, 3, 2]

    probs = torch.softmax(torch.randn(max_time, batch, num_classes), dim=-1)
    return [
        probs,
        torch.tensor(labels, dtype=torch.int64),
        torch.tensor(label_lengths, dtype=torch.int64),
        torch.tensor(input_lengths, dtype=torch.int64),
    ]


def make_inputs(max_time, batch, num_classes, input_lengths, label_lengths,
                labels, blank=0, dtype="float32", seed=0):
    """按公开/隐藏/性能案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    消费顺序固定：probs（randn 后 softmax 沿类别维归一化，再 cast 到 dtype）
    为唯一随机量，labels / label_lengths / input_lengths 为案例给定的确定
    值、不消费随机数；同 seed 下逐位可复现。blank 同时是构造 Model 的 init
    超参，此处校验标签域与可行性（标签 != blank、无连续重复、
    1 <= L <= 有效帧数 <= max_time、num_classes >= 2）。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert max_time >= 1 and batch >= 1, "需 max_time >= 1 且 batch >= 1"
    assert num_classes >= 2, "至少需要 blank + 1 个非空类别（num_classes >= 2）"
    assert 0 <= blank < num_classes, "blank 必须是合法类别 id"
    assert len(input_lengths) == batch and len(label_lengths) == batch, "长度向量须与 batch 一致"
    assert len(labels) == sum(label_lengths), "labels 长度须等于 sum(label_lengths)"
    for il, ll in zip(input_lengths, label_lengths):
        assert 1 <= ll <= il <= max_time, "需 1 <= label_length <= input_length <= max_time"
    pos = 0
    for ll in label_lengths:
        seg = labels[pos:pos + ll]
        pos += ll
        for a, b in zip(seg, seg[1:]):
            assert a != b, "生成域保证标签无连续重复"
        assert all(v != blank for v in seg), "标签不得等于 blank"

    probs = torch.softmax(torch.randn(max_time, batch, num_classes, generator=gen), dim=-1).to(dt)
    return (
        probs,
        torch.tensor(labels, dtype=torch.int64),
        torch.tensor(label_lengths, dtype=torch.int64),
        torch.tensor(input_lengths, dtype=torch.int64),
    )
