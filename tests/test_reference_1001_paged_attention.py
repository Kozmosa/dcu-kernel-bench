"""1001_paged_attention — reference 语义测试。

naive_attention 是独立 oracle：逐 token 经块表取 K/V、float32 softmax，
与 aiter 官方测试 op_tests/triton_tests/test_pa_decode.py 的
paged_attention_decode_ref 语义一致（实现独立重写，不共享代码）。
"""
import math

import pytest
import torch

from conftest import load_reference

ref = load_reference("1001_paged_attention")


def naive_attention(query, key_cache, value_cache, block_tables, seq_lens, scale):
    B, H_Q, D = query.shape
    H_KV = key_cache.shape[1]
    bs = key_cache.shape[2]
    out = torch.empty_like(query)
    for s in range(B):
        seq_len = int(seq_lens[s])
        keys, values = [], []
        for j in range(seq_len):
            blk = int(block_tables[s, j // bs])
            off = j % bs
            keys.append(key_cache[blk, :, off, :].float())     # [H_KV, D]
            values.append(value_cache[blk, :, off].float())
        k = torch.stack(keys)                                  # [L, H_KV, D]
        v = torch.stack(values)
        for h in range(H_Q):
            hkv = h // (H_Q // H_KV)
            logits = torch.tensor([
                scale * float(query[s, h].float() @ k[j, hkv]) for j in range(seq_len)
            ])
            p = torch.softmax(logits, dim=-1)
            out[s, h] = sum(p[j] * v[j, hkv] for j in range(seq_len))
    return out.to(query.dtype)


CASES = [
    dict(B=3, H_Q=6,  H_KV=2, D=16, bs=4,  lens=[1, 5, 8],    dtype=torch.float32),   # GQA group=3、跨块/尾块
    dict(B=3, H_Q=4,  H_KV=4, D=32, bs=16, lens=[1, 16, 17],  dtype=torch.float32),   # MHA、恰好整块
    dict(B=2, H_Q=8,  H_KV=2, D=96, bs=8,  lens=[9, 33],      dtype=torch.float32),   # 非 2 次幂 head_size
    dict(B=2, H_Q=4,  H_KV=4, D=32, bs=1,  lens=[1, 7],       dtype=torch.float32),   # block_size=1
    dict(B=2, H_Q=4,  H_KV=2, D=32, bs=8,  lens=[9, 17],      dtype=torch.float16),   # 半精度（任务容差）
    dict(B=2, H_Q=4,  H_KV=2, D=32, bs=8,  lens=[9, 17],      dtype=torch.bfloat16),
]


def build(case, seed):
    g = torch.Generator().manual_seed(seed)
    c = dict(case)
    lens, dtype = c.pop("lens"), c.pop("dtype")
    B, H_Q, H_KV, D, bs = c["B"], c["H_Q"], c["H_KV"], c["D"], c["bs"]
    nb = (max(lens) + bs - 1) // bs
    pool = B * nb + 8                                   # 垃圾槽位：未引用块
    bt = torch.randperm(pool, generator=g)[: B * nb].reshape(B, nb).to(torch.int32)
    q = torch.randn(B, H_Q, D, generator=g).to(dtype)
    kc = torch.randn(pool, H_KV, bs, D, generator=g).to(dtype)
    vc = torch.randn(pool, H_KV, bs, D, generator=g).to(dtype)
    return q, kc, vc, bt, torch.tensor(lens, dtype=torch.int32)


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"H{c['H_Q']}kv{c['H_KV']}D{c['D']}bs{c['bs']}")
def test_reference_matches_naive_oracle(case):
    q, kc, vc, bt, lens = build(case, seed=42)
    scale = 1.0 / math.sqrt(case["D"])
    out = ref.reference(q, kc, vc, bt, lens, scale)
    expected = naive_attention(q, kc, vc, bt, lens, scale).float()
    assert out.shape == q.shape and out.dtype == q.dtype
    # fp32 用紧容差验证数学；半精度与 task.yaml 一致（一个 ULP 的求和顺序差异是正常的）
    if case["dtype"] == torch.float32:
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
    else:
        torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)


def test_scale_default_is_inv_sqrt_d():
    q, kc, vc, bt, lens = build(CASES[0], seed=1)
    D = CASES[0]["D"]
    a = ref.reference(q, kc, vc, bt, lens, 1.0 / math.sqrt(D))
    b = ref.reference(q, kc, vc, bt, lens, None)
    torch.testing.assert_close(a, b)


def test_make_inputs_determinism_and_garbage_pool():
    args = (4, 8, 8, 64, 16, 200, "float16", 7)
    q1, kc1, vc1, bt1, l1 = ref.make_inputs(*args)
    q2, kc2, vc2, bt2, l2 = ref.make_inputs(*args)
    assert torch.equal(q1, q2) and torch.equal(kc1, kc2) and torch.equal(bt1, bt2)
    nb = (200 + 15) // 16
    assert kc1.shape[0] == 4 * nb + 8          # 存在未引用垃圾块
    assert l1.dtype == torch.int32 and (l1 >= 1).all() and (l1 <= 200).all()


def test_make_inputs_explicit_and_broadcast_seq_lens():
    q, kc, vc, bt, lens = ref.make_inputs(4, 8, 8, 64, 16, 128, "bfloat16", 3, seq_lens=[128])
    assert lens.tolist() == [128] * 4           # 单值广播
    _, _, _, _, l2 = ref.make_inputs(2, 8, 8, 64, 16, 128, "bfloat16", 3, seq_lens=[1, 128])
    assert l2.tolist() == [1, 128]
