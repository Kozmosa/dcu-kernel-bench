# `kernelweaver-dcu-wiki` — DCU 公开 Wiki 语料 + 装载装配

DCU 侧的 `knowledge/kernelwiki_dcu/`（36 个页面 + 逐文件锁，40 个文件）、四个消融配置，以及让它们能被
装载的框架侧改动。

**语料是仓库里的普通文件，不是补丁**：直接看
`patches/kernelweaver-dcu-wiki/overlay/knowledge/kernelwiki_dcu/wiki/`，
例如 `wiki/contracts/compliance.md`、`wiki/hardware/gfx928.md`、`wiki/operators/attention.md`。
只有「修改框架既有文件」的部分是 patch。

## 目录

```
patches/kernelweaver-dcu-wiki/
├── 0001-dcu-wiki-gitattributes.patch   .gitattributes 追加一条 -text 规则
├── 0002-dcu-wiki-loader.patch          改 3 个框架既有文件（见下表）
├── overlay/                            逐字节拷入目标仓库根目录的文件
│   ├── knowledge/kernelwiki_dcu/       36 页 + corpus.lock.json（可直接阅读/编辑）
│   └── configs/search/dcu_*.yaml       4 个消融配置
└── README.md
```

| 补丁 | 改什么 | 为什么必须 |
|---|---|---|
| `0001-dcu-wiki-gitattributes` | `.gitattributes` 追加 `knowledge/kernelwiki_dcu/** -text` | 语料锁是**逐文件字节** SHA256；Windows 上 `core.autocrlf=true` 会把 JSON 改写成 CRLF，整份锁立刻失效（实测只有 `corpus.lock.json` 变 CRLF 就报 `Public Wiki changed since freeze`） |
| `0002-dcu-wiki-loader` | `src/knowledge/public_wiki.py`、`src/knowledge/domain_wiki.py`、`scripts/audit_public_wiki.py` | 没有它语料装了也加载不了：实测页 provenance（`local_measurement`+`script`，要求 `device/stack/measured_on` 与 `confidence: measured`）、`as_posix()` 键（Windows/Linux 哈希一致）、ROCm 来源白名单、`public-compliance` 页常驻上下文、多 backend 页面、冻结脚本的 `--wiki/--target-arch/--kernelbench-root`。另外修了检索的两个槽位 bug（见下） |

`overlay/` 里的东西**不进 patch**，因为它们是内容资产而不是代码 diff：`git apply` 会过 checkout
过滤器（CRLF 重写），而我们的语料锁要求字节稳定。脚本改成逐字节 `write_bytes` 拷贝，
装完立刻重算哈希对账。

## 检索槽位（`public_wiki.retrieve`）

`index.yaml` 的 `policy.max_entries: 4`，槽位分配是**保留槽 + 算子页**：

```
[算子页 × (max_entries - 2)] + [硬件页] + [合规页]
```

- 硬件页与合规页是保留槽，任何算子页都挤不掉。硬件页承载 wavefront 64、静态共享内存 65536 B、
  96 KiB 编译期失败这类**别处没有的实测事实**，挤掉它等于把这一轮最硬的 DCU 知识丢出上下文。
- 没有专门算子页时（按 API 符号与 family 规则都命不中）发 `public-contract` 通用检查单，它写明
  「没有专门页、按题面推语义、别自己编算子算法」，同时 `coverage: general_contract_only` 落在检索证据里。
- 之前的两处 bug 已修：`fallback[:1]` 按路径序取到的是合规页，导致通用检查单**永远发不出去**
  （11/68 题只拿到 2 条、缺那句提醒）；合规页插入时截断的正是硬件页，导致 26/68 题看不到硬件事实。

## 安装

```bash
python benchmark/tools/apply_framework_fixes.py kernelweaver-dcu-wiki /path/to/KernelWeaver --check
python benchmark/tools/apply_framework_fixes.py kernelweaver-dcu-wiki /path/to/KernelWeaver
```

顺序：校验基座 → 逐个 `git apply`（不做模糊匹配、失败即停）→ 拷 overlay（44 个文件）→
重算语料哈希并与 `corpus.lock.json` 对账（不一致直接报错退出，`--skip-verify` 可跳过）。

## 改语料的两条路径

1. **在 dcukb 里改**（推荐，单一事实来源）：直接编辑 `overlay/knowledge/kernelwiki_dcu/**`，然后
   `cd /path/to/KernelWeaver && python3 scripts/audit_public_wiki.py --wiki knowledge/kernelwiki_dcu --freeze`
   之前的做法是先把 overlay 装进检出再冻结；冻结后把新的汇总 SHA 写回本 README。
2. **在 KernelWeaver 里改，再拉回来**：

   ```bash
   python benchmark/tools/apply_framework_fixes.py kernelweaver-dcu-wiki /path/to/KernelWeaver --pull-overlay
   ```

   `--pull-overlay` 会先确认检出里的语料与其锁一致（否则拒绝，避免把半改状态拉回仓库），
   再把 `knowledge/kernelwiki_dcu/` 与 `configs/search/dcu_*.yaml` 覆盖回 `overlay/`。

## 校验（装完应当这样）

```bash
python3 -c "import sys;sys.path.insert(0,'.');from pathlib import Path;\
from src.knowledge.public_wiki import audit_public_corpus as a;\
print(a(Path('knowledge/kernelwiki_dcu'))['sha256'])"
# 期望：54b6fdbf6105003fafef93964216f7ee8d8459c631ed5c4f48486a392578b718
```

端到端（两轨检索、合规页与硬件页在上下文、无算子页时给通用检查单）：

```bash
python3 - <<'PY'
import sys; sys.path.insert(0, '.')
from pathlib import Path
from src.knowledge.public_wiki import audit_public_corpus, PublicKernelWiki
print('audit OK', audit_public_corpus(Path('knowledge/kernelwiki_dcu'), check_lock=True)['pages'])
w = PublicKernelWiki(Path('knowledge/kernelwiki_dcu'), target_arch='gfx928')
class T:  backend = 'triton'; reference_code = 'import torch\ns=torch.matmul(q,k)\nreturn torch.softmax(s,-1)'
print([e['id'] for e in w.retrieve(T())])          # attention / softmax / hardware / compliance
class U:  backend = 'triton'; reference_code = 'import torch\nreturn torch.gather(x, 0, i)'
print([e['id'] for e in w.retrieve(U())])          # indexing / hardware / compliance
class V:  backend = 'triton'; reference_code = 'import torch\nreturn a * b + c'
print([e['id'] for e in w.retrieve(V())])          # public-contract / hardware / compliance
PY
```

## 基座与可组合性

| 序列 | 目标仓库 | 基座 | 与本案 |
|---|---|---|---|
| `kernelweaver-dcu-wiki` | KernelWeaver | **`cd5d85c`** | — |
| `kernelweaver`（1116 契约） | KernelWeaver | `c11ab7f` | 与本案**可组合，但需要一次 rebase**：`0001` 在 `cd5d85c` 上带 offset 干净适用；`0002` 只有一个 hunk 撞（`src/core/workflow.py` 的 `_evaluate_plan_code` 调用点，要换成 `_evaluate_generated(...)` 包裹），手工 3-way 可解。两序列改的文件不重叠（本案改 `knowledge/` 与 `scripts/`，1116 改 `loader/evaluator/cli/workflow/io/models/validation`） |
| `pyramidkernel-1116` | PyramidKernel | `8352750` | 无重叠，可自由组合 |

实测依据：本案 `0002` 在 `c11ab7f` 上直接报 `src/knowledge/public_wiki.py: No such file or directory`；
在 `cd5d85c` 干净 worktree 上 `0001`+`0002`+overlay 装完，冻结语料可加载、检索矩阵符合预期。

## 语料版本

| 字段 | 值 |
|---|---|
| 汇总 SHA256 | `54b6fdbf6105003fafef93964216f7ee8d8459c631ed5c4f48486a392578b718` |
| 文件数 | 39（`index.yaml` + `data/` 2 + `sources/` 13 + `wiki/` 23）+ `corpus.lock.json` |
| 页面数 | 36（2 契约 + 1 硬件实测 + 20 算子 + 13 来源） |
| 覆盖 family | 20：matmul / elementwise / normalization / maxpool / avgpool / reduction / scan / convolution / transposed_convolution / cross_entropy / softmax / attention / rope / moe / quantization / indexing / topk / ctc_loss / deform_convolution / sparse_convolution |
| 冻结时间 | 2026-11-21（gfx928 迁移后第二次重冻） |
| 本轮内容改动 | 删 4 个永远命不中的归一化页（batchnorm/groupnorm/layernorm/rmsnorm），合并为 `normalization`（按 `mean`+`rsqrt` 算术命中，改写进 running-stats/group 映射/RMS 无均值等要点）；新增 rope / moe / quantization / indexing / topk / ctc_loss / deform_convolution / sparse_convolution 8 页；`matmul` 收 `einsum`、`attention` 收 `[einsum, softmax]`；`softmax` 收 `[exp, sum]`；新增来源页 `source-pytorch-functional`（PyTorch v2.6.0 固定 URL） |
| 目标架构 | **gfx928（K500SM_AI，120 CU）**；所有页面 `architectures: [gfx928]`，硬件页为 `gfx928.md`（实测）；配置里 `operator_memory_target_arch: gfx928` |

### 检索验收矩阵（2026-11-21 重测，68 题）

| 指标 | 改前 | 改后 |
|---|---|---|
| 拿到硬件页 | 31/68 | **68/68** |
| 无专门算子页（退通用检查单） | 11/68，且**拿不到**通用检查单 | 5/68，能拿到通用检查单 |
| 无算子页的题 | 0001、2003、2004、2008、3010、4018、5004、5007、6001、8002、8003 | 0001、2003、5007、6001、8003（纯算术，无 API 符号可判） |
| 明显误配 | `2009_softmax`→reduction；6 道 norm 题→reduction；1001/1002/1010/1021/1022/1029（einsum 写法）拿不到 attention 页；3003/3004/3006/3007（MoE）→attention 页 | 全部归位；MoE→moe 页、量化→quantization 页、norm→normalization 页、einsum attention→attention 页 |

复跑方法（开跑前必跑一遍，退出码 0 才算知识面就绪）：

```bash
python benchmark/tools/wiki_retrieval_matrix.py --kw-root ../KernelWeaver   --allow 0001_vecadd 2003_fused_mul_add 5007_spconv_indice_pairs 6001_fused_adamw 8003_scatter_nd
```

它直接调用框架自己的 `PublicKernelWiki.retrieve`，逐题打印 family 与命中的页面 id，并在
「有题拿不到硬件页」或「有题拿不到任何页面」时非零退出；`--allow` 声明哪些题按设计走通用检查单。

## 上限提醒

`index.yaml` 的 `policy.max_body_chars: 1800`。当前 `wiki/contracts/compliance.md` 正文 1703 字符、
`wiki/hardware/gfx928.md` 1401 字符，合规页已贴上限；再加内容要先删旧内容，否则检索证据里会被静默截断。
`max_entries: 4` 现在是「2 个算子页 + 硬件页 + 合规页」；要再放宽就得同步考虑上下文预算。
