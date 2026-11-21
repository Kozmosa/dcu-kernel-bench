# `kernelweaver-dcu-wiki` — DCU 公开 Wiki 语料 + 装载装配

DCU 侧的 `knowledge/kernelwiki_dcu/`（30 个页面 + 逐文件锁，34 个文件）、四个消融配置，以及让它们能被
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
│   ├── knowledge/kernelwiki_dcu/       30 页 + corpus.lock.json（可直接阅读/编辑）
│   └── configs/search/dcu_*.yaml       4 个消融配置
└── README.md
```

| 补丁 | 改什么 | 为什么必须 |
|---|---|---|
| `0001-dcu-wiki-gitattributes` | `.gitattributes` 追加 `knowledge/kernelwiki_dcu/** -text` | 语料锁是**逐文件字节** SHA256；Windows 上 `core.autocrlf=true` 会把 JSON 改写成 CRLF，整份锁立刻失效（实测只有 `corpus.lock.json` 变 CRLF 就报 `Public Wiki changed since freeze`） |
| `0002-dcu-wiki-loader` | `src/knowledge/public_wiki.py`、`src/knowledge/domain_wiki.py`、`scripts/audit_public_wiki.py` | 没有它语料装了也加载不了：实测页 provenance（`local_measurement`+`script`，要求 `device/stack/measured_on` 与 `confidence: measured`）、`as_posix()` 键（Windows/Linux 哈希一致）、ROCm 来源白名单、`public-compliance` 页常驻上下文、多 backend 页面、冻结脚本的 `--wiki/--target-arch/--kernelbench-root` |

`overlay/` 里的东西**不进 patch**，因为它们是内容资产而不是代码 diff：`git apply` 会过 checkout
过滤器（CRLF 重写），而我们的语料锁要求字节稳定。脚本改成逐字节 `write_bytes` 拷贝，
装完立刻重算哈希对账。

## 安装

```bash
python benchmark/tools/apply_framework_fixes.py kernelweaver-dcu-wiki /path/to/KernelWeaver --check
python benchmark/tools/apply_framework_fixes.py kernelweaver-dcu-wiki /path/to/KernelWeaver
```

顺序：校验基座 → 逐个 `git apply`（不做模糊匹配、失败即停）→ 拷 overlay（38 个文件）→
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
# 期望：b75296feeba7a9664826d47882bf8c5e3f33d065aa5a9489524678855451649a
```

端到端（两轨检索、合规页在上下文、sm86 被拒）：

```bash
python3 - <<'PY'
import sys; sys.path.insert(0, '.')
from pathlib import Path
from src.knowledge.public_wiki import audit_public_corpus, PublicKernelWiki
print('audit OK', audit_public_corpus(Path('knowledge/kernelwiki_dcu'), check_lock=True)['pages'])
w = PublicKernelWiki(Path('knowledge/kernelwiki_dcu'), target_arch='gfx928')
class T:  backend = 'triton'; reference_code = 'import torch\ns=torch.matmul(q,k)\nreturn torch.softmax(s,-1)'
ev = w.retrieve(T()); print([e['id'] for e in ev], 'Platform symbols' in str(ev))
PY
```

## 基座与可组合性（重要）

| 序列 | 目标仓库 | 基座 | 与本案 |
|---|---|---|---|
| `kernelweaver-dcu-wiki` | KernelWeaver | **`cd5d85c`** | — |
| `kernelweaver` | KernelWeaver | `c11ab7f` | **暂时互斥**：该序列按 `c11ab7f` 生成，撞在 `src/cli.py` / `src/core/workflow.py` 上；而本案依赖 `cd5d85c` 才引入的 `src/knowledge/public_wiki.py` 与 `.gitattributes`。要同时装，需先把代码序列 rebase 到 ≥ `cd5d85c` |
| `pyramidkernel-1116` | PyramidKernel | `8352750` | 无重叠，可自由组合 |

实测依据：`patches/kernelweaver/0001-0002` 在 `c11ab7f` 干净 worktree 上两补丁都能打上；本案的 `0002`
在 `c11ab7f` 上直接报 `src/knowledge/public_wiki.py: No such file or directory`。

## 语料版本

| 字段 | 值 |
|---|---|
| 汇总 SHA256 | `b75296feeba7a9664826d47882bf8c5e3f33d065aa5a9489524678855451649a` |
| 文件数 | 33（`index.yaml` + `data/` 2 + `sources/` 12 + `wiki/` 17）+ `corpus.lock.json` |
| 页面数 | 30（含 gfx928 实测硬件页） |
| 冻结时间 | 2026-11-21（迁移到 gfx928 后重新冻结） |
| 本轮内容改动 | `wiki/contracts/compliance.md`（平台符号类 / 别名解析 / 审计前置）、`wiki/hardware/gfx928.md`（hipcc 可用与不可用 API 面、Triton libdevice 路径、实测资源上限）、`wiki/contracts/general.md`（原生 dtype + 任务容差） |
| 目标架构 | **gfx928（K500SM_AI，120 CU）**；所有页面 `architectures: [gfx928]`，硬件页为 `gfx928.md`（实测）；配置里 `operator_memory_target_arch: gfx928` |

## 上限提醒

`index.yaml` 的 `policy.max_body_chars: 1800`。当前 `wiki/contracts/compliance.md` 1788 字符、
`wiki/hardware/gfx928.md` 约 1730 字符，已贴上限；再加内容要先删旧内容，否则检索证据里会被静默截断。
另外 `max_entries: 3` 且合规页常驻——题面命中 2 个算子页时硬件页会被挤出去（实测 attention 题如此）。
要保证硬件页常在，需要把 `max_entries` 提到 4，或把最关键的硬件约束并进合规页。
