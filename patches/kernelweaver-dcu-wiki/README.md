# `kernelweaver-dcu-wiki` — DCU 公开 Wiki 语料 + 装载装配（外部补丁）

把 DCU 侧的 `knowledge/kernelwiki_dcu/`（30 个页面 + 逐文件锁，34 个文件）连同**让它能装载的装配**
装进 KernelWeaver 快照：`knowledge/`、`src/knowledge/`、`configs/search/dcu_*.yaml`、`.gitattributes`。
不碰评测器、provider、CLI，因此与 `patches/pyramidkernel-1116/`（另一个仓库）天然不冲突。

## 基座与可组合性（重要）

| 序列 | 目标仓库 | 基座 | 与本案 |
|---|---|---|---|
| `kernelweaver-dcu-wiki` | KernelWeaver | **`cd5d85c`** | — |
| `kernelweaver` | KernelWeaver | `c11ab7f` | **暂时互斥**：该序列按 `c11ab7f` 生成，撞在 `src/cli.py` / `src/core/workflow.py` 上，而本语料依赖 `cd5d85c` 才引入的 `src/knowledge/public_wiki.py` 与 `.gitattributes`。要同时装，需先把代码序列 rebase 到 ≥ `cd5d85c` |
| `pyramidkernel-1116` | PyramidKernel | `8352750` | 无重叠，可自由组合 |

实测：`patches/kernelweaver/0001-0002` 在 `c11ab7f` 干净 worktree 上两补丁都能打上；本案的
`0002` 在 `c11ab7f` 上直接报 `src/knowledge/public_wiki.py: No such file or directory`。


## 四个补丁（按文件名顺序应用）

| 补丁 | 内容 | 为什么需要 |
|---|---|---|
| `0001-dcu-wiki-gitattributes` | `.gitattributes` 追加 `knowledge/kernelwiki_dcu/** -text` | 语料锁是**逐文件字节** SHA256；Windows 上 `core.autocrlf=true` 会把 JSON 改成 CRLF，哈希整份失效（实测：只改了 `corpus.lock.json`，审计直接 `Public Wiki changed since freeze`）。`-text` 让整个语料按字节存取 |
| `0002-dcu-wiki-loader` | `src/knowledge/public_wiki.py`、`src/knowledge/domain_wiki.py`、`scripts/audit_public_wiki.py` | ① 允许 `origin: local_measurement` + `reproducibility: script` 的**实测页**（要求 `device/stack/measured_on` + `confidence: measured`）；② 页面路径与 manifest 键统一 `as_posix()`（Windows/Linux 哈希一致）；③ 语料来源白名单加 ROCm HIP 文档 URL；④ `public-compliance` 页**常驻**上下文；⑤ 多 backend（cuda + triton）页面；⑥ 冻结脚本支持 `--wiki/--target-arch/--kernelbench-root` |
| `0003-dcu-wiki-configs` | `configs/search/dcu_{a_short,b_longterm,c_wiki,d_combined}.yaml` | 四个消融配置指向该语料（`operator_memory_path: knowledge/kernelwiki_dcu`、`operator_memory_target_arch: gfx936`）。没有它们，语料装了也用不上 |
| `0004-dcu-wiki-corpus` | `knowledge/kernelwiki_dcu/**`（30 页 + `corpus.lock.json`） | 语料本体 |

## 安装

```bash
python benchmark/tools/apply_framework_fixes.py kernelweaver-dcu-wiki /path/to/KernelWeaver --check
python benchmark/tools/apply_framework_fixes.py kernelweaver-dcu-wiki /path/to/KernelWeaver
```

基座：`cd5d85c` 必须是目标 HEAD 的祖先（该提交才引入 `src/knowledge/public_wiki.py`、
`src/knowledge/domain_wiki.py`、`scripts/audit_public_wiki.py` 与 `.gitattributes`）。补丁逐个 `git apply`，
不做模糊匹配、失败即停；装完脚本会**重算语料哈希并与 `corpus.lock.json` 对账**，不一致直接报错退出
（`--skip-verify` 可跳过）。

## 校验（装完应当这样）

```bash
python3 -c "import sys;sys.path.insert(0,'.');from pathlib import Path;\
from src.knowledge.public_wiki import audit_public_corpus as a;\
print(a(Path('knowledge/kernelwiki_dcu'))['sha256'])"
# 期望：98f5d2e9ffcd45ed248918cdff62fcba1463df10394c1b58808812a8a31612b0
```

更强的端到端检查（两个框架轨都能检索、sm86 语料在 DCU 上被拒、审计页进了上下文）：

```bash
python3 - <<'PY'
import sys; sys.path.insert(0, '.')
from pathlib import Path
from src.knowledge.public_wiki import audit_public_corpus, PublicKernelWiki
r = audit_public_corpus(Path('knowledge/kernelwiki_dcu'), check_lock=True)
print('audit OK', r['pages'], r['sha256'][:16])
w = PublicKernelWiki(Path('knowledge/kernelwiki_dcu'), target_arch='gfx936')
class T:  backend = 'triton'; reference_code = 'import torch\ns=torch.matmul(q,k)\nreturn torch.softmax(s,-1)'
print([e['id'] for e in w.retrieve(T())])
PY
```

KernelWeaver 自带的 `tests/test_dcu_adaptation.py::test_frozen_dcu_wiki_serves_both_tracks_and_rejects_other_arch`
覆盖同一件事；注意该测试文件本身**不在** KernelWeaver 主干里（随 DCU 适配层一起本地维护），所以本补丁不包含它。

## 语料版本

| 字段 | 值 |
|---|---|
| 汇总 SHA256 | `98f5d2e9ffcd45ed248918cdff62fcba1463df10394c1b58808812a8a31612b0` |
| 文件数 | 33（`index.yaml` + `data/` 2 + `sources/` 12 + `wiki/` 17）+ `corpus.lock.json` |
| 页面数 | 30（另有 `data/` 两个检索表） |
| 冻结时间 | 2026-11-21 |
| 本轮内容改动 | `wiki/contracts/compliance.md`（平台符号类 / 别名解析 / 审计前置）、`wiki/hardware/gfx936.md`（hipcc 可用与不可用 API 面、Triton libdevice 路径）、`wiki/contracts/general.md`（原生 dtype + 任务容差） |

## 改语料后重新冻结

```bash
cd /path/to/KernelWeaver
python3 scripts/audit_public_wiki.py --wiki knowledge/kernelwiki_dcu --freeze
```

`--wiki` 必须显式给 DCU 路径：脚本默认语料是 NVIDIA 侧的 `knowledge/kernelwiki_public`，不加参数会把
那个语料的 `frozen_at` 一起刷新（不会改内容，但会污染无关 diff）。冻结后把新 SHA 同步写回本 README，
否则下一个人对着旧期望值会以为装坏了。

## 上限提醒

`index.yaml` 的 `policy.max_body_chars: 1800`。当前 `wiki/contracts/compliance.md` 1788 字符、
`wiki/hardware/gfx936.md` 1793 字符，已贴上限；再加内容要先删旧内容，否则检索证据里会被静默截断。
另外 `max_entries: 3` 且合规页常驻——当题面命中 2 个算子页时，硬件页会被挤出去（实测 attention 题就是
这种情况）。要保证硬件页常在，需要把 `max_entries` 提到 4，或在合规页里保留最关键的硬件约束。
