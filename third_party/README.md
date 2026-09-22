# third_party/

外部仓库以 **git submodule** 注册——本仓库只记录 URL + commit（gitlink），**不 vendor 源码、不上传源码内容**。子仓内容仅供维护者审阅（语义依据、来源快照、官方性能基线），**不给生成 Agent**。

## 克隆本仓库

```bash
git clone --recurse-submodules git@github.com:Kozmosa/dcu-kernel-bench.git
# 或克隆后补拉：
git submodule update --init third_party/aiter
```

## third_party/aiter

- 上游：`https://developer.sourcefind.cn/codes/OpenDAS/aiter.git`（光合社区 GitLab，直连，注意 `/codes` 前缀），跟踪分支 `das-main`。
- **Windows 注意**：aiter 有两个文件名带冒号的 Triton autotune JSON，NTFS 无法 checkout。本子仓已配置 `core.protectNTFS=false` + sparse-checkout 排除（`/*` + `!**/*:*`），配置保存在本仓库 `.git/modules/` 内、**不随提交传播**——在别的 Windows 机器上重新 clone 后需重配一次（Linux 评测机无此问题）：

```bash
cd third_party/aiter
git config core.protectNTFS false
git sparse-checkout init --no-cone && git sparse-checkout set '/*' '!**/*:*'
git checkout -
```

- 因上述排除，子仓工作区常显 1 个删除状态的脏文件，属预期；`.gitmodules` 已设 `ignore = dirty`。**不要在该仓库内执行 `git reset --hard`**。
