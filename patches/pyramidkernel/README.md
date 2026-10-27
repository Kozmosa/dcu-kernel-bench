# PyramidKernel 的本地补丁（不含上游收录的提交）

`PyramidKernel`（`https://github.com/yijiaduan111/PyramidKernel`）是**上游仓库**，
本评测集对它的改动**不提交、不推送到上游**，而是以补丁序列的形式留在这里：

```
patches/pyramidkernel/
  0001-fix-cuda-DCU-dcc-use_fast_math.patch          # dcc 不认 --use_fast_math，按平台条件化
  0002-fix-loader.patch                              # 多行签名截断 + 语句块缩进错乱
  0003-fix-loader-Model-root-AttributeError-NameError.patch  # 补齐辅助方法与模块级定义
  0004-fix-prompt-ROCm-agent-evaluated-on-CUDA-native-cuda-.patch  # ROCm 上别再说 CUDA + native-cuda 补 HIP 提示
  0005-fix-provider-body-JSON.patch                  # 空 body/JSON 解析失败算可重试 + 开题前探活
  0006-fix-provider.patch                            # 只有「连不上」才中止分片，读超时只告警
```

基线：`origin/main` = `da26b70`（2026-09-04）。

## 用法

```bash
# 1) 拿一份干净的上游
git clone https://github.com/yijiaduan111/PyramidKernel /root/dcudeploy/PyramidKernel
# 2) 打补丁（顺序即文件名顺序）
cd /root/dcudeploy/PyramidKernel
git apply /path/to/dcu-kernel-bench/patches/pyramidkernel/*.patch
# 或一次性：bash benchmark/tools/apply_pyramid_patches.sh /root/dcudeploy/PyramidKernel
```

脚本会先 `git apply --check` 再 `--3way` 应用，并在目标是 git 仓库时打印应用后的 HEAD。

## 为什么不用 fork

上游只有 `main` 一个分支、没有改动需求提交通道；这 6 个补丁都是**DCU 环境适配**
（hipcc/`--use_fast_math`/锚点/重试判定），对 CUDA 上游未必通用。以补丁形式保存：
- 上游仓库保持干净，随时能 `git pull` 跟上游；
- dcukb 里带着"评测需要的框架改动"这份**可审查的差异**；
- 想 fork 时直接 `git am` 到自己的 fork，一样的六条提交。
