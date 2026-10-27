#!/usr/bin/env bash
# 把 patches/pyramidkernel/*.patch 应用到一份 PyramidKernel 检出上。
#
# 上游仓库不动：本评测集对 PyramidKernel 的改动只以补丁形式随 dcukb 提供
# （见 patches/pyramidkernel/README.md）。补丁之间有上下文依赖，必须按文件名顺序应用。
#
#   bash benchmark/tools/apply_pyramid_patches.sh <path>          # 应用
#   bash benchmark/tools/apply_pyramid_patches.sh <path> --check  # 在临时 worktree 里按序试应用
set -uo pipefail

TARGET=${1:-}
MODE=${2:-}
HERE=$(cd "$(dirname "$0")/../.." && pwd)
PATCH_DIR="$HERE/patches/pyramidkernel"

if [ -z "$TARGET" ] || [ ! -d "$TARGET" ]; then
  echo "usage: $0 <PyramidKernel-checkout> [--check]" >&2
  exit 2
fi
[ -d "$PATCH_DIR" ] || { echo "[FAIL] 缺补丁目录 $PATCH_DIR" >&2; exit 1; }

shopt -s nullglob
patches=("$PATCH_DIR"/*.patch)
[ ${#patches[@]} -gt 0 ] || { echo "[FAIL] $PATCH_DIR 下没有 .patch" >&2; exit 1; }

apply_all() {           # $1 = 目录
  local dir=$1 ok=0 fail=0 p rc
  for p in "${patches[@]}"; do
    ( cd "$dir" || exit 1
      if git apply "$p" >/dev/null 2>&1; then exit 0; fi
      if patch -p1 --forward --silent < "$p" >/dev/null 2>&1; then exit 0; fi
      exit 1 )
    rc=$?
    if [ $rc -eq 0 ]; then
      echo "   [ok]   $(basename "$p")" >&2; ok=$((ok+1))
    else
      echo "   [FAIL] $(basename "$p")" >&2; fail=$((fail+1))
    fi
  done
  echo "$ok $fail"
}

if [ "$MODE" = "--check" ]; then
  cd "$TARGET" || exit 1
  [ -d .git ] || { echo "[FAIL] --check 需要 git 检出" >&2; exit 1; }
  base=$(git rev-parse --verify origin/main 2>/dev/null || git rev-parse HEAD)
  tmp=$(mktemp -d)
  echo "== 临时 worktree 按序试应用 ${#patches[@]} 个补丁（base=$(git rev-parse --short "$base")）"
  git worktree add --detach "$tmp" "$base" >/dev/null 2>&1 || { echo "[FAIL] worktree 创建失败"; exit 1; }
  out=$(apply_all "$tmp")
  ok=${out%% *}; fail=${out##* }
  echo "--- 应用后的 diffstat"
  ( cd "$tmp" && git add -A >/dev/null 2>&1; git diff --cached --stat | tail -4 )
  git worktree remove --force "$tmp" >/dev/null 2>&1 || rm -rf "$tmp"
  echo "== 结果: ok=$ok fail=$fail"
  [ "$fail" -eq 0 ] || exit 1
  exit 0
fi

cd "$TARGET" || exit 1
echo "== 目标: $TARGET"
[ -d .git ] && echo "   HEAD(应用前): $(git rev-parse --short HEAD 2>/dev/null || echo '-')"
out=$(apply_all "$TARGET")
echo "--- 工作区改动（补丁不产生提交）"
[ -d .git ] && git diff --stat | tail -3
echo "== 完成: $out"
