#!/usr/bin/env python3
"""Apply the versioned framework fixes to a separate framework checkout.

Three independent series (pick only the ones you need):

  kernelweaver           DCU 公开评测契约 + oracle root 不参与选优（代码补丁）
  pyramidkernel          同上，PyramidKernel 版（代码补丁）
  kernelweaver-dcu-wiki  DCU 公开 Wiki 语料 + 装载装配。补丁只负责改既有的框架文件
                         （.gitattributes、src/knowledge/*、scripts/audit_public_wiki.py）；
                         语料与消融配置是**直接可见的文件**，放在
                         patches/kernelweaver-dcu-wiki/overlay/ 下，按目标仓库根目录
                         逐字节拷入，因此不受 CRLF 转换影响。

    python benchmark/tools/apply_framework_fixes.py kernelweaver-dcu-wiki /path/to/KernelWeaver --check
    python benchmark/tools/apply_framework_fixes.py kernelweaver-dcu-wiki /path/to/KernelWeaver
    # 语料改过之后，把检出的文件反向同步回本仓库（刷新 overlay）
    python benchmark/tools/apply_framework_fixes.py kernelweaver-dcu-wiki /path/to/KernelWeaver --pull-overlay
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SERIES = {
    'kernelweaver': ('c11ab7f', 'kernelweaver'),
    'pyramidkernel': ('8352750', 'pyramidkernel-1116'),
    # Wiki 语料的装载装配依赖 cd5d85c 引入的 src/knowledge/public_wiki.py 与 .gitattributes，
    # 而 kernelweaver 代码序列是按 c11ab7f 生成的（两者在 cli/workflow 上会撞），
    # 所以这里如实记录不同的基座。
    'kernelweaver-dcu-wiki': ('cd5d85c', 'kernelweaver-dcu-wiki'),
}
OVERLAY_DIR = 'overlay'
CORPUS = Path('knowledge') / 'kernelwiki_dcu'


def git(target, *args, **kwargs):
    return subprocess.run(['git', '-C', str(target), *args], check=True, **kwargs)


def apply(target, patches):
    for patch in patches:
        # Already-applied patches are explicit and do not turn partial failures
        # into success. No fuzzy patch fallback and no forced reset.
        done = subprocess.run(['git', '-C', str(target), 'apply', '--reverse', '--check', str(patch)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if done.returncode == 0:
            print('already applied:', patch.name)
            continue
        git(target, 'apply', '--check', str(patch))
        git(target, 'apply', str(patch))
        print('applied:', patch.name)


def overlay_files(series_dir):
    overlay = series_dir / OVERLAY_DIR
    if not overlay.is_dir():
        return []
    return sorted(p for p in overlay.rglob('*') if p.is_file())


def install_overlay(target, series_dir):
    """Byte-exact copy of the visible files into the checkout (no git filters)."""
    files = overlay_files(series_dir)
    for src in files:
        dst = target / src.relative_to(series_dir / OVERLAY_DIR)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
    print(f'overlay installed: {len(files)} files (byte-exact copy)')
    return files


def corpus_digest(corpus):
    files = [corpus / 'index.yaml']
    for sub in ('data', 'sources', 'wiki'):
        files.extend(p for p in (corpus / sub).rglob('*') if p.is_file())
    hashes = {p.relative_to(corpus).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return hashes, hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def verify_wiki_corpus(target):
    """After installation the corpus must hash-identical to its frozen lock."""
    corpus = target / CORPUS
    lock = json.loads((corpus / 'corpus.lock.json').read_text(encoding='utf-8'))
    hashes, digest = corpus_digest(corpus)
    if hashes != lock.get('files') or digest != lock.get('sha256'):
        raise SystemExit(
            'DCU wiki corpus does not match corpus.lock.json '
            f'(files identical: {hashes == lock.get("files")}, digest match: {digest == lock.get("sha256")}). '
            'A CRLF-rewriting checkout, a truncated copy or an edited page without re-freezing is the usual cause; '
            'check `git config core.autocrlf`, the `knowledge/kernelwiki_dcu/** -text` attribute, and '
            'scripts/audit_public_wiki.py --wiki knowledge/kernelwiki_dcu --freeze.')
    print(f'wiki corpus verified: {len(hashes)} files, sha256 {digest[:16]}')
    return digest


def pull_overlay(target, series_dir):
    """Refresh the repo's visible copy from a checkout (single source of truth = this repo)."""
    hashes, digest = corpus_digest(target / CORPUS)
    lock = json.loads((target / CORPUS / 'corpus.lock.json').read_text(encoding='utf-8'))
    if hashes != lock.get('files') or digest != lock.get('sha256'):
        raise SystemExit('refusing to pull: the checkout corpus does not match its own lock; re-freeze it first')
    overlay = series_dir / OVERLAY_DIR
    shutil.rmtree(overlay, ignore_errors=True)
    shutil.copytree(target / CORPUS, overlay / CORPUS)
    for cfg in sorted((target / 'configs' / 'search').glob('dcu_*.yaml')):
        dst = overlay / 'configs' / 'search' / cfg.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(cfg.read_bytes())
    print(f'overlay refreshed from {target}: {len(overlay_files(series_dir))} files, corpus sha256 {digest[:16]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('framework', choices=SERIES)
    parser.add_argument('checkout', type=Path)
    parser.add_argument('--check', action='store_true', help='check the full series in a temporary worktree')
    parser.add_argument('--skip-verify', action='store_true', help='skip the post-apply corpus lock check')
    parser.add_argument('--pull-overlay', action='store_true',
                        help='refresh patches/<series>/overlay/ from the checkout instead of installing')
    parser.add_argument('--skip-base-check', action='store_true',
                        help='跳过基座祖先检查（用于交付的快照 tar 包：没有上游 git 历史）')
    args = parser.parse_args()
    base, folder = SERIES[args.framework]
    target = args.checkout.resolve()
    series_dir = ROOT / 'patches' / folder
    if not args.skip_base_check:
        git(target, 'merge-base', '--is-ancestor', base, 'HEAD')
    else:
        print(f'skip base check (expect {base})')

    if args.pull_overlay:
        pull_overlay(target, series_dir)
        return

    patches = sorted(series_dir.glob('*.patch'))
    if not patches and not overlay_files(series_dir):
        parser.error('empty patch series')
    if args.check:
        with tempfile.TemporaryDirectory(prefix='dcukb_patch_check_') as tmp:
            worktree = Path(tmp) / 'framework'
            git(target, 'worktree', 'add', '--detach', str(worktree), 'HEAD', stdout=subprocess.DEVNULL)
            try:
                apply(worktree, patches)
                install_overlay(worktree, series_dir)
                if folder == 'kernelweaver-dcu-wiki' and not args.skip_verify:
                    verify_wiki_corpus(worktree)
            finally:
                git(target, 'worktree', 'remove', '--force', str(worktree))
    else:
        apply(target, patches)
        install_overlay(target, series_dir)
        if folder == 'kernelweaver-dcu-wiki' and not args.skip_verify:
            verify_wiki_corpus(target)


if __name__ == '__main__':
    main()
