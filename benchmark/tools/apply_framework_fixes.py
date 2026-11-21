#!/usr/bin/env python3
"""Apply the versioned framework fixes to a separate framework checkout.

Three independent series (pick only the ones you need):

  kernelweaver           DCU 公开评测契约 + oracle root 不参与选优（代码）
  pyramidkernel          同上，PyramidKernel 版（代码）
  kernelweaver-dcu-wiki  DCU 公开 Wiki 语料 knowledge/kernelwiki_dcu/（内容资产）
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SERIES = {
    'kernelweaver': ('c11ab7f', 'kernelweaver'),
    'pyramidkernel': ('8352750', 'pyramidkernel-1116'),
    # Wiki 语料依赖 cd5d85c 引入的 src/knowledge/public_wiki.py 与 .gitattributes，
    # 而 kernelweaver 代码序列是按 c11ab7f 生成的（两者在 cli/workflow 上会撞），
    # 所以这里如实记录不同的基座。
    'kernelweaver-dcu-wiki': ('cd5d85c', 'kernelweaver-dcu-wiki'),
}


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


def verify_wiki_corpus(target):
    """After the wiki series, the corpus must hash-identical to its frozen lock.

    The lock stores per-file byte hashes, so a checkouts' CRLF rewriting (or a
    truncated patch) shows up here instead of as a confusing loader error later.
    """
    import hashlib
    import json
    corpus = target / 'knowledge' / 'kernelwiki_dcu'
    lock = json.loads((corpus / 'corpus.lock.json').read_text(encoding='utf-8'))
    files = [corpus / 'index.yaml']
    for sub in ('data', 'sources', 'wiki'):
        files.extend(p for p in (corpus / sub).rglob('*') if p.is_file())
    hashes = {p.relative_to(corpus).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    if hashes != lock.get('files') or digest != lock.get('sha256'):
        raise SystemExit(
            'DCU wiki corpus does not match corpus.lock.json '
            f'(files identical: {hashes == lock.get("files")}, digest match: {digest == lock.get("sha256")}). '
            'A CRLF-rewriting checkout or a truncated patch is the usual cause; check '
            '`git config core.autocrlf` and the `knowledge/kernelwiki_dcu/** -text` attribute.')
    print(f'wiki corpus verified: {len(hashes)} files, sha256 {digest[:16]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('framework', choices=SERIES)
    parser.add_argument('checkout', type=Path)
    parser.add_argument('--check', action='store_true', help='check the full series in a temporary worktree')
    parser.add_argument('--skip-verify', action='store_true', help='skip the post-apply corpus lock check')
    args = parser.parse_args()
    base, folder = SERIES[args.framework]
    target = args.checkout.resolve()
    git(target, 'merge-base', '--is-ancestor', base, 'HEAD')
    patches = sorted((ROOT / 'patches' / folder).glob('*.patch'))
    if not patches:
        parser.error('empty patch series')
    if args.check:
        with tempfile.TemporaryDirectory(prefix='dcukb_patch_check_') as tmp:
            worktree = Path(tmp) / 'framework'
            git(target, 'worktree', 'add', '--detach', str(worktree), 'HEAD', stdout=subprocess.DEVNULL)
            try:
                apply(worktree, patches)
                if folder == 'kernelweaver-dcu-wiki' and not args.skip_verify:
                    verify_wiki_corpus(worktree)
            finally:
                git(target, 'worktree', 'remove', '--force', str(worktree))
    else:
        apply(target, patches)
        if folder == 'kernelweaver-dcu-wiki' and not args.skip_verify:
            verify_wiki_corpus(target)


if __name__ == '__main__':
    main()
