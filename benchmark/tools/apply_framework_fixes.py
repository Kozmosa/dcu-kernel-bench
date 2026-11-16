#!/usr/bin/env python3
"""Apply the versioned 1116 fixes to a separate framework checkout."""
import argparse
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SERIES = {
    'kernelweaver': ('c11ab7f', 'kernelweaver'),
    'pyramidkernel': ('8352750', 'pyramidkernel-1116'),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('framework', choices=SERIES)
    parser.add_argument('checkout', type=Path)
    parser.add_argument('--check', action='store_true', help='check the full series in a temporary worktree')
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
            finally:
                git(target, 'worktree', 'remove', '--force', str(worktree))
    else:
        apply(target, patches)


if __name__ == '__main__':
    main()
