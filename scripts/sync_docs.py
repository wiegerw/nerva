#!/usr/bin/env python3

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""Copy the shared documentation in doc/shared to the Nerva repositories.

The repositories are self-contained, so each of them contains a copy of the shared documentation files.
This script keeps the copies identical to the files in doc/shared. Differences between the repositories,
such as the name of the library or the location of a script, are expressed with AsciiDoc attributes that
each repository defines in its doc/asciidoc/settings.adoc. See doc/shared/README.md.

Usage:
    python scripts/sync_docs.py            # copy the files to the sibling checkouts ../nerva-<name>
    python scripts/sync_docs.py --check    # report differences, exit with status 1 if there are any
"""

import argparse
import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHARED_DIR = ROOT / 'doc' / 'shared'

# (group of shared files, target repositories, target directory inside a repository)
TARGETS = [
    ('python', ['nerva-jax', 'nerva-numpy', 'nerva-tensorflow', 'nerva-torch'], Path('doc/asciidoc/include')),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true', help='Only report differences; exit with status 1 if there are any')
    parser.add_argument('--root', type=Path, default=ROOT.parent,
                        help='Directory that contains the repository checkouts (default: the parent directory of nerva)')
    args = parser.parse_args()

    differences = 0
    count = 0
    for group, repositories, target_dir in TARGETS:
        files = sorted((SHARED_DIR / group).glob('*.adoc'))
        if not files:
            print(f'Error: no shared files found in {SHARED_DIR / group}')
            return 1
        for repository in repositories:
            directory = args.root / repository / target_dir
            if not directory.is_dir():
                print(f'Error: {directory} not found')
                return 1
            for file in files:
                count += 1
                expected = file.read_text(encoding='utf-8')
                target = directory / file.name
                actual = target.read_text(encoding='utf-8') if target.exists() else ''
                if actual == expected:
                    continue
                differences += 1
                if args.check:
                    sys.stdout.writelines(difflib.unified_diff(actual.splitlines(keepends=True), expected.splitlines(keepends=True),
                                                               fromfile=str(target), tofile=str(file.relative_to(ROOT))))
                else:
                    target.write_text(expected, encoding='utf-8')
                    print(f'Updated {target}')

    if args.check:
        if differences:
            print(f'{differences} file(s) differ from doc/shared. Run scripts/sync_docs.py to update them.')
            return 1
        print(f'All {count} shared documentation files are in sync.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
