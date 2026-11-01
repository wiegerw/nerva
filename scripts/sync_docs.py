#!/usr/bin/env python3

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""Copy the shared documentation in doc/shared to the Nerva repositories.

The repositories are self-contained, so each of them contains a copy of the shared documentation files.
This script keeps the copies identical to the files in doc/shared:

- The AsciiDoc files are copied verbatim. Differences between the repositories, such as the name of the
  library or the location of a script, are expressed with AsciiDoc attributes that each repository defines
  in its doc/asciidoc/settings.adoc.
- The README.md files of the Python backends are rendered from doc/shared/readme/README.md, by replacing
  the tokens nerva-BACKEND, nerva_BACKEND, {{FRAMEWORK}} and {{CPU_GPU}}.

See doc/shared/README.md.

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

# The README.md files of the Python backends: backend -> name of the framework
README_TEMPLATE = SHARED_DIR / 'readme' / 'README.md'
README_BACKENDS = {'jax': 'JAX', 'numpy': 'NumPy', 'tensorflow': 'TensorFlow', 'torch': 'PyTorch'}


def render_readme(backend: str) -> str:
    """Render the README.md of a Python backend from the template."""
    cpu_gpu = (SHARED_DIR / 'readme' / 'cpu-gpu' / f'{backend}.md').read_text(encoding='utf-8').rstrip('\n')
    text = README_TEMPLATE.read_text(encoding='utf-8')
    text = text.replace('{{FRAMEWORK}}', README_BACKENDS[backend]).replace('{{CPU_GPU}}', cpu_gpu)
    return text.replace('nerva-BACKEND', f'nerva-{backend}').replace('nerva_BACKEND', f'nerva_{backend}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true', help='Only report differences; exit with status 1 if there are any')
    parser.add_argument('--root', type=Path, default=ROOT.parent,
                        help='Directory that contains the repository checkouts (default: the parent directory of nerva)')
    args = parser.parse_args()

    # (expected contents, target file, source) for all files
    files = []
    for group, repositories, target_dir in TARGETS:
        sources = sorted((SHARED_DIR / group).glob('*.adoc'))
        if not sources:
            print(f'Error: no shared files found in {SHARED_DIR / group}')
            return 1
        for repository in repositories:
            for source in sources:
                files.append((source.read_text(encoding='utf-8'), args.root / repository / target_dir / source.name, source))
    for backend in README_BACKENDS:
        files.append((render_readme(backend), args.root / f'nerva-{backend}' / 'README.md', README_TEMPLATE))

    differences = 0
    for expected, target, source in files:
        if not target.parent.is_dir():
            print(f'Error: {target.parent} not found')
            return 1
        actual = target.read_text(encoding='utf-8') if target.exists() else ''
        if actual == expected:
            continue
        differences += 1
        if args.check:
            sys.stdout.writelines(difflib.unified_diff(actual.splitlines(keepends=True), expected.splitlines(keepends=True),
                                                       fromfile=str(target), tofile=str(source.relative_to(ROOT))))
        else:
            target.write_text(expected, encoding='utf-8')
            print(f'Updated {target}')

    if args.check:
        if differences:
            print(f'{differences} file(s) differ from doc/shared. Run scripts/sync_docs.py to update them.')
            return 1
        print(f'All {len(files)} shared documentation files are in sync.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
