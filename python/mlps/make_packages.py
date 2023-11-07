#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import re
import shutil
from pathlib import Path
from typing import Tuple

VERSION = '0.1'

SETUP_CFG = '''
[metadata]
name = NAME
version = VERSION
author = Wieger Wesselink
author_email = j.w.wesselink@tue.nl
description = An implementation of multilayer perceptrons in FRAMEWORK.
long_description = file: README.md
long_description_content_type = text/markdown
url = https://github.com/wiegerw/NAME
project_urls =
    Bug Tracker = https://github.com/wiegerw/NAME/issues
classifiers =
    Programming Language :: Python :: 3
    License :: OSI Approved :: Boost Software License 1.0 (BSL-1.0)
    Operating System :: OS Independent

[options]
package_dir =
  = src
packages = find:
python_requires = >=3.10

[options.packages.find]
where = src
'''

PYPROJECT_TOML = '''
[build-system]
requires = [
    "setuptools>=42",
    "wheel"
]
build-backend = "setuptools.build_meta"
'''

README_MD = '''
# The NAME library
This repository contains a Python implementation of multilayer perceptrons in FRAMEWORK.
'''

package_requirements = {
    'nerva_jax': ['numpy', 'jax'],
    'nerva_numpy': ['numpy'],
    'nerva_sympy': ['numpy', 'sympy'],
    'nerva_tensorflow': ['numpy', 'tensorflow'],
    'nerva_torch': ['numpy', 'torch'],
}

package_frameworks = {
    'nerva_jax': 'JAX',
    'nerva_numpy': 'NumPy',
    'nerva_sympy': 'SymPy',
    'nerva_tensorflow': 'Tensorflow',
    'nerva_torch': 'PyTorch',
}

package_names = {
    'nerva_jax': 'nerva-jax',
    'nerva_numpy': 'nerva-numpy',
    'nerva_sympy': 'nerva-sympy',
    'nerva_tensorflow': 'nerva-tensorflow',
    'nerva_torch': 'nerva-torch',
}

def package_folder(package: str) -> Path:
    return Path('../dist') / package_names[package] / 'src' / package


def save_text(path: Path, text: str):
    print(f'Saving {path}')
    path.write_text(text)


def copy_file(source: Path, target: Path):
    print(f"Copy '{source}' to '{target}'")
    shutil.copy(source, target)


def create_folder(folder: Path):
    print(f'Create folder {folder}')
    folder.mkdir(parents=True, exist_ok=True)


def remove_file(path: Path):
    print(f'Removing {path}')
    path.unlink()


def replace_string_in_file(path, source, target):
    text = path.read_text()
    text = text.replace(source, target)
    path.write_text(text)


def extract_function_from_file(path: Path, function_name: str) -> str:
    text = path.read_text().strip()
    lines = re.split(r'\n', text)

    in_function = False
    function_lines = []

    for line in lines:
        if not in_function and f'def {function_name}(' in line:
            in_function = True

        if in_function:
            function_lines.append(line)
            if function_lines[-1].isspace() and function_lines[-2].isspace():
                break

    return '\n'.join(function_lines).strip()


def remove_functions_from_file(path: Path, word: str):
    """Removes all function definitions from `text` that have `word` in their name"""
    text = path.read_text().strip()
    lines = re.split(r'\n', text)
    inside_function = False  # True if we are inside a function containing `word` in its name

    def is_start_of_function(line):
        return re.match(rf'^def \w*{word}\w*\(', line)

    # We assume that a function is ended by a line that does not start with a white space character
    def is_end_of_function(line):
        return line and not re.match(r'^\s', line)

    remaining_lines = []

    for line in lines:
        if not inside_function and is_start_of_function(line):
            inside_function = True
        elif inside_function and is_end_of_function(line):
            inside_function = False

        if not inside_function:
            remaining_lines.append(line)

    text = '\n'.join(remaining_lines).strip() + '\n'
    path.write_text(text)


def remove_lines_containing_word(text: str, word: str):
    pattern = r".*?\b{}\b.*?\n".format(re.escape(word))
    return re.sub(pattern, "", text)


def make_package_folders():
    for package in package_requirements:
        create_folder(Path('../dist') / package_names[package] / 'src' / package)
        create_folder(Path('../dist') / package_names[package] / 'tests')
        create_folder(Path('../dist') / package_names[package] / 'tools')


def split_license(text):
    lines = text.split('\n')
    license = []
    code = []

    def is_license(line: str):
        return line.strip().startswith("# ")

    inside_license = True

    for line in lines:
        if not is_license(line):
            inside_license = False
        if inside_license:
            license.append(line)
        else:
            code.append(line)

    return '\n'.join(license).strip(), '\n'.join(code).strip()


def split_imports(text: str) -> Tuple[str, str]:
    lines = text.split('\n')
    imports = []
    code = []

    def is_import(line: str):
        return line.startswith('import ') or line.startswith('from ') or line.isspace() or not line

    inside_imports = True
    for line in lines:
        if not is_import(line):
            inside_imports = False
        if inside_imports:
            imports.append(line)
        else:
            code.append(line)

    return '\n'.join(imports).strip(), '\n'.join(code).strip()


def join_files(path1: Path, path2: Path):
    if not path1.exists():
        return
    license1, text1 = split_license(path1.read_text())
    license2, text2 = split_license(path2.read_text())
    imports1, code1 = split_imports(text1)
    imports2, code2 = split_imports(text2)
    text = f'{license1}\n\n{imports1}\n{imports2}\n\n{code1}\n\n{code2}\n'
    save_text(path1, text)
    remove_file(path2)


def create_init_files():
    for package, requirements in package_requirements.items():
        path = package_folder(package) / '__init__.py'
        path.touch(mode=0o644, exist_ok=True)


def create_requirements_files():
    for package, requirements in package_requirements.items():
        dest = Path('../dist') / package_names[package] / 'requirements.txt'
        text = '\n'.join(requirements)
        save_text(dest, text)


def create_readme_files():
    for package, requirements in package_requirements.items():
        dest = Path('../dist') / package_names[package] / 'README.MD'
        text = README_MD.lstrip()
        text = text.replace('FRAMEWORK', package_frameworks[package])
        text = text.replace('NAME', package_names[package])
        text = text.replace('VERSION', VERSION)
        save_text(dest, text)


def create_setup_files():
    for package, requirements in package_requirements.items():
        dest = Path('../dist') / package_names[package] / 'setup.cfg'
        text = SETUP_CFG.strip()
        text = text.replace('FRAMEWORK', package_frameworks[package])
        text = text.replace('NAME', package_names[package])
        text = text.replace('VERSION', VERSION)
        save_text(dest, text)

        dest = Path('../dist') / package_names[package] / 'pyproject.toml'
        text = PYPROJECT_TOML.lstrip()
        save_text(dest, text)


def copy_source_files():
    for package, requirements in package_requirements.items():
        destination_folder = package_folder(package)
        for source_file in Path(package).glob('*.py'):
            if source_file.stem.endswith('_colwise'):
                continue
            destination_file = destination_folder / source_file.name
            copy_file(source_file, destination_file)
        join_files(destination_folder / 'loss_functions.py', destination_folder / 'loss_functions_rowwise.py')
        join_files(destination_folder / 'parse_mlp.py', destination_folder / 'parse_mlp_rowwise.py')
        join_files(destination_folder / 'softmax_functions.py', destination_folder / 'softmax_functions_rowwise.py')
        join_files(destination_folder / 'training.py', destination_folder / 'training_rowwise.py')


def copy_test_files():
    destination_folder = Path('../dist') / package_names['nerva_sympy'] / 'tests'
    test_folder = Path('tests')
    for source_file in test_folder.glob('*.py'):
        if not 'sympy' in source_file.name:
            continue
        destination_file = destination_folder / source_file.name
        print(f"Copy '{source_file}' to '{destination_file}'")
        shutil.copy(source_file, destination_file)


def copy_mlp_files():
    source_folder = Path('..')
    for package in package_requirements:
        if 'sympy' in package:
            continue
        framework = package.replace('nerva_', '')
        target_folder = Path('../dist') / package_names[package] / 'tools'
        mlp_utilities_file = source_folder / f'mlp_utilities.py'
        mlp_file = target_folder / 'mlp.py'
        copy_file(source_folder / f'mlp_{framework}_rowwise.py', mlp_file)
        text = extract_function_from_file(mlp_utilities_file, 'make_argument_parser')
        replace_string_in_file(mlp_file, 'def main():', text + '\n\n\ndef main():')
        replace_string_in_file(mlp_file, '_rowwise', '')
        replace_string_in_file(mlp_file, 'from mlp_utilities import make_argument_parser', 'import argparse')


def fix_source_files():

    def fix_datasets_file(path):
        if not path.exists():
            return
        text = path.read_text()
        text = re.sub(r"if self\.rowwise:\n\s*(.*?)\n\s*else:\n.*?\n\n", r"\1\n", text, flags=re.DOTALL)  # simplify if statements
        text = text.replace('_rowwise', '')
        text = text.replace(', rowwise = True', '')
        text = text.replace(', rowwise', '')
        text = remove_lines_containing_word(text, 'rowwise')
        path.write_text(text)
        remove_functions_from_file(path, 'to_one_hot_colwise')

    def rename_file(folder, src, target):
        src = folder / src
        target = folder / target
        src.rename(target)

    for package in package_requirements:
        folder = package_folder(package)
        fix_datasets_file(folder / 'datasets.py')
        remove_functions_from_file(folder / 'loss_functions.py', '_colwise')
        remove_functions_from_file(folder / 'softmax_functions.py', '_colwise')
        for path in folder.glob('*.py'):
            replace_string_in_file(path, '_rowwise', '')
        rename_file(folder, 'multilayer_perceptron_rowwise.py', 'multilayer_perceptron.py')
        rename_file(folder, 'layers_rowwise.py', 'layers.py')


def main():
    make_package_folders()
    create_init_files()
    create_requirements_files()
    create_readme_files()
    create_setup_files()
    copy_mlp_files()
    copy_source_files()
    copy_test_files()
    fix_source_files()


if __name__ == '__main__':
    main()
