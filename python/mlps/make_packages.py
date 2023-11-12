#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import re
import shutil
from pathlib import Path
from typing import Tuple

VERSION = '0.1'

SETUP_CFG = '''[metadata]
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

PYPROJECT_TOML = '''[build-system]
requires = [
    "setuptools>=42",
    "wheel"
]
build-backend = "setuptools.build_meta"
'''

README_MD = '''# The NAME library
This repository contains a Python implementation of multilayer perceptrons in FRAMEWORK.
'''

CIFAR10_SH = r'''#!/bin/sh

PYTHONPATH=../src

if [ ! -f ../data/cifar10.npz ]; then
    echo "Error: file ../data/cifar10.npz does not exist."
    echo "Please provide the correct location or run the prepare_datasets.sh script first."
    exit 1
fi

python3 ../tools/mlp.py \
        --layers="ReLU;ReLU;Linear" \
        --sizes="3072,1024,512,10" \
        --optimizers="Momentum(mu=0.9);Momentum(mu=0.9);Momentum(mu=0.9)" \
        --init-weights="Xavier,Xavier,Xavier" \
        --batch-size=100 \
        --epochs=10 \
        --loss=SoftmaxCrossEntropy \
        --learning-rate="Constant(0.01)" \
        --dataset=../data/cifar10.npz
'''

CIFAR10_BAT = r'''@echo off

set PYTHONPATH=..\src

if not exist ..\data\cifar10.npz (
    echo Error: file ..\data\cifar10.npz does not exist.
    echo Please provide the correct location or run the prepare_datasets.bat script first.
    exit /b 1
)

python ..\tools\mlp.py ^
    --layers="ReLU;ReLU;Linear" ^
    --sizes="784,1024,512,10" ^
    --optimizers="Momentum(mu=0.9);Momentum(mu=0.9);Momentum(mu=0.9)" ^
    --init-weights="Xavier,Xavier,Xavier" ^
    --batch-size=100 ^
    --epochs=10 ^
    --loss=SoftmaxCrossEntropy ^
    --learning-rate="Constant(0.01)" ^
    --dataset=..\data\cifar10.npz
'''

MNIST_SH = r'''#!/bin/sh

PYTHONPATH=../src

if [ ! -f ../data/mnist.npz ]; then
    echo "Error: file ../data/mnist.npz does not exist."
    echo "Please provide the correct location or run the prepare_datasets.sh script first."
    exit 1
fi

python3 ../tools/mlp.py \
        --layers="ReLU;ReLU;Linear" \
        --sizes="784,1024,512,10" \
        --optimizers="Momentum(mu=0.9);Momentum(mu=0.9);Momentum(mu=0.9)" \
        --init-weights="Xavier,Xavier,Xavier" \
        --batch-size=100 \
        --epochs=10 \
        --loss=SoftmaxCrossEntropy \
        --learning-rate="Constant(0.01)" \
        --dataset=../data/mnist.npz
'''

MNIST_BAT = r'''@echo off

set PYTHONPATH=..\src

if not exist ..\data\mnist.npz (
    echo Error: file ..\data\mnist.npz does not exist.
    echo Please provide the correct location or run the prepare_datasets.bat script first.
    exit /b 1
)

python ..\tools\mlp.py ^
    --layers="ReLU;ReLU;Linear" ^
    --sizes="784,1024,512,10" ^
    --optimizers="Momentum(mu=0.9);Momentum(mu=0.9);Momentum(mu=0.9)" ^
    --init-weights="Xavier,Xavier,Xavier" ^
    --batch-size=100 ^
    --epochs=10 ^
    --loss=SoftmaxCrossEntropy ^
    --learning-rate="Constant(0.01)" ^
    --dataset=..\data\mnist.npz
'''

PREPARE_DATASETS_SH = r'''#!/bin/sh

PYTHONPATH=../src

python3 ../tools/create_datasets.py cifar10 --root=../data
python3 ../tools/create_datasets.py mnist --root=../data
'''

PREPARE_DATASETS_BAT = r'''@echo off

set PYTHONPATH=..\src

python ..\tools\create_datasets.py cifar10 --root=..\data
python ..\tools\create_datasets.py mnist --root=..\data
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
    return Path('dist') / package_names[package] / 'src' / package


def is_empty_line(line: str):
    return not line or line.isspace()


def save_text(path: Path, text: str, mode=0o644):
    print(f'Saving {path}')
    path.write_text(text)
    path.chmod(mode)

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


def replace_pattern_in_file(path, pattern, replacement):
    text = path.read_text()
    text = re.sub(pattern, replacement, text)
    path.write_text(text)


def remove_lines_containing_word(path: Path, word: str):
    text = path.read_text()
    lines = text.split('\n')
    lines = [line for line in lines if not word in line]
    text = '\n'.join(lines)
    path.write_text(text)


def remove_line_continuations(path: Path):
    replace_pattern_in_file(path, r'\\\n', '')


def split_lines(text: str, is_section_start, is_section_end) -> Tuple[str, str]:
    """ Splits text into two parts. The first part consists of sections defined by `is_section_start`
    and `is_section_end`. The second part contains the remaining lines.
    :param text:
    :param is_section_start:
    :param is_section_end:
    :return:
    """
    lines = text.split('\n')
    inside_section = False  # True if we are inside a section

    section_lines = []
    other_lines = []

    for line in lines:
        if inside_section and is_section_end(line):
            inside_section = False

        if not inside_section and is_section_start(line):
            inside_section = True

        if inside_section:
            section_lines.append(line)
        else:
            other_lines.append(line)

    return '\n'.join(section_lines), '\n'.join(other_lines)


def split_imports(text: str) -> Tuple[str, str]:
    def is_import_start(line: str):
        return line.startswith('import ') or line.startswith('from ')

    def is_import_end(line: str):
        return is_empty_line(line)

    return split_lines(text, is_import_start, is_import_end)


def split_license(text):

    def is_license_start(line: str):
        return line.startswith('# Copyright')

    def is_license_end(line: str):
        return not line.startswith('#')

    return split_lines(text, is_license_start, is_license_end)


def split_code_file(path: Path):
    license, text = split_license(path.read_text())
    imports, code = split_imports(text)
    return license.strip(), imports.strip(), code.strip()


def has_max_indent(line: str, n: int) -> bool:
    return len(line) > 0 and not line[:n+1].isspace()

def remove_functions_from_file(path: Path, word: str):
    """Removes all function definitions from `text` that have the substring `word` in their name"""
    text = path.read_text().strip()

    def is_function_start(line):
        return re.search(rf'^def [a-zA-Z0-9_]*{word}[a-zA-Z0-9_]*\(', line) is not None

    # We assume that a function is ended by a line that does not start with a white space character
    def is_function_end(line):
        return has_max_indent(line, 0)

    _, text = split_lines(text, is_function_start, is_function_end)
    path.write_text(text)


def remove_methods_from_file(path: Path, word: str):
    """Removes all method definitions from `text` that have the substring `word` in their name.
       We assume that the indentation is 4 spaces.
    """
    text = path.read_text().strip()

    def is_function_start(line):
        return re.search(rf'^    def [a-zA-Z0-9_]*{word}[a-zA-Z0-9_]*\(', line) is not None

    def is_function_end(line):
        return has_max_indent(line, 4)

    _, text = split_lines(text, is_function_start, is_function_end)
    path.write_text(text)


def remove_classes_from_file(path: Path, word: str):
    """Removes all class definitions from `text` that have the substring `word` in their name.
       We assume that the indentation is 4 spaces.
    """
    text = path.read_text().strip()

    def is_function_start(line):
        return re.search(rf'^class [a-zA-Z0-9_]*{word}[a-zA-Z0-9_]*\(', line) is not None

    def is_function_end(line):
        return has_max_indent(line, 0)

    _, text = split_lines(text, is_function_start, is_function_end)
    path.write_text(text)


def extract_function_from_file(path: Path, function_name: str) -> str:
    text = path.read_text().strip()

    def is_function_start(line):
        return line.startswith(f'def {function_name}(')

    def is_function_end(line):
        return line and not re.match(r'^\s', line)

    text, _ = split_lines(text, is_function_start, is_function_end)
    return text


def make_package_folders():
    for package in package_requirements:
        create_folder(Path('dist') / package_names[package] / 'src' / package)
        create_folder(Path('dist') / package_names[package] / 'examples')
        create_folder(Path('dist') / package_names[package] / 'tools')
        if 'sympy' in package:
            create_folder(Path('dist') / package_names[package] / 'tests')


def join_files(package: str, path1: Path, path2: Path):
    if not path1.exists():
        return
    word = f'{package}.{path1.stem}'
    remove_line_continuations(path1)
    remove_line_continuations(path2)
    remove_lines_containing_word(path2, word)
    license1, imports1, code1 = split_code_file(path1)
    license2, imports2, code2 = split_code_file(path2)
    text = f'{license1}\n\n{imports1}\n{imports2}\n\n{code1}\n\n{code2}\n'
    save_text(path1, text)
    remove_file(path2)


def create_init_files():
    for package, requirements in package_requirements.items():
        path = package_folder(package) / '__init__.py'
        path.touch(mode=0o644, exist_ok=True)


def create_requirements_files():
    for package, requirements in package_requirements.items():
        dest = Path('dist') / package_names[package] / 'requirements.txt'
        text = '\n'.join(requirements)
        save_text(dest, text)


def create_readme_files():
    for package, requirements in package_requirements.items():
        dest = Path('dist') / package_names[package] / 'README.MD'
        text = README_MD.lstrip()
        text = text.replace('FRAMEWORK', package_frameworks[package])
        text = text.replace('NAME', package_names[package])
        text = text.replace('VERSION', VERSION)
        save_text(dest, text)


def create_setup_files():
    for package, requirements in package_requirements.items():
        dest = Path('dist') / package_names[package] / 'setup.cfg'
        text = SETUP_CFG.strip()
        text = text.replace('FRAMEWORK', package_frameworks[package])
        text = text.replace('NAME', package_names[package])
        text = text.replace('VERSION', VERSION)
        save_text(dest, text)

        dest = Path('dist') / package_names[package] / 'pyproject.toml'
        text = PYPROJECT_TOML.lstrip()
        save_text(dest, text)


def create_create_datasets_files():
    copy_file(Path('tools') / 'create_datasets_numpy.py', Path('dist') / package_names['nerva_numpy'] / 'tools' / 'create_datasets.py')
    copy_file(Path('tools') / 'create_datasets_numpy.py', Path('dist') / package_names['nerva_jax'] / 'tools' / 'create_datasets.py')
    copy_file(Path('tools') / 'create_datasets_torch.py', Path('dist') / package_names['nerva_torch'] / 'tools' / 'create_datasets.py')
    copy_file(Path('tools') / 'create_datasets_tensorflow.py', Path('dist') / package_names['nerva_tensorflow'] / 'tools' / 'create_datasets.py')


def copy_source_files():
    for package, requirements in package_requirements.items():
        destination_folder = package_folder(package)
        for source_file in Path(package).glob('*.py'):
            if source_file.stem.endswith('_colwise'):
                continue
            destination_file = destination_folder / source_file.name
            copy_file(source_file, destination_file)
        join_files(package, destination_folder / 'loss_functions.py', destination_folder / 'loss_functions_rowwise.py')
        join_files(package, destination_folder / 'softmax_functions.py', destination_folder / 'softmax_functions_rowwise.py')
        join_files(package, destination_folder / 'training.py', destination_folder / 'training_rowwise.py')


def copy_test_files():
    destination_folder = Path('dist') / package_names['nerva_sympy'] / 'tests'
    test_folder = Path('tests')
    for source_file in test_folder.glob('*.py'):
        destination_file = destination_folder / source_file.name
        copy_file(source_file, destination_file)


def create_mlp_files():
    source_folder = Path('.')
    for package in package_requirements:
        if 'sympy' in package:
            continue
        framework = package.replace('nerva_', '')
        examples_folder = Path('dist') / package_names[package] / 'examples'
        tools_folder = Path('dist') / package_names[package] / 'tools'
        mlp_utilities_file = source_folder / f'mlp_utilities.py'
        mlp_file = tools_folder / 'mlp.py'
        copy_file(source_folder / f'mlp_{framework}_rowwise.py', mlp_file)
        text = extract_function_from_file(mlp_utilities_file, 'make_argument_parser')
        replace_string_in_file(mlp_file, 'def main():', text + '\n\n\ndef main():')
        replace_string_in_file(mlp_file, '_rowwise', '')
        replace_string_in_file(mlp_file, 'from mlps.mlp_utilities import make_argument_parser', 'import argparse')

        save_text(examples_folder / 'cifar10.sh', CIFAR10_SH, mode=0o755)
        save_text(examples_folder / 'cifar10.bat', CIFAR10_BAT)
        save_text(examples_folder / 'mnist.sh', MNIST_SH, mode=0o755)
        save_text(examples_folder / 'mnist.bat', MNIST_BAT)
        save_text(examples_folder / 'prepare_datasets.sh', PREPARE_DATASETS_SH, mode=0o755)
        save_text(examples_folder / 'prepare_datasets.bat', PREPARE_DATASETS_BAT)


def fix_source_files():

    def fix_datasets_file(path):
        if not path.exists():
            return
        text = path.read_text()
        text = re.sub(r"if self\.rowwise:\n\s*(.*?)\n\s*else:\n.*?\n\n", r"\1\n", text, flags=re.DOTALL)  # simplify if statements
        text = text.replace('_rowwise', '')
        text = text.replace(', rowwise = True', '')
        text = text.replace(', rowwise', '')
        path.write_text(text)
        remove_lines_containing_word(path, 'rowwise')
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
            replace_string_in_file(path, '_colwise', '')
        rename_file(folder, 'multilayer_perceptron_rowwise.py', 'multilayer_perceptron.py')
        rename_file(folder, 'layers_rowwise.py', 'layers.py')


def fix_python_files():
    for path in Path('dist').rglob('*.py'):
        replace_string_in_file(path, 'mlps.', '')
        replace_pattern_in_file(path, r'\n\n\n(\n*)', r'\n\n\n')


def fix_test_files():
    tests_folder = Path('dist') / package_names['nerva_sympy'] / 'tests'
    for path in tests_folder.glob('test*.py'):
        remove_line_continuations(path)
        remove_functions_from_file(path, '_colwise')
        remove_methods_from_file(path, '_colwise')
        remove_classes_from_file(path, 'Colwise')
        replace_string_in_file(path, 'mlps.tests.', '')
        replace_string_in_file(path, 'tests.', '')
        replace_string_in_file(path, ', to_eigen', '')
        replace_string_in_file(path, ', x6', '')
        replace_string_in_file(path, '_rowwise', '')
        remove_lines_containing_word(path, 'eigen')


def main():
    make_package_folders()
    create_init_files()
    create_requirements_files()
    create_readme_files()
    create_setup_files()
    create_create_datasets_files()
    create_mlp_files()
    copy_source_files()
    copy_test_files()
    fix_source_files()
    fix_python_files()
    fix_test_files()


if __name__ == '__main__':
    main()
