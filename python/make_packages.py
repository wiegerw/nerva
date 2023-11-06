#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import re
import shutil
from pathlib import Path
from typing import Tuple

package_requirements = {
    'nerva_jax': ['numpy', 'jax'],
    'nerva_numpy': ['numpy'],
    'nerva_tensorflow': ['numpy', 'tensorflow'],
    'nerva_torch': ['numpy', 'torch'],
}

def package_folder(package: str) -> Path:
    return Path('dist') / package / 'src' / package


def save_text(path: Path, text: str):
    print(f'Saving {path}')
    path.write_text(text)


def remove_file(path: Path):
    print(f'Removing {path}')
    path.unlink()


def make_package_folders():
    for package in package_requirements:
        folder = package_folder(package)
        print(f'Create folder {folder}')
        folder.mkdir(parents=True, exist_ok=True)


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
    license1, text1 = split_license(path1.read_text())
    license2, text2 = split_license(path2.read_text())
    imports1, code1 = split_imports(text1)
    imports2, code2 = split_imports(text2)
    text = f'{license1}\n\n{imports1}\n{imports2}\n\n{code1}\n\n{code2}\n'
    save_text(path1, text)
    remove_file(path2)


def create_requirements():
    for package, requirements in package_requirements.items():
        dest = Path('dist') / package / 'requirements.txt'
        text = '\n'.join(requirements)
        save_text(dest, text)


def copy_files():
    for package, requirements in package_requirements.items():
        destination_folder = package_folder(package)
        for source_file in Path(package).glob('*.py'):
            if source_file.stem.endswith('_colwise'):
                continue
            destination_file = destination_folder / source_file.name
            print(f"Copy '{source_file}' to '{destination_file}'")
            shutil.copy(source_file, destination_file)
        join_files(destination_folder / 'loss_functions.py', destination_folder / 'loss_functions_rowwise.py')
        join_files(destination_folder / 'parse_mlp.py', destination_folder / 'parse_mlp_rowwise.py')
        join_files(destination_folder / 'softmax_functions.py', destination_folder / 'softmax_functions_rowwise.py')
        join_files(destination_folder / 'training.py', destination_folder / 'training_rowwise.py')


def remove_function_definition(text: str, function_name: str):
    pattern = r"def {}\(.*?\):[\s\S]*?(?:(?=\bdef\b)|$)".format(re.escape(function_name))
    return re.sub(pattern, "", text)


def remove_matching_function_definitions(text, word):
    """Removes all function definitions that have `word` in their name"""
    pattern = rf"def .*{re.escape(word)}.*\(.*\)(?: -> \w+)?:[\s\S]*?(?:(?=\bdef\b)|$)"
    return re.sub(pattern, "", text)


def remove_lines_containing_word(text: str, word: str):
    pattern = r".*?\b{}\b.*?\n".format(re.escape(word))
    return re.sub(pattern, "", text)


def fix_files():

    def fix_datasets_file(path):
        text = path.read_text()
        text = re.sub(r"if self\.rowwise:\n\s*(.*?)\n\s*else:\n.*?\n\n", r"\1\n", text, flags=re.DOTALL)  # simplify if statements
        text = remove_function_definition(text, 'to_one_hot_colwise')
        text = text.replace('_rowwise', '')
        text = text.replace(', rowwise = True', '')
        text = text.replace(', rowwise', '')
        text = remove_lines_containing_word(text, 'rowwise')
        path.write_text(text)

    def remove_colwise_functions(path):
        text = path.read_text()
        text = remove_matching_function_definitions(text, '_colwise')
        path.write_text(text)

    def remove_rowwise(path):
        text = path.read_text()
        text = remove_matching_function_definitions(text, '_colwise')
        path.write_text(text)

    def rename_file(folder, src, target):
        src = folder / src
        target = folder / target
        src.rename(target)

    for package in package_requirements:
        folder = package_folder(package)
        fix_datasets_file(folder / 'datasets.py')
        remove_colwise_functions(folder / 'loss_functions.py')
        remove_colwise_functions(folder / 'softmax_functions.py')
        for path in folder.glob('*.py'):
            remove_rowwise(path)
        rename_file(folder, 'multilayer_perceptron_rowwise.py', 'multilayer_perceptron.py')
        rename_file(folder, 'layers_rowwise.py', 'layers.py')


def main():
    make_package_folders()
    create_requirements()
    copy_files()
    fix_files()


if __name__ == '__main__':
    main()
