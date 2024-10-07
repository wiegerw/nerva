# Copyright 2024 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from io import StringIO
from pathlib import Path
import numpy as np
import torch


def random_float_matrix(rows: int, cols: int, a: float, b: float):
    """
    Generates a random numpy array with the given shape and float values in the range [a, b].

    Parameters:
    shape (tuple): The shape of the numpy array to generate.
    a (float): The minimum value in the range.
    b (float): The maximum value in the range.

    Returns:
    np.ndarray: A numpy array of the specified shape with random float values in the range [a, b].
    """
    # Generate a random array with values in the range [0, 1)
    rand_array = np.random.rand(rows, cols)

    # Scale and shift the array to the range [a, b]
    scaled_array = a + (b - a) * rand_array

    return scaled_array


def make_target_rowwise(rows, cols, cover_all_classes=False) -> np.ndarray:
    """
    Creates a boolean matrix T where each row of T has exactly one value set to 1.
    If cover_classes is True, ensures that every column in the result has at least one value set to 1.
    """

    T = np.zeros((rows, cols), dtype=bool)

    # Set one random element in each row to True
    for i in range(rows):
        j = np.random.randint(0, cols)
        T[i, j] = True

    if cover_all_classes:
        for j in range(cols):
            # Check if column j has at least one True value
            if not T[:, j].any():
                # If not, set a random row in column j to True
                i = np.random.randint(0, rows)
                # To maintain the constraint that each row has exactly one True value,
                # find the current True value in row i and set it to False
                current_true_index = np.where(T[i])[0][0]
                T[i, current_true_index] = False
                T[i, j] = True

    return T


def make_target_colwise(rows, cols, cover_all_classes=False) -> np.ndarray:
    """
    Creates a boolean matrix T where each column of T has exactly one value set to 1.
    """

    return make_target_rowwise(cols, rows, cover_all_classes).T


#   eigen::matrix Y {
#     {0.36742274, 0.35949028, 0.27308698},
#     {0.30354068, 0.41444678, 0.28201254},
#     {0.34972793, 0.32481684, 0.32545523},
#     {0.34815459, 0.44543710, 0.20640831},
#     {0.19429503, 0.32073754, 0.48496742},
#   };
def print_cpp_matrix_declaration(out: StringIO, name: str, x: np.ndarray, num_decimal_places: int=8, indent: int=2) -> None:
    """
    Prints a NumPy array in Eigen matrix format.

    Args:
        out: Where the output is written.
        name: The name of the matrix (e.g., "A").
        x: The NumPy array to be printed.
        num_decimal_places: The precision.
        rowwise: If True vectors are printed as rows, otherwise as columns.
    """
    indent = ' ' * indent
    out.write(f"{indent}eigen::matrix {name} {{\n")
    for row in x:
        out.write(f"{indent}  {{")
        out.write(f"{row[0]:.{num_decimal_places}f}")
        for element in row[1:]:
            out.write(f", {element:.{num_decimal_places}f}")
        out.write("},\n")
    out.write(f"{indent}}};\n\n")


#         Y = torch.tensor([
#             [0.36742274, 0.35949028, 0.27308698],
#             [0.30354068, 0.41444678, 0.28201254],
#             [0.34972793, 0.32481684, 0.32545523],
#             [0.34815459, 0.44543710, 0.20640831],
#             [0.19429503, 0.32073754, 0.48496742],
#         ], dtype = torch.float32)
def print_torch_matrix_declaration(out: StringIO, name: str, x: torch.tensor, num_decimal_places: int=8, indent: int=4) -> None:
    """
    Prints a NumPy array in PyTorch matrix format.

    Args:
        out: Where the output is written.
        name: The name of the matrix (e.g., "A").
        x: The NumPy array to be printed.
        num_decimal_places: The precision.
    """
    indent = ' ' * indent
    out.write(f"{indent}{name} = torch.tensor([\n")
    for row in x:
        out.write(f"{indent}    [")
        out.write(f"{row[0]:.{num_decimal_places}f}")
        for element in row[1:]:
            out.write(f", {element:.{num_decimal_places}f}")
        out.write("],\n")
    out.write(f"{indent}], dtype = torch.float32)\n\n")


def insert_text_in_file(filename: str, text: str, begin_label='//--- begin generated code ---//', end_label='//--- end generated code ---//'):
    lines = Path(filename).read_text().split('\n')
    first = lines.index(begin_label)
    last = lines.index(end_label) + 1
    text1 = '\n'.join(lines[:first])
    text2 = '\n'.join(lines[last:])
    Path(filename).write_text(f'{text1}\n{begin_label}\n{text}\n{end_label}\n{text2}')
    pass
