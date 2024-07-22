# Copyright 2024 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import numpy as np


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


def make_target_rowwise(rows, cols) -> np.ndarray:
    """
    Creates a boolean matrix T where each row of T has exactly one value set to 1.
    """

    T = np.zeros((rows, cols), dtype=bool)

    # Set one random element in each row to True
    for i in range(rows):
        j = np.random.randint(0, cols)
        T[i, j] = True

    return T


def make_target_colwise(rows, cols) -> np.ndarray:
    """
    Creates a boolean matrix T where each column of T has exactly one value set to 1.
    """

    return make_target_rowwise(cols, rows).T


def print_cpp_matrix_declaration(name: str, x: np.ndarray, num_decimal_places: int=8, rowwise=True):
    """
    Prints a NumPy array in an Eigen matrix format.

    Args:
        name: The name of the matrix (e.g., "A").
        x: The NumPy array to be printed.
        num_decimal_places: The precision.
        rowwise: If True vectors are printed as rows, otherwise as columns.
    """
    if x.ndim == 1:
        x = x.reshape(1, -1) if rowwise else x.reshape(-1, 1)

    print(f"  eigen::matrix {name} {{")
    for row in x:
        print(f"    {{", end="")
        print(f"{row[0]:.{num_decimal_places}f}", end="")
        for element in row[1:]:
            print(f", {element:.{num_decimal_places}f}", end="")
        print("},")
    print("  };\n")
