#!/usr/bin/env python3

# Copyright 2024 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import Tuple
import numpy as np

import nerva_torch.loss_functions
from nerva_torch.loss_functions_torch import *


def random_float_matrix(shape, a, b):
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
    rand_array = np.random.rand(*shape)

    # Scale and shift the array to the range [a, b]
    scaled_array = a + (b - a) * rand_array

    return scaled_array


def make_target(Y: np.ndarray) -> np.ndarray:
    """
    Creates a boolean matrix T with the same shape as Y,
    where each row of T has exactly one value set to 1.

    Parameters:
    Y (np.ndarray): The input numpy array.

    Returns:
    np.ndarray: A boolean matrix with the same shape as Y,
                with exactly one True value per row.
    """
    if Y.ndim != 2:
        raise ValueError("The input array must be two-dimensional")

    # Get the shape of the input array
    rows, cols = Y.shape

    # Initialize an array of zeros with the same shape as Y
    T = np.zeros((rows, cols), dtype=bool)

    # Set one random element in each row to True
    for i in range(rows):
        random_index = np.random.randint(0, cols)
        T[i, random_index] = True

    return T


def print_cpp_matrix_declaration(name: str, x: np.ndarray):
    """
    Prints a two-dimensional numpy array in the specified format.

    Parameters:
    name (str): The name to be given to the printed matrix.
    x (np.ndarray): The two-dimensional numpy array to print.

    Raises:
    ValueError: If the input array is not two-dimensional.
    """
    if x.ndim != 2:
        raise ValueError("The input array must be two-dimensional")

    print(f"  eigen::matrix {name} {{")
    rows = x.shape[0]
    for i in range(rows):
        # Check the data type and format accordingly
        if x.dtype == bool:
            row_elements = ', '.join(map(lambda el: '1' if el else '0', x[i]))
        else:
            row_elements = ', '.join(map(str, x[i]))

        if i == rows - 1:
            print(f"    {{{row_elements}}}")
        else:
            print(f"    {{{row_elements}}},")
    print("  };\n")


def make_loss_test(shape: Tuple[int, int], a: float, b: float, rowwise=True, index=0):
    print(f'TEST_CASE("test_loss{index}")')
    print('{')

    Y = random_float_matrix(shape, a, b)
    print_cpp_matrix_declaration('Y', Y if rowwise else Y.T)

    T = make_target(Y)
    print_cpp_matrix_declaration('T', T if rowwise else T.T)

    T = T.astype(float)

    Y = torch.Tensor(Y)
    T = torch.Tensor(T)

    loss = squared_error_loss_torch(Y, T)
    name = 'squared_error_loss'
    print(f'  test_loss("{name}", {name}(), {loss}, Y, T);')

    loss = softmax_cross_entropy_loss_torch(Y, T)
    name = 'softmax_cross_entropy_loss'
    print(f'  test_loss("{name}", {name}(), {loss}, Y, T);')

    loss = negative_log_likelihood_loss_torch(Y, T)
    name = 'negative_log_likelihood_loss'
    print(f'  test_loss("{name}", {name}(), {loss}, Y, T);')

    # No PyTorch equivalent available
    loss = nerva_torch.loss_functions.Cross_entropy_loss(Y, T)
    name = 'cross_entropy_loss'
    print(f'  test_loss("{name}", {name}(), {loss}, Y, T);')

    # No PyTorch equivalent available(?)
    loss = nerva_torch.loss_functions.Logistic_cross_entropy_loss(Y, T)
    name = 'logistic_cross_entropy_loss'
    print(f'  test_loss("{name}", {name}(), {loss}, Y, T);')

    print('}\n')


if __name__ == '__main__':
    for i in range(1, 6):
        shape = (3, 4)
        make_loss_test(shape, 0.000001, 10.0, index=i)
