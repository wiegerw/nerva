#!/usr/bin/env python3

# Copyright 2024 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse

import numpy as np

import nerva_torch.loss_functions
from nerva_torch.loss_functions_torch import *
from test_utilities import random_float_matrix, print_cpp_matrix_declaration, make_target_rowwise


def make_loss_test(N: int, K: int, a: float, b: float, index=0, rowwise=True):
    print(f'TEST_CASE("test_loss{index}")')
    print('{')

    rows, cols = (N, K) if rowwise else (K, N)

    Y = random_float_matrix(rows, cols, a, b)
    print_cpp_matrix_declaration('Y', Y)

    T = make_target_rowwise(rows, cols) if rowwise else make_target_rowwise(rows, cols)
    print_cpp_matrix_declaration('T', T)

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
    parser = argparse.ArgumentParser(description='Convert and publish analysis files')
    parser.add_argument('--colwise', help='Generate tests for colwise layout', action='store_true')
    args = parser.parse_args()

    np.random.seed(42)
    np.set_printoptions(precision=6)
    rowwise = not args.colwise

    for i in range(1, 6):
        N = 3  # the number of examples
        K = 4  # the number of outputs
        make_loss_test(N, K, 0.000001, 10.0, index=i, rowwise=rowwise)
