#!/usr/bin/env python3

# Copyright 2024 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse
from io import StringIO

import numpy as np

import nerva_torch.loss_functions
from nerva_numpy.softmax_functions import stable_softmax
from nerva_torch.loss_functions_torch import *
from test_utilities import random_float_matrix, print_cpp_matrix_declaration, make_target_rowwise, insert_text_in_file


def make_loss_test(out: StringIO, N: int, K: int, a: float, b: float, index=0, rowwise=True):
    out.write(f'TEST_CASE("test_loss{index}")\n')
    out.write('{\n')

    Y = random_float_matrix(N, K, a, b)
    Y = stable_softmax(Y)  # This is to avoid problems with negative log likelihood
    print_cpp_matrix_declaration(out, 'Y', Y if rowwise else Y.T)

    T = make_target_rowwise(N, K, cover_all_classes=True)
    print_cpp_matrix_declaration(out, 'T', T if rowwise else T.T)

    T = T.astype(float)

    Y = torch.Tensor(Y)
    T = torch.Tensor(T)

    loss = squared_error_loss_torch(Y, T)
    name = 'squared_error_loss'
    out.write(f'  test_loss("{name}", {name}(), {loss}, Y, T);\n')

    loss = softmax_cross_entropy_loss_torch(Y, T)
    name = 'softmax_cross_entropy_loss'
    out.write(f'  test_loss("{name}", {name}(), {loss}, Y, T);\n')

    # No PyTorch equivalent available (PyTorch's NLLLoss does not apply the log)
    loss = nerva_torch.loss_functions.Negative_log_likelihood_loss(Y, T)
    name = 'negative_log_likelihood_loss'
    out.write(f'  test_loss("{name}", {name}(), {loss}, Y, T);\n')

    # No PyTorch equivalent available
    loss = nerva_torch.loss_functions.Cross_entropy_loss(Y, T)
    name = 'cross_entropy_loss'
    out.write(f'  test_loss("{name}", {name}(), {loss}, Y, T);\n')

    # No PyTorch equivalent available(?)
    loss = nerva_torch.loss_functions.Logistic_cross_entropy_loss(Y, T)
    name = 'logistic_cross_entropy_loss'
    out.write(f'  test_loss("{name}", {name}(), {loss}, Y, T);\n')

    out.write('}\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert and publish analysis files')
    parser.add_argument('--colwise', help='Generate tests for colwise layout', action='store_true')
    args = parser.parse_args()

    # np.random.seed(42)
    np.set_printoptions(precision=6)
    rowwise = not args.colwise

    out = StringIO()
    for i in range(1, 6):
        N = 5  # the number of examples
        K = 3  # the number of classes
        make_loss_test(out, N, K, 0.000001, 1.0, index=i, rowwise=rowwise)
    text = out.getvalue()
    insert_text_in_file('loss_function_test.cpp', text)
