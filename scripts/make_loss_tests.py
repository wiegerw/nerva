#!/usr/bin/env python3

# Copyright 2024 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse
import io
from io import StringIO

import numpy as np

import nerva_torch.loss_functions
from nerva_numpy.softmax_functions import stable_softmax
from nerva_torch.loss_functions_torch import *
from test_utilities import random_float_matrix, print_cpp_matrix_declaration, make_target_rowwise, insert_text_in_file, \
    print_torch_matrix_declaration


def generate_loss_test(N: int, K: int, a: float, b: float):
    Y = random_float_matrix(N, K, a, b)
    Y = stable_softmax(Y)  # This is to avoid problems with negative log likelihood

    T = make_target_rowwise(N, K, cover_all_classes=True)
    T = T.astype(float)

    Y = torch.tensor(Y)
    T = torch.tensor(T)

    loss_se = squared_error_loss_torch(Y, T)
    loss_sce = softmax_cross_entropy_loss_torch(Y, T)
    loss_nll = nerva_torch.loss_functions.Negative_log_likelihood_loss(Y, T)
    loss_ce = nerva_torch.loss_functions.Cross_entropy_loss(Y, T)
    loss_lce = nerva_torch.loss_functions.Logistic_cross_entropy_loss(Y, T)

    return Y, T, loss_se, loss_sce, loss_nll, loss_ce, loss_lce


def make_loss_test_cpp(out: StringIO, index: int, Y, T, loss_se, loss_sce, loss_nll, loss_ce, loss_lce):
    out.write(f'TEST_CASE("test_loss{index}")\n')
    out.write('{\n')

    print_cpp_matrix_declaration(out, 'Y', Y)
    print_cpp_matrix_declaration(out, 'T', T)

    name = 'squared_error_loss'
    out.write(f'  test_loss("{name}", {name}(), {loss_se}, Y, T);\n')

    name = 'softmax_cross_entropy_loss'
    out.write(f'  test_loss("{name}", {name}(), {loss_sce}, Y, T);\n')

    name = 'negative_log_likelihood_loss'
    out.write(f'  test_loss("{name}", {name}(), {loss_nll}, Y, T);\n')

    name = 'cross_entropy_loss'
    out.write(f'  test_loss("{name}", {name}(), {loss_ce}, Y, T);\n')

    name = 'logistic_cross_entropy_loss'
    out.write(f'  test_loss("{name}", {name}(), {loss_lce}, Y, T);\n')

    out.write('}\n\n')


#     def test_loss1(self):
#         Y = torch.tensor([
#             [0.36742274, 0.35949028, 0.27308698],
#             [0.30354068, 0.41444678, 0.28201254],
#             [0.34972793, 0.32481684, 0.32545523],
#             [0.34815459, 0.44543710, 0.20640831],
#             [0.19429503, 0.32073754, 0.48496742],
#         ], dtype = torch.float32)
#
#         T = torch.tensor([
#             [0.00000000, 1.00000000, 0.00000000],
#             [1.00000000, 0.00000000, 0.00000000],
#             [0.00000000, 0.00000000, 1.00000000],
#             [0.00000000, 1.00000000, 0.00000000],
#             [0.00000000, 1.00000000, 0.00000000],
#         ], dtype = torch.float32)
#
#         self._test_loss("SquaredErrorLoss", SquaredErrorLoss(), 3.2447052001953125, Y, T)
#         self._test_loss("SoftmaxCrossEntropyLoss", SoftmaxCrossEntropyLoss(), 5.419629096984863, Y, T)
#         self._test_loss("NegativeLogLikelihoodLoss", NegativeLogLikelihoodLoss(), 5.283669471740723, Y, T)
#         self._test_loss("CrossEntropyLoss", CrossEntropyLoss(), 5.283669471740723, Y, T)
#         self._test_loss("LogisticCrossEntropyLoss", LogisticCrossEntropyLoss(), 2.666532516479492, Y, T)
def make_loss_test_python(out: StringIO, index: int, Y, T, loss_se, loss_sce, loss_nll, loss_ce, loss_lce, print_matrix, indent=4):
    out.write(f'    def test_loss{index}(self):\n')

    print_matrix(out, 'Y', Y, indent=indent+4)
    print_matrix(out, 'T', T, indent=indent+4)

    indent = ' ' * indent

    name = 'SquaredErrorLoss'
    loss = loss_se
    out.write(f'{indent}    self._test_loss("{name}", {name}(), {loss}, Y, T)\n')

    name = 'SoftmaxCrossEntropyLoss'
    loss = loss_sce
    out.write(f'{indent}    self._test_loss("{name}", {name}(), {loss}, Y, T)\n')

    name = 'NegativeLogLikelihoodLoss'
    loss = loss_nll
    out.write(f'{indent}    self._test_loss("{name}", {name}(), {loss}, Y, T)\n')

    name = 'CrossEntropyLoss'
    loss = loss_ce
    out.write(f'{indent}    self._test_loss("{name}", {name}(), {loss}, Y, T)\n')

    name = 'LogisticCrossEntropyLoss'
    loss = loss_lce
    out.write(f'{indent}    self._test_loss("{name}", {name}(), {loss}, Y, T)\n')

    out.write('\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make loss function tests')
    parser.add_argument('--seed', help='The seed for the random generator', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    np.set_printoptions(precision=6)

    out_rowwise = io.StringIO()
    out_rowwise_python = io.StringIO()
    out_colwise = io.StringIO()
    out_colwise_python = io.StringIO()
    out_jax = io.StringIO()
    out_numpy = io.StringIO()
    out_tensorflow = io.StringIO()
    out_torch = io.StringIO()

    for i in range(1, 6):
        N = 5  # the number of examples
        K = 3  # the number of classes
        a = 0.000001
        b = 1.0
        Y, T, loss_se, loss_sce, loss_nll, loss_ce, loss_lce = generate_loss_test(N, K, a, b)
        make_loss_test_cpp(out_rowwise, i, Y, T, loss_se, loss_sce, loss_nll, loss_ce, loss_lce)
        make_loss_test_cpp(out_colwise, i, Y.T, T.T, loss_se, loss_sce, loss_nll, loss_ce, loss_lce)
        make_loss_test_python(out_rowwise_python, i, Y, T, loss_se, loss_sce, loss_nll, loss_ce, loss_lce, print_matrix=print_torch_matrix_declaration)
        make_loss_test_python(out_colwise_python, i, Y.T, T.T, loss_se, loss_sce, loss_nll, loss_ce, loss_lce, print_matrix=print_torch_matrix_declaration)
        # make_loss_test_python(out_jax, i, Y, T, loss_se, loss_sce, loss_nll, loss_ce, loss_lce, print_matrix=print_jax_matrix_declaration)
        # make_loss_test_python(out_numpy, i, Y, T, loss_se, loss_sce, loss_nll, loss_ce, loss_lce, print_matrix=print_numpy_matrix_declaration)
        # make_loss_test_python(out_tensorflow, i, Y, T, loss_se, loss_sce, loss_nll, loss_ce, loss_lce, print_matrix=print_tensorflow_matrix_declaration)
        # make_loss_test_python(out_torch, i, Y, T, loss_se, loss_sce, loss_nll, loss_ce, loss_lce, print_matrix=print_torch_matrix_declaration)

    begin_label='//--- begin generated code ---//'
    end_label='//--- end generated code ---//'
    insert_text_in_file('../../nerva-rowwise/tests/loss_function_test.cpp', out_rowwise.getvalue(), begin_label=begin_label, end_label=end_label)
    insert_text_in_file('../../nerva-colwise/tests/loss_function_test.cpp', out_colwise.getvalue(), begin_label=begin_label, end_label=end_label)

    begin_label='#--- begin generated code ---#'
    end_label='#--- end generated code ---#'
    insert_text_in_file('../../nerva-rowwise/python/tests/loss_function_test.py', out_rowwise_python.getvalue(), begin_label=begin_label, end_label=end_label)
    insert_text_in_file('../../nerva-colwise/python/tests/loss_function_test.py', out_colwise_python.getvalue(), begin_label=begin_label, end_label=end_label)
