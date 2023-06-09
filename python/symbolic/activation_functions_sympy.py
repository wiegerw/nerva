# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from sympy import Matrix
from symbolic import activation_functions_sympy_1d

def relu(X: Matrix) -> Matrix:
    return X.applyfunc(activation_functions_sympy_1d.relu)


def relu_prime(X: Matrix) -> Matrix:
    return X.applyfunc(activation_functions_sympy_1d.relu_prime)


def leaky_relu(alpha):
    return lambda x: x.applyfunc(activation_functions_sympy_1d.leaky_relu(alpha))


def leaky_relu_prime(alpha):
    return lambda x: x.applyfunc(activation_functions_sympy_1d.leaky_relu_prime(alpha))


def all_relu(alpha):
    return lambda x: x.applyfunc(activation_functions_sympy_1d.all_relu(alpha))


def all_relu_prime(alpha):
    return lambda x: x.applyfunc(activation_functions_sympy_1d.all_relu_prime(alpha))


def hyperbolic_tangent(X: Matrix) -> Matrix:
    return X.applyfunc(activation_functions_sympy_1d.hyperbolic_tangent)


def hyperbolic_tangent_prime(X: Matrix) -> Matrix:
    return X.applyfunc(activation_functions_sympy_1d.hyperbolic_tangent_prime)


def sigmoid(X: Matrix) -> Matrix:
    return X.applyfunc(activation_functions_sympy_1d.sigmoid)


def sigmoid_prime(X: Matrix) -> Matrix:
    return X.applyfunc(activation_functions_sympy_1d.sigmoid_prime)


def srelu(al, tl, ar, tr):
    return lambda x: x.applyfunc(activation_functions_sympy_1d.srelu(al, tl, ar, tr))


def srelu_prime(al, tl, ar, tr):
    return lambda x: x.applyfunc(activation_functions_sympy_1d.srelu_prime(al, tl, ar, tr))
