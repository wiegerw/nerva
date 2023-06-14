# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import sympy as sp
from sympy import Lambda, Matrix, Piecewise

# Naming conventions:
# - lowercase functions operate on real numbers
# - uppercase functions operate on matrices

def relu(x):
    # return max(0, x)
    return Piecewise((0, x < 0), (x, True))


def relu_derivative(x):
    # return 0 if x < 0 else 1
    return Piecewise((0, x < 0), (1, True))


def leaky_relu(alpha):
    x = sp.symbols('x')
    # fx = max(alpha * x, x)
    fx = Piecewise((alpha * x, x < alpha * x), (x, True))
    return Lambda(x, fx)


def leaky_relu_derivative(alpha):
    x = sp.symbols('x')
    # fx = alpha if x < alpha * x else 1
    fx = Piecewise((alpha, x < alpha * x), (1, True))
    return Lambda(x, fx)


def all_relu(alpha):
    x = sp.symbols('x')
    # fx = alpha * x if x < 0 else x
    fx = Piecewise((alpha * x, x < 0), (x, True))
    return Lambda(x, fx)


def all_relu_derivative(alpha):
    x = sp.symbols('x')
    # fx = alpha if x < 0 else 1
    fx = Piecewise((alpha, x < 0), (1, True))
    return Lambda(x, fx)


def hyperbolic_tangent(x):
    return sp.tanh(x)


def hyperbolic_tangent_derivative(x):
    y = hyperbolic_tangent(x)
    return 1 - y * y


def sigmoid(x):
    return 1 / (1 + sp.exp(-x))


def sigmoid_derivative(x):
    y = sigmoid(x)
    return y * (1 - y)


def srelu(al, tl, ar, tr):
    x = sp.symbols('x')
    return Lambda(x, Piecewise((tl + al * (x - tl), x <= tl), (x, x < tr), (tr + ar * (x - tr), True)))


def srelu_derivative(al, tl, ar, tr):
    x = sp.symbols('x')
    return Lambda(x, Piecewise((al, x <= tl), (1, x < tr), (ar, True)))


def Relu(X: Matrix) -> Matrix:
    return X.applyfunc(relu)


def Relu_gradient(X: Matrix) -> Matrix:
    return X.applyfunc(relu_derivative)


def Leaky_relu(alpha):
    return lambda x: x.applyfunc(leaky_relu(alpha))


def Leaky_relu_gradient(alpha):
    return lambda x: x.applyfunc(leaky_relu_derivative(alpha))


def All_relu(alpha):
    return lambda x: x.applyfunc(all_relu(alpha))


def All_relu_gradient(alpha):
    return lambda x: x.applyfunc(all_relu_derivative(alpha))


def Hyperbolic_tangent(X: Matrix) -> Matrix:
    return X.applyfunc(hyperbolic_tangent)


def Hyperbolic_tangent_gradient(X: Matrix) -> Matrix:
    return X.applyfunc(hyperbolic_tangent_derivative)


def Sigmoid(X: Matrix) -> Matrix:
    return X.applyfunc(sigmoid)


def Sigmoid_gradient(X: Matrix) -> Matrix:
    return X.applyfunc(sigmoid_derivative)


def Srelu(al, tl, ar, tr):
    return lambda x: x.applyfunc(srelu(al, tl, ar, tr))


def Srelu_gradient(al, tl, ar, tr):
    return lambda x: x.applyfunc(srelu_derivative(al, tl, ar, tr))
