# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def leaky_relu(alpha):
    return lambda x: np.maximum(alpha * x, x)


def leaky_relu_derivative(alpha):
    return lambda x: np.where(x > 0, 1, alpha)


def all_relu(alpha):
    return lambda x: np.where(x < 0, alpha * x, x)


def all_relu_derivative(alpha):
    return lambda x: np.where(x < 0, alpha, 1)


def hyperbolic_tangent(x):
    return np.tanh(x)


def hyperbolic_tangent_derivative(x):
    return 1 - np.tanh(x) ** 2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def srelu(al, tl, ar, tr):
    return lambda x: np.where(x <= tl, tl + al * (x - tl),
                     np.where(x >= tr, tr + ar * (x - tr), x))


def srelu_derivative(al, tl, ar, tr):
    return lambda x: np.where((x <= tl) | (x >= tr), 0,
                     np.where((tl < x) & (x < tr), 1, al if x < tl else ar))
