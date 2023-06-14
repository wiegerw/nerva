# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import numpy as np


def Relu(X: np.ndarray):
    return np.maximum(0, X)


def Relu_gradient(X: np.ndarray):
    return np.where(X > 0, 1, 0)


def Leaky_relu(alpha):
    return lambda X: np.maximum(alpha * X, X)


def Leaky_relu_gradient(alpha):
    return lambda X: np.where(X > 0, 1, alpha)


def All_relu(alpha):
    return lambda X: np.where(X < 0, alpha * X, X)


def All_relu_gradient(alpha):
    return lambda X: np.where(X < 0, alpha, 1)


def Hyperbolic_tangent(X: np.ndarray):
    return np.tanh(X)


def Hyperbolic_tangent_gradient(X: np.ndarray):
    return 1 - np.tanh(X) ** 2


def Sigmoid(X: np.ndarray):
    return 1 / (1 + np.exp(-X))


def Sigmoid_gradient(X: np.ndarray):
    return Sigmoid(X) * (1 - Sigmoid(X))


def Srelu(al, tl, ar, tr):
    return lambda X: np.where(X <= tl, tl + al * (X - tl),
                     np.where(X < tr, X, tr + ar * (X - tr)))


def Srelu_gradient(al, tl, ar, tr):
    return lambda X: np.where(X <= tl, al,
                     np.where(X < tr, 1, ar))
