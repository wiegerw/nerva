# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import torch


def Relu(X: torch.Tensor):
    return torch.max(torch.zeros_like(X), X)


def Relu_gradient(X: torch.Tensor):
    return torch.where(X > 0, torch.ones_like(X), torch.zeros_like(X))


def Leaky_relu(alpha):
    return lambda X: torch.max(X, alpha * X)


def Leaky_relu_gradient(alpha):
    return lambda X: torch.where(X > 0, torch.ones_like(X), torch.full_like(X, alpha))


def All_relu(alpha):
    return lambda X: torch.where(X < 0, alpha * X, X)


def All_relu_gradient(alpha):
    return lambda X: torch.where(X < 0, alpha, 1)


def Hyperbolic_tangent(X: torch.Tensor):
    return torch.tanh(X)


def Hyperbolic_tangent_gradient(X: torch.Tensor):
    return 1 - torch.tanh(X) ** 2


def Sigmoid(X: torch.Tensor):
    return torch.sigmoid(X)


def Sigmoid_gradient(X: torch.Tensor):
    y = torch.sigmoid(X)
    return y * (1 - y)


def Srelu(al, tl, ar, tr):
    return lambda X: torch.where(X <= tl, tl + al * (X - tl),
                     torch.where(X < tr, X, tr + ar * (X - tr)))


def Srelu_gradient(al, tl, ar, tr):
    return lambda X: torch.where(X <= tl, al,
                     torch.where(X < tr, 1, ar))
