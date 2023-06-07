# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import torch


def relu(x):
    return torch.relu(x)


def relu_derivative(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))


def leaky_relu(alpha):
    return lambda x: torch.nn.functional.leaky_relu(x, alpha)


def leaky_relu_derivative(alpha):
    return lambda x: torch.where(x > 0, torch.ones_like(x), torch.full_like(x, alpha))


def all_relu(alpha):
    return lambda x: torch.nn.functional.prelu(x, torch.full_like(x, alpha))


def all_relu_derivative(alpha):
    return lambda x: torch.where(x < 0, torch.full_like(x, alpha), torch.ones_like(x))


def hyperbolic_tangent(x):
    return torch.tanh(x)


def hyperbolic_tangent_derivative(x):
    return 1 - torch.tanh(x) ** 2


def sigmoid(x):
    return torch.sigmoid(x)


def sigmoid_derivative(x):
    y = torch.sigmoid(x)
    return y * (1 - y)


# TODO: check this
def srelu(al, tl, ar, tr):
    return lambda x: torch.where(x <= tl, tl + al * (x - tl),
                     torch.where(x >= tr, tr + ar * (x - tr), x))

# TODO: check this
def srelu_derivative(al, tl, ar, tr, x):
    return lambda x: torch.where((x <= tl) | (x >= tr), torch.zeros_like(x),
                     torch.where((tl < x) & (x < tr), torch.ones_like(x),
                     torch.where(x < tl, torch.full_like(x, al), torch.full_like(x, ar))))
