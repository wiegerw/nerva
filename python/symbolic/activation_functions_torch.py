# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import torch


def relu(x):
    return torch.max(torch.zeros_like(x), x)


def relu_prime(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))


def leaky_relu(alpha):
    return lambda x: torch.max(x, alpha * x)


def leaky_relu_prime(alpha):
    return lambda x: torch.where(x > 0, torch.ones_like(x), torch.full_like(x, alpha))


def all_relu(alpha):
    return lambda x: torch.where(x < 0, alpha * x, x)


def all_relu_prime(alpha):
    return lambda x: torch.where(x < 0, alpha, 1)


def hyperbolic_tangent(x):
    return torch.tanh(x)


def hyperbolic_tangent_prime(x):
    return 1 - torch.tanh(x) ** 2


def sigmoid(x):
    return torch.sigmoid(x)


def sigmoid_prime(x):
    y = torch.sigmoid(x)
    return y * (1 - y)


def srelu(al, tl, ar, tr):
    return lambda x: torch.where(x <= tl, tl + al * (x - tl),
                     torch.where(x < tr, x, tr + ar * (x - tr)))


def srelu_prime(al, tl, ar, tr):
    return lambda x: torch.where(x <= tl, al,
                     torch.where(x < tr, 1, ar))
