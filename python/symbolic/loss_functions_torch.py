# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import torch


def squared_error_loss(Y: torch.Tensor, T: torch.Tensor):
    return torch.sum(torch.pow(Y - T, 2))


def squared_error_loss_gradient(Y: torch.Tensor, T: torch.Tensor):
    return 2 * (Y - T)


def mean_squared_error_loss(Y, T):
    return torch.sum(torch.pow(Y - T, 2)) / Y.numel()


def mean_squared_error_loss_gradient(Y, T):
    return 2 * (Y - T) / Y.numel()
