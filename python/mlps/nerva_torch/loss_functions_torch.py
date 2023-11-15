# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)


import torch

#----------------------------------------------#
#         torch loss functions (used for testing)
#----------------------------------------------#

def mean_squared_error_loss_torch(Y, T):
    loss = torch.nn.MSELoss(reduction='sum')
    return loss(Y, T)


def softmax_cross_entropy_loss_torch(Y, T):
    loss = torch.nn.CrossEntropyLoss(reduction='sum')
    return loss(Y, T)


def negative_log_likelihood_loss_torch(Y, T):
    loss = torch.nn.NLLLoss(reduction='sum')
    return loss(Y, T)
