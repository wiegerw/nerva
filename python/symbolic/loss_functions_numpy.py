# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import numpy as np

# https://stackoverflow.com/questions/16774849/mean-squared-error-in-numpy

def squared_error_loss(Y, T):
    return np.sum(np.square(Y - T))


def squared_error_loss_gradient(Y, T):
    return 2 * (Y - T)


def mean_squared_error_loss(Y, T):
    return np.sum(np.square(Y - T)) / Y.size


def mean_squared_error_loss_gradient(Y, T):
    return 2 * (Y - T) / Y.size
