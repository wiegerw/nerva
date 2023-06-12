# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from symbolic.softmax_numpy import *

# https://stackoverflow.com/questions/16774849/mean-squared-error-in-numpy

def squared_error_loss_colwise(Y, T):
    return np.sum(np.square(Y - T))


def squared_error_loss_colwise_gradient(Y, T):
    return 2 * (Y - T)


def mean_squared_error_loss_colwise(Y, T):
    return np.sum(np.square(Y - T)) / Y.size


def mean_squared_error_loss_colwise_gradient(Y, T):
    return 2 * (Y - T) / Y.size


def cross_entropy_loss_colwise(Y, T):
    m = Y.shape[0]
    return -np.sum(T * np.log(Y)) / m


def softmax_cross_entropy_loss_colwise(Y, T):
    m = Y.shape[0]
    return -np.sum(T * np.log(stable_softmax_rowwise(Y))) / m


def stable_softmax_cross_entropy_loss_colwise(Y, T):
    m = Y.shape[0]
    Y = Y - np.max(Y, axis=1, keepdims=True)
    softmax_Y = np.exp(Y) / np.sum(np.exp(Y), axis=1, keepdims=True)
    return -np.sum(T * np.log(softmax_Y)) / m
