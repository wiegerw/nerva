# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import jax.numpy as jnp

from mlps.nerva_jax.softmax_functions import log_softmax_rowwise, log_softmax_rowwise_jacobian, softmax_rowwise, \
    softmax_rowwise_jacobian, stable_log_softmax_rowwise, stable_log_softmax_rowwise_jacobian, stable_softmax_rowwise, \
    stable_softmax_rowwise_jacobian

Matrix = jnp.ndarray

class SoftmaxFunction(object):
    def __call__(self, X: Matrix) -> Matrix:
        return softmax_rowwise(X)

    def jacobian(self, X: Matrix) -> Matrix:
        return softmax_rowwise_jacobian(X)


class StableSoftmaxFunction(object):
    def __call__(self, X: Matrix) -> Matrix:
        return stable_softmax_rowwise(X)

    def jacobian(self, X: Matrix) -> Matrix:
        return stable_softmax_rowwise_jacobian(X)


class LogSoftmaxFunction(object):
    def __call__(self, X: Matrix) -> Matrix:
        return log_softmax_rowwise(X)

    def jacobian(self, X: Matrix) -> Matrix:
        return log_softmax_rowwise_jacobian(X)


class StableLogSoftmaxFunction(object):
    def __call__(self, X: Matrix) -> Matrix:
        return stable_log_softmax_rowwise(X)

    def jacobian(self, X: Matrix) -> Matrix:
        return stable_log_softmax_rowwise_jacobian(X)
