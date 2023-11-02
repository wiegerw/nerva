# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from symbolic.nerva_jax.softmax_functions import *

Matrix = jnp.ndarray

class SoftmaxFunction(object):
    def __call__(self, X: Matrix) -> Matrix:
        return softmax_colwise(X)

    def jacobian(self, X: Matrix) -> Matrix:
        return softmax_colwise_jacobian(X)


class StableSoftmaxFunction(object):
    def __call__(self, X: Matrix) -> Matrix:
        return stable_softmax_colwise(X)

    def jacobian(self, X: Matrix) -> Matrix:
        return stable_softmax_colwise_jacobian(X)


class LogSoftmaxFunction(object):
    def __call__(self, X: Matrix) -> Matrix:
        return log_softmax_colwise(X)

    def jacobian(self, X: Matrix) -> Matrix:
        return log_softmax_colwise_jacobian(X)


class StableLogSoftmaxFunction(object):
    def __call__(self, X: Matrix) -> Matrix:
        return stable_log_softmax_colwise(X)

    def jacobian(self, X: Matrix) -> Matrix:
        return stable_log_softmax_colwise_jacobian(X)
