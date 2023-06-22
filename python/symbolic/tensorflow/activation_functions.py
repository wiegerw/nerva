# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

Matrix = tf.Tensor


def Relu(X: Matrix):
    return tf.maximum(0.0, X)


def Relu_gradient(X: Matrix):
    return tf.where(X > 0, tf.ones_like(X), tf.zeros_like(X))


def Leaky_relu(alpha):
    return lambda X: tf.maximum(alpha * X, X)


def Leaky_relu_gradient(alpha):
    return lambda X: tf.where(X > 0, tf.ones_like(X), tf.fill(tf.shape(X), tf.cast(alpha, X.dtype)))


def All_relu(alpha):
    return lambda X: tf.where(X < 0, tf.cast(alpha, X.dtype) * X, X)


def All_relu_gradient(alpha):
    return lambda X: tf.where(X < 0, tf.fill(tf.shape(X), tf.cast(alpha, X.dtype)), tf.ones_like(X))


def Hyperbolic_tangent(X: Matrix):
    return tf.math.tanh(X)


def Hyperbolic_tangent_gradient(X: Matrix):
    return 1 - tf.math.tanh(X) ** 2


def Sigmoid(X: Matrix):
    return tf.math.sigmoid(X)


def Sigmoid_gradient(X: Matrix):
    y = tf.math.sigmoid(X)
    return y * (1 - y)


def Srelu(al, tl, ar, tr):
    return lambda X: tf.where(X <= tl, tl + al * (X - tl),
                     tf.where(X < tr, X, tr + ar * (X - tr)))


def Srelu_gradient(al, tl, ar, tr):
    return lambda X: tf.where(X <= tl, al,
                     tf.where(X < tr, 1., ar))


class ActivationFunction(object):
    def __call__(self, X: Matrix) -> Matrix:
        raise NotImplementedError

    def gradient(self, X: Matrix) -> Matrix:
        raise NotImplementedError


class ReLUActivation(ActivationFunction):
    def __call__(self, X: Matrix) -> Matrix:
        return Relu(X)

    def gradient(self, X: Matrix) -> Matrix:
        return Relu_gradient(X)


class LeakyReLUActivation(ActivationFunction):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, X: Matrix) -> Matrix:
        return Leaky_relu(self.alpha)(X)

    def gradient(self, X: Matrix) -> Matrix:
        return Leaky_relu_gradient(self.alpha)(X)


class AllReLUActivation(ActivationFunction):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, X: Matrix) -> Matrix:
        return All_relu(self.alpha)(X)

    def gradient(self, X: Matrix) -> Matrix:
        return All_relu_gradient(self.alpha)(X)


class HyperbolicTangentActivation(ActivationFunction):
    def __call__(self, X: Matrix) -> Matrix:
        return Hyperbolic_tangent(X)

    def gradient(self, X: Matrix) -> Matrix:
        return Hyperbolic_tangent_gradient(X)


class SigmoidActivation(ActivationFunction):
    def __call__(self, X: Matrix) -> Matrix:
        return Sigmoid(X)

    def gradient(self, X: Matrix) -> Matrix:
        return Sigmoid_gradient(X)


class SReLUActivation(ActivationFunction):
    def __init__(self, al=0.0, tl=0.0, ar=0.0, tr=1.0):
        # Store the parameters and their gradients in matrices.
        # This is to make them usable for optimizers.
        self.x = tf.Variable(tf.constant([al, tl, ar, tr], dtype=tf.float32))
        self.Dx = tf.Variable(tf.constant([0, 0, 0, 0], dtype=tf.float32))

    def __call__(self, X: Matrix) -> Matrix:
        al, tl, ar, tr = self.x
        return Srelu(al, tl, ar, tr)(X)

    def gradient(self, X: Matrix) -> Matrix:
        al, tl, ar, tr = self.x
        return Srelu_gradient(al, tl, ar, tr)(X)


def parse_activation(text: str) -> ActivationFunction:
    try:
        if text == 'ReLU':
            return ReLUActivation()
        elif text == 'HyperbolicTangent':
            return HyperbolicTangentActivation()
        elif text.startswith('AllReLU'):
            m = re.match(r'AllReLU\((.*)\)$', text)
            alpha = float(m.group(1))
            return AllReLUActivation(alpha)
        elif text.startswith('LeakyReLU'):
            m = re.match(r'LeakyReLU\((.*)\)$', text)
            alpha = float(m.group(1))
            return LeakyReLUActivation(alpha)
    except:
        pass
    raise RuntimeError(f'Could not parse activation "{text}"')


def parse_srelu_activation(text: str) -> SReLUActivation:
    try:
        if text == 'SReLU':
            return SReLUActivation()
        else:
            m = re.match(r'SReLU\(([^,]*),([^,]*),([^,]*),([^,]*)\)$', text)
            al = float(m.group(1))
            tl = float(m.group(2))
            ar = float(m.group(3))
            tr = float(m.group(4))
            return SReLUActivation(al, tl, ar, tr)
    except:
        pass
    raise RuntimeError(f'Could not parse SReLU activation "{text}"')
