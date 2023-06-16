# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import re
import torch

Matrix = torch.Tensor


def Relu(X: Matrix):
    return torch.max(torch.zeros_like(X), X)


def Relu_gradient(X: Matrix):
    return torch.where(X > 0, torch.ones_like(X), torch.zeros_like(X))


def Leaky_relu(alpha):
    return lambda X: torch.max(X, alpha * X)


def Leaky_relu_gradient(alpha):
    return lambda X: torch.where(X > 0, torch.ones_like(X), torch.full_like(X, alpha))


def All_relu(alpha):
    return lambda X: torch.where(X < 0, alpha * X, X)


def All_relu_gradient(alpha):
    return lambda X: torch.where(X < 0, alpha, 1)


def Hyperbolic_tangent(X: Matrix):
    return torch.tanh(X)


def Hyperbolic_tangent_gradient(X: Matrix):
    return 1 - torch.tanh(X) ** 2


def Sigmoid(X: Matrix):
    return torch.sigmoid(X)


def Sigmoid_gradient(X: Matrix):
    y = torch.sigmoid(X)
    return y * (1 - y)


def Srelu(al, tl, ar, tr):
    return lambda X: torch.where(X <= tl, tl + al * (X - tl),
                     torch.where(X < tr, X, tr + ar * (X - tr)))


def Srelu_gradient(al, tl, ar, tr):
    return lambda X: torch.where(X <= tl, al,
                     torch.where(X < tr, 1, ar))


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
    def __init__(self, al=0, tl=0, ar=0, tr=1):
        self.al = al
        self.tl = tl
        self.ar = ar
        self.tr = tr

    def __call__(self, X: Matrix) -> Matrix:
        return Srelu(self.al, self.tl, self.ar, self.tr)(X)

    def gradient(self, X: Matrix) -> Matrix:
        return Srelu_gradient(self.al, self.tl, self.ar, self.tr)(X)


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
