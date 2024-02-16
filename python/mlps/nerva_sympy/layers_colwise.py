# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import sympy as sp
from sympy import Piecewise

from mlps.nerva_sympy.activation_functions import ActivationFunction, SReLUActivation, parse_activation
from mlps.nerva_sympy.matrix_operations import column_repeat, columns_sum, diag, elements_sum, hadamard, \
    identity, ones, div_sqrt, row_repeat, rows_mean, rows_sum, vector_size, zeros
from mlps.nerva_sympy.optimizers import CompositeOptimizer, parse_optimizer
from mlps.nerva_sympy.softmax_functions import log_softmax_colwise, softmax_colwise
from mlps.nerva_sympy.weight_initializers import set_layer_weights

Matrix = sp.Matrix


class Layer(object):
    """
    Base class for layers of a neural network with data in column layout
    """
    def __init__(self):
        self.X = None
        self.DX = None
        self.optimizer = None

    def feedforward(self, X: Matrix) -> Matrix:
        raise NotImplementedError

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        raise NotImplementedError

    def optimize(self, eta):
        if self.optimizer:
            self.optimizer.update(eta)


class LinearLayer(Layer):
    """
    Linear layer of a neural network
    """
    def __init__(self, D: int, K: int):
        super().__init__()
        self.W = zeros(K, D)
        self.DW = zeros(K, D)
        self.b = zeros(K, 1)
        self.Db = zeros(K, 1)
        self.optimizer = None

    def feedforward(self, X: Matrix) -> Matrix:
        self.X = X
        D, N = X.shape
        W = self.W
        b = self.b

        Y = W * X + column_repeat(b, N)

        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        X = self.X
        W = self.W

        DW = DY * X.T
        Db = rows_sum(DY)
        DX = W.T * DY

        self.DW[:] = DW
        self.Db[:] = Db
        self.DX = DX

    def input_size(self) -> int:
        return self.W.shape[1]

    def output_size(self) -> int:
        return self.W.shape[0]

    def set_optimizer(self, optimizer: str):
        make_optimizer = parse_optimizer(optimizer)
        self.optimizer = CompositeOptimizer([make_optimizer(self.W, self.DW), make_optimizer(self.b, self.Db)])

    def set_weights(self, weight_initializer):
        set_layer_weights(self, weight_initializer)


class ActivationLayer(LinearLayer):
    """
    Linear layer with an activation function
    """
    def __init__(self, D: int, K: int, act: ActivationFunction):
        super().__init__(D, K)
        self.Z = None
        self.DZ = None
        self.act = act

    def feedforward(self, X: Matrix) -> Matrix:
        self.X = X
        D, N = X.shape
        W = self.W
        b = self.b
        act = self.act

        Z = W * X + column_repeat(b, N)
        Y = act(Z)

        self.Z = Z
        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        X = self.X
        W = self.W
        Z = self.Z
        act = self.act

        DZ = hadamard(DY, act.gradient(Z))
        DW = DZ * X.T
        Db = rows_sum(DZ)
        DX = W.T * DZ

        self.DZ = DZ
        self.DW[:] = DW
        self.Db[:] = Db
        self.DX = DX


class SReLULayer(ActivationLayer):
    """
    Linear layer with an SReLU activation function. It adds learning of the parameters
    al, tl, ar and tr.
    """
    def __init__(self, D: int, K: int, act: SReLUActivation):
        super().__init__(D, K, act)
        self.Dal = 0
        self.Dtl = 0
        self.Dar = 0
        self.Dtr = 0

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        super().backpropagate(Y, DY)
        Z = self.Z
        al = self.act.al
        tl = self.act.tl
        ar = self.act.ar
        tr = self.act.tr

        Al = lambda Z: Z.applyfunc(lambda Zij: Piecewise((Zij - tl, Zij <= tl), (0, True)))
        Ar = lambda Z: Z.applyfunc(lambda Zij: Piecewise((0, Zij <= tl), (0, Zij < tr), (Zij - tr, True)))
        Tl = lambda Z: Z.applyfunc(lambda Zij: Piecewise((1 - al, Zij <= tl), (0, True)))
        Tr = lambda Z: Z.applyfunc(lambda Zij: Piecewise((0, Zij <= tl), (0, Zij < tr), (1 - ar, True)))

        self.Dal = elements_sum(hadamard(DY, Al(Z)))
        self.Dar = elements_sum(hadamard(DY, Ar(Z)))
        self.Dtl = elements_sum(hadamard(DY, Tl(Z)))
        self.Dtr = elements_sum(hadamard(DY, Tr(Z)))

    def set_optimizer(self, optimizer: str):
        make_optimizer = parse_optimizer(optimizer)
        self.optimizer = CompositeOptimizer([make_optimizer(self.W, self.DW),
                                             make_optimizer(self.b, self.Db),
                                             make_optimizer(self.act.x, self.act.Dx)
                                            ])


class SoftmaxLayer(LinearLayer):
    """
    Linear layer with a softmax activation function
    """
    def __init__(self, D: int, K: int):
        super().__init__(D, K)
        self.Z = None
        self.DZ = None

    def feedforward(self, X: Matrix) -> Matrix:
        self.X = X
        D, N = X.shape
        W = self.W
        b = self.b

        Z = W * X + column_repeat(b, N)
        Y = softmax_colwise(Z)

        self.Z = Z
        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        K, N = self.Z.shape
        X = self.X
        W = self.W

        DZ = hadamard(Y, DY - row_repeat(diag(Y.T * DY).T, K))
        DW = DZ * X.T
        Db = rows_sum(DZ)
        DX = W.T * DZ

        self.DZ = DZ
        self.DW[:] = DW
        self.Db[:] = Db
        self.DX = DX


class LogSoftmaxLayer(LinearLayer):
    """
    Linear layer with a log_softmax activation function
    """
    def __init__(self, D: int, K: int):
        super().__init__(D, K)
        self.Z = None
        self.DZ = None

    def feedforward(self, X: Matrix) -> Matrix:
        self.X = X
        D, N = X.shape
        W = self.W
        b = self.b

        Z = W * X + column_repeat(b, N)
        Y = log_softmax_colwise(Z)

        self.Z = Z
        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        K, N = self.Z.shape
        X = self.X
        W = self.W
        Z = self.Z

        DZ = DY - hadamard(softmax_colwise(Z), row_repeat(columns_sum(DY), K))
        DW = DZ * X.T
        Db = rows_sum(DZ)
        DX = W.T * DZ

        self.DZ = DZ
        self.DW[:] = DW
        self.Db[:] = Db
        self.DX = DX


class BatchNormalizationLayer(Layer):
    """
    A batch normalization layer
    """
    def __init__(self, D: int):
        super().__init__()
        self.Z = None
        self.DZ = None
        self.gamma = ones(D, 1)
        self.Dgamma = zeros(D, 1)
        self.beta = zeros(D, 1)
        self.Dbeta = zeros(D, 1)
        self.div_sqrt_Sigma = zeros(D, 1)
        self.optimizer = None

    def feedforward(self, X: Matrix) -> Matrix:
        self.X = X
        D, N = X.shape
        gamma = self.gamma
        beta = self.beta

        R = X - column_repeat(rows_mean(X), N)
        Sigma = diag(R * R.T) / N
        div_sqrt_Sigma = div_sqrt(Sigma)
        Z = hadamard(column_repeat(div_sqrt_Sigma, N), R)
        Y = hadamard(column_repeat(gamma, N), Z) + column_repeat(beta, N)

        self.div_sqrt_Sigma[:] = div_sqrt_Sigma
        self.Z = Z
        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        D, N = self.X.shape
        Z = self.Z
        gamma = self.gamma
        div_sqrt_Sigma = self.div_sqrt_Sigma

        DZ = hadamard(column_repeat(gamma, N), DY)
        Dbeta = rows_sum(DY)
        Dgamma = rows_sum(hadamard(DY, Z))
        DX = hadamard(column_repeat(div_sqrt_Sigma / N, N), hadamard(Z, column_repeat(-diag(DZ * Z.T), N)) + DZ * (N * identity(N) - ones(N, N)))

        self.DZ = DZ
        self.Dbeta[:] = Dbeta
        self.Dgamma[:] = Dgamma
        self.DX = DX

    def input_size(self) -> int:
        return vector_size(self.gamma)

    def output_size(self) -> int:
        return vector_size(self.gamma)

    def set_optimizer(self, optimizer: str):
        make_optimizer = parse_optimizer(optimizer)
        self.optimizer = CompositeOptimizer([make_optimizer(self.beta, self.Dbeta), make_optimizer(self.gamma, self.Dgamma)])


def parse_linear_layer(text: str,
                       D: int,
                       K: int,
                       optimizer: str,
                       weight_initializer: str
                      ) -> Layer:
    if text == 'Linear':
        layer = LinearLayer(D, K)
    elif text == 'Softmax':
        layer = SoftmaxLayer(D, K)
    elif text == 'LogSoftmax':
        layer = LogSoftmaxLayer(D, K)
    elif text.startswith('SReLU'):
        act = parse_activation(text)
        layer = SReLULayer(D, K, act)
    else:
        act = parse_activation(text)
        layer = ActivationLayer(D, K, act)
    layer.set_optimizer(optimizer)
    layer.set_weights(weight_initializer)
    return layer
