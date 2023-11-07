# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import sympy as sp

from mlps.nerva_sympy.activation_functions import ActivationFunction, SReLUActivation, Sigmoid
from mlps.nerva_sympy.matrix_operations import column_repeat, columns_mean, columns_sum, diag, elements_sum, hadamard, \
    identity, ones, power_minus_half, row_repeat, rows_sum, vector_size, zeros
from mlps.nerva_sympy.optimizers import CompositeOptimizer
from mlps.nerva_sympy.parse_mlp import parse_optimizer
from mlps.nerva_sympy.softmax_functions import log_softmax_rowwise, softmax_rowwise
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
        self.b = zeros(1, K)
        self.Db = zeros(1, K)
        self.optimizer = None

    def feedforward(self, X: Matrix) -> Matrix:
        self.X = X
        N, D = X.shape
        W = self.W
        b = self.b

        Y = X * W.T + row_repeat(b, N)

        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        X = self.X
        W = self.W

        DW = DY.T * X
        Db = columns_sum(DY)
        DX = DY * W

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
        N, D = X.shape
        W = self.W
        b = self.b
        act = self.act

        Z = X * W.T + row_repeat(b, N)
        Y = act(Z)

        self.Z = Z
        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        X = self.X
        W = self.W
        Z = self.Z
        act = self.act

        DZ = hadamard(DY, act.gradient(Z))
        DW = DZ.T * X
        Db = columns_sum(DZ)
        DX = DZ * W

        self.DZ = DZ
        self.DW[:] = DW
        self.Db[:] = Db
        self.DX = DX


class SigmoidLayer(LinearLayer):
    """
    Linear layer with a sigmoid activation function. This is not strictly needed,
    but it shows that the backpropagation can be calculated in a different way
    """
    def __init__(self, D: int, K: int):
        super().__init__(D, K)
        self.Z = None
        self.DZ = None

    def feedforward(self, X: Matrix) -> Matrix:
        self.X = X
        N, D = X.shape
        W = self.W
        b = self.b

        Z = X * W.T + row_repeat(b, N)
        Y = Sigmoid(Z)

        self.Z = Z
        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        K, N = self.Z.shape
        X = self.X
        W = self.W

        DZ = hadamard(DY, hadamard(Y, ones(N, K) - Y))
        DW = DZ.T * X
        Db = columns_sum(DZ)
        DX = DZ * W

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
        N, D = X.shape
        W = self.W
        b = self.b

        Z = X * W.T + row_repeat(b, N)
        Y = softmax_rowwise(Z)

        self.Z = Z
        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        K, N = self.Z.shape
        X = self.X
        W = self.W

        DZ = hadamard(Y, DY - column_repeat(diag(DY * Y.T), N))
        DW = DZ.T * X
        Db = columns_sum(DZ)
        DX = DZ * W

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
        N, D = X.shape
        W = self.W
        b = self.b

        Z = X * W.T + row_repeat(b, N)
        Y = log_softmax_rowwise(Z)

        self.Z = Z
        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        K, N = self.Z.shape
        X = self.X
        W = self.W
        Z = self.Z

        DZ = DY - hadamard(softmax_rowwise(Z), column_repeat(rows_sum(DY), N))
        DW = DZ.T * X
        Db = columns_sum(DZ)
        DX = DZ * W

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
        self.gamma = ones(1, D)
        self.Dgamma = zeros(1, D)
        self.beta = zeros(1, D)
        self.Dbeta = zeros(1, D)
        self.power_minus_half_Sigma = zeros(1, D)
        self.optimizer = None

    def feedforward(self, X: Matrix) -> Matrix:
        self.X = X
        N, D = X.shape
        gamma = self.gamma
        beta = self.beta

        R = X - row_repeat(columns_mean(X), N)
        Sigma = diag(R.T * R).T / N
        power_minus_half_Sigma = power_minus_half(Sigma)
        Z = hadamard(row_repeat(power_minus_half_Sigma, N), R)
        Y = hadamard(row_repeat(gamma, N), Z) + row_repeat(beta, N)

        self.power_minus_half_Sigma[:] = power_minus_half_Sigma
        self.Z = Z
        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        N, D = self.X.shape
        Z = self.Z
        gamma = self.gamma
        power_minus_half_Sigma = self.power_minus_half_Sigma

        DZ = hadamard(row_repeat(gamma, N), DY)
        Dbeta = columns_sum(DY)
        Dgamma = columns_sum(hadamard(Z, DY))
        DX = hadamard(row_repeat(power_minus_half_Sigma / N, N), (N * identity(N) - ones(N, N)) * DZ - hadamard(Z, row_repeat(diag(Z.T * DZ).T, N)))

        self.DZ = DZ
        self.Dbeta[:] = Dbeta
        self.Dgamma[:] = Dgamma
        self.DX = DX
