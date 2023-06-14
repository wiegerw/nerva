# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from activation_functions_sympy import *
from matrix_operations_sympy import *
from optimizers_sympy import *
from softmax_sympy import *


class LayerColwise(object):
    def __init__(self, D: int, N: int):
        self.X = zeros(D, N)
        self.DX = zeros(D, N)

    def feedforward(self, X: Matrix) -> Matrix:
        raise NotImplementedError

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        raise NotImplementedError

    def optimize(self, eta):
        raise NotImplementedError


class LinearLayerColwise(LayerColwise):
    def __init__(self, D: int, K: int, N: int, optimizer: Optimizer):
        super().__init__(D, N)
        self.W = zeros(K, D)
        self.DW = zeros(K, D)
        self.b = zeros(K, 1)
        self.Db = zeros(K, 1)
        self.optimizer = optimizer

    def feedforward(self, X: Matrix) -> Matrix:
        D, N = self.X.shape
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

        self.DW = DW
        self.Db = Db
        self.DX = DX

    def optimize(self, eta):
        self.optimizer.update(eta)


class ActivationLayerColwise(LinearLayerColwise):
    def __init__(self, D: int, K: int, N: int, act: ActivationFunction, optimizer: Optimizer):
        super().__init__(D, K, N, optimizer)
        self.Z = zeros(K, N)
        self.DZ = zeros(K, N)
        self.act = act

    def feedforward(self, X: Matrix) -> Matrix:
        D, N = self.X.shape
        W = self.W
        b = self.b
        act = self.act

        Z = W * X + column_repeat(b, N)
        Y = act(Z)

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
        self.DW = DW
        self.Db = Db
        self.DX = DX


class SigmoidLayerColwise(LinearLayerColwise):
    def __init__(self, D: int, K: int, N: int, optimizer: Optimizer):
        super().__init__(D, K, N, optimizer)
        self.Z = zeros(K, N)
        self.DZ = zeros(K, N)

    def feedforward(self, X: Matrix) -> Matrix:
        D, N = self.X.shape
        W = self.W
        b = self.b

        Z = W * X + column_repeat(b, N)
        Y = Sigmoid(Z)

        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        K, N = self.Z.shape
        X = self.X
        W = self.W

        DZ = hadamard(DY, hadamard(Y, ones(K, N) - Y))
        DW = DZ * X.T
        Db = rows_sum(DZ)
        DX = W.T * DZ

        self.DZ = DZ
        self.DW = DW
        self.Db = Db
        self.DX = DX


class SReLULayerColwise(ActivationLayerColwise):
    def __init__(self, D: int, K: int, N: int, act: SReLUActivation, optimizer: Optimizer):
        super().__init__(D, K, N, act, optimizer)
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

        Zij = sp.symbols('Zij')
        Al = Lambda(Zij, Piecewise((Zij - tl, Zij <= tl), (0, True)))
        Ar = Lambda(Zij, Piecewise((0, Zij <= tl), (0, Zij < tr), (Zij - tr, True)))
        Tl = Lambda(Zij, Piecewise((1 - al, Zij <= tl), (0, True)))
        Tr = Lambda(Zij, Piecewise((0, Zij <= tl), (0, Zij < tr), (1 - ar, True)))

        self.Dal = elements_sum(hadamard(DY, apply(Al, Z)))
        self.Dar = elements_sum(hadamard(DY, apply(Ar, Z)))
        self.Dtl = elements_sum(hadamard(DY, apply(Tl, Z)))
        self.Dtr = elements_sum(hadamard(DY, apply(Tr, Z)))


class SoftmaxLayerColwise(LinearLayerColwise):
    def __init__(self, D: int, K: int, N: int, optimizer: Optimizer):
        super().__init__(D, K, N, optimizer)
        self.Z = zeros(K, N)
        self.DZ = zeros(K, N)

    def feedforward(self, X: Matrix) -> Matrix:
        D, N = self.X.shape
        W = self.W
        b = self.b

        Z = W * X + column_repeat(b, N)
        Y = softmax_colwise(Z)

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
        self.DW = DW
        self.Db = Db
        self.DX = DX


class LogSoftmaxLayerColwise(LinearLayerColwise):
    def __init__(self, D: int, K: int, N: int, optimizer: Optimizer):
        super().__init__(D, K, N, optimizer)
        self.Z = zeros(K, N)
        self.DZ = zeros(K, N)

    def feedforward(self, X: Matrix) -> Matrix:
        D, N = self.X.shape
        W = self.W
        b = self.b

        Z = W * X + column_repeat(b, N)
        Y = log_softmax_colwise(Z)

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
        self.DW = DW
        self.Db = Db
        self.DX = DX


class BatchNormalizationLayerColwise(LayerColwise):
    def __init__(self, D: int, N: int):
        super().__init__(D, N)
        self.Z = zeros(D, N)
        self.DZ = zeros(D, N)
        self.gamma = ones(D, 1)
        self.Dgamma = zeros(D, 1)
        self.beta = zeros(D, 1)
        self.Dbeta = zeros(D, 1)
        self.power_minus_half_Sigma = zeros(D, 1)

    def feedforward(self, X: Matrix) -> Matrix:
        D, N = self.X.shape
        gamma = self.gamma
        beta = self.beta

        R = X - column_repeat(rows_mean(X), N)
        Sigma = diag(R * R.T) / N
        power_minus_half_Sigma = power_minus_half(Sigma)
        Z = hadamard(column_repeat(power_minus_half_Sigma, N), R)
        Y = hadamard(column_repeat(gamma, N), Z) + column_repeat(beta, N)

        self.power_minus_half_Sigma = power_minus_half_Sigma

        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        D, N = self.X.shape
        Z = self.Z
        gamma = self.gamma
        power_minus_half_Sigma = self.power_minus_half_Sigma

        DZ = hadamard(column_repeat(gamma, N), DY)
        Dbeta = rows_sum(DY)
        Dgamma = rows_sum(hadamard(DY, Z))
        DX = hadamard(column_repeat(power_minus_half_Sigma / N, N), hadamard(Z, column_repeat(-diag(DZ * Z.T), N)) + DZ * (N * identity(N) - ones(N, N)))

        self.DZ = DZ
        self.Dbeta = Dbeta
        self.Dgamma = Dgamma
        self.DX = DX

    def optimize(self, eta):
        # use gradient descent; TODO: generalize this
        self.beta -= eta * self.Dbeta
        self.gamma -= eta * self.Dgamma
