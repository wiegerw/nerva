# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import Tuple

from symbolic.numpy.activation_functions import *
from symbolic.numpy.optimizers import parse_optimizer
from symbolic.numpy.softmax_functions import *
from symbolic.numpy.weight_initializers import set_weights

Matrix = np.ndarray


class Layer(object):
    """
    Base class for layers of a neural network with data in column layout
    """
    def __init__(self, m: int, n: int):
        self.X = zeros(m, n)
        self.DX = zeros(m, n)

    def feedforward(self, X: Matrix) -> Matrix:
        raise NotImplementedError

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        raise NotImplementedError

    def optimize(self, eta):
        raise NotImplementedError


class LinearLayer(Layer):
    """
    Linear layer of a neural network
    """
    def __init__(self, D: int, K: int, N: int):
        super().__init__(D, N)
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

        Y = W @ X + column_repeat(b, N)

        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        X = self.X
        W = self.W

        DW = DY @ X.T
        Db = rows_sum(DY)
        DX = W.T @ DY

        self.DW[:] = DW
        self.Db[:] = Db
        self.DX[:] = DX

    def optimize(self, eta):
        assert id(self.W) == id(self.optimizer.W)
        assert id(self.b) == id(self.optimizer.b)
        self.optimizer.update(eta)

    def input_output_sizes(self) -> Tuple[int, int]:
        """
        Returns the input and output sizes of the layer
        """
        K, D = self.W.shape
        return D, K


class ActivationLayer(LinearLayer):
    """
    Linear layer with an activation function
    """
    def __init__(self, D: int, K: int, N: int, act: ActivationFunction):
        super().__init__(D, K, N)
        self.Z = zeros(K, N)
        self.DZ = zeros(K, N)
        self.act = act

    def feedforward(self, X: Matrix) -> Matrix:
        self.X = X
        D, N = X.shape
        W = self.W
        b = self.b
        act = self.act

        Z = W @ X + column_repeat(b, N)
        Y = act(Z)

        self.Z[:] = Z
        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        X = self.X
        W = self.W
        Z = self.Z
        act = self.act

        DZ = hadamard(DY, act.gradient(Z))
        DW = DZ @ X.T
        Db = rows_sum(DZ)
        DX = W.T @ DZ

        self.DZ[:] = DZ
        self.DW[:] = DW
        self.Db[:] = Db
        self.DX[:] = DX


class SigmoidLayer(LinearLayer):
    """
    Linear layer with a sigmoid activation function. This is not strictly needed,
    but it shows that the backpropagation can be calculated in a different way
    """
    def __init__(self, D: int, K: int, N: int):
        super().__init__(D, K, N)
        self.Z = zeros(K, N)
        self.DZ = zeros(K, N)

    def feedforward(self, X: Matrix) -> Matrix:
        self.X = X
        D, N = X.shape
        W = self.W
        b = self.b

        Z = W @ X + column_repeat(b, N)
        Y = Sigmoid(Z)

        self.Z[:] = Z
        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        K, N = self.Z.shape
        X = self.X
        W = self.W

        DZ = hadamard(DY, hadamard(Y, ones(K, N) - Y))
        DW = DZ @ X.T
        Db = rows_sum(DZ)
        DX = W.T @ DZ

        self.DZ[:] = DZ
        self.DW[:] = DW
        self.Db[:] = Db
        self.DX[:] = DX


class SReLULayer(ActivationLayer):
    """
    Linear layer with an SReLU activation function. It adds learning of the parameters
    al, tl, ar and tr.
    """
    def __init__(self, D: int, K: int, N: int, act: SReLUActivation):
        super().__init__(D, K, N, act)
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

        Al = lambda Z: np.where(Z <= tl, Z - tl, 0)
        Ar = lambda Z: np.where((Z <= tl) | (Z < tr), 0, Z - tr)
        Tl = lambda Z: np.where(Z <= tl, 1 - al, 0)
        Tr = lambda Z: np.where((Z <= tl) | (Z < tr), 0, 1 - ar)

        self.Dal = elements_sum(hadamard(DY, Al(Z)))
        self.Dar = elements_sum(hadamard(DY, Ar(Z)))
        self.Dtl = elements_sum(hadamard(DY, Tl(Z)))
        self.Dtr = elements_sum(hadamard(DY, Tr(Z)))


class SoftmaxLayer(LinearLayer):
    """
    Linear layer with a softmax activation function
    """
    def __init__(self, D: int, K: int, N: int):
        super().__init__(D, K, N)
        self.Z = zeros(K, N)
        self.DZ = zeros(K, N)

    def feedforward(self, X: Matrix) -> Matrix:
        self.X = X
        D, N = X.shape
        W = self.W
        b = self.b

        Z = W @ X + column_repeat(b, N)
        Y = softmax_colwise(Z)

        self.Z[:] = Z
        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        K, N = self.Z.shape
        X = self.X
        W = self.W

        DZ = hadamard(Y, DY - row_repeat(diag(Y.T @ DY).T, K))
        DW = DZ @ X.T
        Db = rows_sum(DZ)
        DX = W.T @ DZ

        self.DZ[:] = DZ
        self.DW[:] = DW
        self.Db[:] = Db
        self.DX[:] = DX


class LogSoftmaxLayer(LinearLayer):
    """
    Linear layer with a log_softmax activation function
    """
    def __init__(self, D: int, K: int, N: int):
        super().__init__(D, K, N)
        self.Z = zeros(K, N)
        self.DZ = zeros(K, N)

    def feedforward(self, X: Matrix) -> Matrix:
        self.X = X
        D, N = X.shape
        W = self.W
        b = self.b

        Z = W @ X + column_repeat(b, N)
        Y = log_softmax_colwise(Z)

        self.Z[:] = Z
        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        K, N = self.Z.shape
        X = self.X
        W = self.W
        Z = self.Z

        DZ = DY - hadamard(softmax_colwise(Z), row_repeat(columns_sum(DY), K))
        DW = DZ @ X.T
        Db = rows_sum(DZ)
        DX = W.T @ DZ

        self.DZ[:] = DZ
        self.DW[:] = DW
        self.Db[:] = Db
        self.DX[:] = DX


class BatchNormalizationLayer(Layer):
    """
    A batch normalization layer
    """
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
        self.X = X
        D, N = X.shape
        gamma = self.gamma
        beta = self.beta

        R = X - column_repeat(rows_mean(X), N)
        Sigma = diag(R @ R.T) / N
        power_minus_half_Sigma = power_minus_half(Sigma)
        Z = hadamard(column_repeat(power_minus_half_Sigma, N), R)
        Y = hadamard(column_repeat(gamma, N), Z) + column_repeat(beta, N)

        self.power_minus_half_Sigma[:] = power_minus_half_Sigma
        self.Z[:] = Z
        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        D, N = self.X.shape
        Z = self.Z
        gamma = self.gamma
        power_minus_half_Sigma = self.power_minus_half_Sigma

        DZ = hadamard(column_repeat(gamma, N), DY)
        Dbeta = rows_sum(DY)
        Dgamma = rows_sum(hadamard(DY, Z))
        DX = hadamard(column_repeat(power_minus_half_Sigma / N, N), hadamard(Z, column_repeat(-diag(DZ @ Z.T), N)) + DZ @ (N * identity(N) - ones(N, N)))

        self.DZ[:] = DZ
        self.Dbeta[:] = Dbeta
        self.Dgamma[:] = Dgamma
        self.DX[:] = DX

    def optimize(self, eta):
        # use gradient descent; TODO: generalize this
        self.beta -= eta * self.Dbeta
        self.gamma -= eta * self.Dgamma


def parse_linear_layer(text: str,
                       D: int,
                       K: int,
                       N: int,
                       optimizer: str,
                       weight_initializer: str
                      ) -> Layer:
    if text == 'Linear':
        layer = LinearLayer(D, K, N)
    elif text == 'Sigmoid':
        layer = SigmoidLayer(D, K, N)
    elif text == 'Softmax':
        layer = SoftmaxLayer(D, K, N)
    elif text == 'LogSoftmax':
        layer = LogSoftmaxLayer(D, K, N)
    elif text.startswith('SReLU'):
        act = parse_srelu_activation(text)
        layer = SReLULayer(D, K, N, act)
    else:
        act = parse_activation(text)
        layer = ActivationLayer(D, K, N, act)
    layer.optimizer = parse_optimizer(optimizer, layer)
    set_weights(layer, weight_initializer)
    return layer
