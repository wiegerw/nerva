# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import Tuple

from symbolic.tensorflow.activation_functions import *
from symbolic.tensorflow.optimizers import parse_optimizer
from symbolic.tensorflow.softmax_functions import *
from symbolic.tensorflow.weight_initializers import set_layer_weights

Matrix = tf.Tensor


class Layer(object):
    """
    Base class for layers of a neural network with data in column layout
    """
    def __init__(self, m: int, n: int):
        self.X = tf.Variable(zeros(m, n))
        self.DX = tf.Variable(zeros(m, n))

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
        self.W = tf.Variable(zeros(K, D))
        self.DW = tf.Variable(zeros(K, D))
        self.b = tf.Variable(zeros(K, 1))
        self.Db = tf.Variable(zeros(K, 1))
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

        DW = DY @ tf.transpose(X)
        Db = rows_sum(DY)
        DX = tf.transpose(W) @ DY

        self.DW.assign(DW)
        self.Db.assign(Db)
        self.DX.assign(DX)

    def optimize(self, eta):
        self.optimizer.update(eta)

    def input_output_sizes(self) -> Tuple[int, int]:
        """
        Returns the input and output sizes of the layer
        """
        K, D = self.W.shape
        return D, K

    def set_weights(self, weight_initializer):
        set_layer_weights(self, weight_initializer)


class ActivationLayer(LinearLayer):
    """
    Linear layer with an activation function
    """
    def __init__(self, D: int, K: int, N: int, act: ActivationFunction):
        super().__init__(D, K, N)
        self.Z = tf.Variable(zeros(K, N))
        self.DZ = tf.Variable(zeros(K, N))
        self.act = act

    def feedforward(self, X: Matrix) -> Matrix:
        self.X = X
        D, N = X.shape
        W = self.W
        b = self.b
        act = self.act

        Z = W @ X + column_repeat(b, N)
        Y = act(Z)

        self.Z.assign(Z)
        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        X = self.X
        W = self.W
        Z = self.Z
        act = self.act

        DZ = hadamard(DY, act.gradient(Z))
        DW = DZ @ tf.transpose(X)
        Db = rows_sum(DZ)
        DX = tf.transpose(W) @ DZ

        self.DZ.assign(DZ)
        self.DW.assign(DW)
        self.Db.assign(Db)
        self.DX.assign(DX)


class SigmoidLayer(LinearLayer):
    """
    Linear layer with a sigmoid activation function. This is not strictly needed,
    but it shows that the backpropagation can be calculated in a different way
    """
    def __init__(self, D: int, K: int, N: int):
        super().__init__(D, K, N)
        self.Z = tf.Variable(zeros(K, N))
        self.DZ = tf.Variable(zeros(K, N))

    def feedforward(self, X: Matrix) -> Matrix:
        self.X = X
        D, N = X.shape
        W = self.W
        b = self.b

        Z = W @ X + column_repeat(b, N)
        Y = Sigmoid(Z)

        self.Z.assign(Z)
        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        K, N = self.Z.shape
        X = self.X
        W = self.W

        DZ = hadamard(DY, hadamard(Y, ones(K, N) - Y))
        DW = DZ @ tf.transpose(X)
        Db = rows_sum(DZ)
        DX = tf.transpose(W) @ DZ

        self.DZ.assign(DZ)
        self.DW.assign(DW)
        self.Db.assign(Db)
        self.DX.assign(DX)


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

        Al = lambda Z: tf.where(Z <= tl, Z - tl, 0)
        Ar = lambda Z: tf.where((Z <= tl) | (Z < tr), 0, Z - tr)
        Tl = lambda Z: tf.where(Z <= tl, 1 - al, 0)
        Tr = lambda Z: tf.where((Z <= tl) | (Z < tr), 0, 1 - ar)

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
        self.Z = tf.Variable(zeros(K, N))
        self.DZ = tf.Variable(zeros(K, N))

    def feedforward(self, X: Matrix) -> Matrix:
        self.X = X
        D, N = X.shape
        W = self.W
        b = self.b

        Z = W @ X + column_repeat(b, N)
        Y = softmax_colwise(Z)

        self.Z.assign(Z)
        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        K, N = self.Z.shape
        X = self.X
        W = self.W

        DZ = hadamard(Y, DY - row_repeat(tf.transpose(diag(tf.transpose(Y) @ DY)), K))
        DW = DZ @ tf.transpose(X)
        Db = rows_sum(DZ)
        DX = tf.transpose(W) @ DZ

        self.DZ.assign(DZ)
        self.DW.assign(DW)
        self.Db.assign(Db)
        self.DX.assign(DX)


class LogSoftmaxLayer(LinearLayer):
    """
    Linear layer with a log_softmax activation function
    """
    def __init__(self, D: int, K: int, N: int):
        super().__init__(D, K, N)
        self.Z = tf.Variable(zeros(K, N))
        self.DZ = tf.Variable(zeros(K, N))

    def feedforward(self, X: Matrix) -> Matrix:
        self.X = X
        D, N = X.shape
        W = self.W
        b = self.b

        Z = W @ X + column_repeat(b, N)
        Y = log_softmax_colwise(Z)

        self.Z.assign(Z)
        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        K, N = self.Z.shape
        X = self.X
        W = self.W
        Z = self.Z

        DZ = DY - hadamard(softmax_colwise(Z), row_repeat(columns_sum(DY), K))
        DW = DZ @ tf.transpose(X)
        Db = rows_sum(DZ)
        DX = tf.transpose(W) @ DZ

        self.DZ.assign(DZ)
        self.DW.assign(DW)
        self.Db.assign(Db)
        self.DX.assign(DX)


class BatchNormalizationLayer(Layer):
    """
    A batch normalization layer
    """
    def __init__(self, D: int, N: int):
        super().__init__(D, N)
        self.Z = tf.Variable(zeros(D, N))
        self.DZ = tf.Variable(zeros(D, N))
        self.gamma = tf.Variable(ones(D, 1))
        self.Dgamma = tf.Variable(zeros(D, 1))
        self.beta = tf.Variable(zeros(D, 1))
        self.Dbeta = tf.Variable(zeros(D, 1))
        self.power_minus_half_Sigma = tf.Variable(zeros(D, 1))

    def feedforward(self, X: Matrix) -> Matrix:
        self.X = X
        D, N = X.shape
        gamma = self.gamma
        beta = self.beta

        R = X - column_repeat(rows_mean(X), N)
        Sigma = diag(R @ tf.transpose(R)) / N
        power_minus_half_Sigma = power_minus_half(Sigma)
        Z = hadamard(column_repeat(power_minus_half_Sigma, N), R)
        Y = hadamard(column_repeat(gamma, N), Z) + column_repeat(beta, N)

        self.power_minus_half_Sigma.assign(power_minus_half_Sigma)
        self.Z.assign(Z)
        return Y

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        D, N = self.X.shape
        Z = self.Z
        gamma = self.gamma
        power_minus_half_Sigma = self.power_minus_half_Sigma

        DZ = hadamard(column_repeat(gamma, N), DY)
        Dbeta = rows_sum(DY)
        Dgamma = rows_sum(hadamard(DY, Z))
        DX = hadamard(column_repeat(power_minus_half_Sigma / N, N), hadamard(Z, column_repeat(-diag(DZ @ tf.transpose(Z)), N)) + DZ @ (N * identity(N) - ones(N, N)))

        self.DZ.assign(DZ)
        self.Dbeta.assign(Dbeta)
        self.Dgamma.assign(Dgamma)
        self.DX.assign(DX)

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
    layer.set_weights(weight_initializer)
    return layer
