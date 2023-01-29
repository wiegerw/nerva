#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import math
import random
import numpy as np
from nerva.activation import ReLU, NoActivation, AllReLU
from nerva.dataset import DataSet
from nerva.layers import Sequential, Dense, Dropout, Sparse, BatchNormalization
from nerva.learning_rate import ConstantScheduler
from nerva.loss import SoftmaxCrossEntropyLoss, SquaredErrorLoss
from nerva.optimizers import GradientDescent
from nerva.random import manual_seed
from nerva.training import minibatch_gradient_descent, minibatch_gradient_descent_python, SGDOptions, compute_accuracy, compute_statistics
from nerva.utilities import set_num_threads, StopWatch
from nerva.weights import Xavier
from regrow import regrow_weights


def make_dataset_chessboard(n: int):
    X = np.zeros(shape=(2, n), dtype=np.float32, order='F')
    T = np.zeros(shape=(2, n), dtype=np.float32, order='F')
    N = 8

    for i in range(n):
        x = random.uniform(0.0, 1.0)
        y = random.uniform(0.0, 1.0)
        col = math.floor(x * N)
        row = math.floor(y * N)
        is_dark = (row + col) % 2 == 0
        X[0, i] = x
        X[1, i] = y
        T[0, i] = 1 if is_dark else 0
        T[1, i] = 0 if is_dark else 1

    return X, T


def create_model(density: float):
    model = Sequential()
    if density == 0:
        model.add(Dense(64, activation=ReLU(), optimizer=GradientDescent(), weight_initializer=Xavier()))
        model.add(Dense(16, activation=ReLU(), optimizer=GradientDescent(), weight_initializer=Xavier()))
        model.add(Dense(2, activation=NoActivation(), optimizer=GradientDescent(), weight_initializer=Xavier()))
    else:
        model.add(Sparse(32, density, activation=ReLU(), optimizer=GradientDescent(), weight_initializer=Xavier()))
        model.add(Sparse(8, density, activation=ReLU(), optimizer=GradientDescent(), weight_initializer=Xavier()))
        model.add(Sparse(2, density, activation=NoActivation(), optimizer=GradientDescent(), weight_initializer=Xavier()))
    return model


def minibatch_gradient_descent_with_regrow(model, dataset, loss, learning_rate, epochs, batch_size, shuffle=True, statistics=True, zeta=0.3, weights_initializer=Xavier()):
    M = model.compiled_model
    N = dataset.Xtrain.shape[1]  # the number of examples
    I = list(range(N))
    K = N // batch_size  # the number of batches
    compute_statistics(M, loss, dataset, batch_size, -1, statistics, 0.0)
    total_time = 0.0
    watch = StopWatch()

    for epoch in range(epochs):
        if epoch > 0:
            regrow_weights(M, zeta)

        watch.reset()
        if shuffle:
            random.shuffle(I)

        eta = learning_rate(epoch)  # update the learning rate at the start of each epoch

        for k in range(K):
            batch = I[k * batch_size: (k + 1) * batch_size]
            X = dataset.Xtrain[:, batch]
            T = dataset.Ttrain[:, batch]
            Y = M.feedforward(X)
            dY = loss.gradient(Y, T) / batch_size  # pytorch uses this division
            M.backpropagate(Y, dY)
            M.optimize(eta)

        seconds = watch.seconds()
        compute_statistics(M, loss, dataset, batch_size, epoch, statistics, seconds)
        total_time += seconds

    print(f'Accuracy of the network on the {dataset.Xtest.shape[1]} test examples: {(100.0 * compute_accuracy(M, dataset.Xtest, dataset.Ttest, batch_size)):.2f}%')
    print(f'Total training time for the {epochs} epochs: {total_time:4.2f}s')


def train_sparse_model_with_regrow(dataset, density):
    seed = random.randint(0, 999999999)
    print('seed', seed)
    loss = SoftmaxCrossEntropyLoss()
    learning_rate_scheduler = ConstantScheduler(0.1)
    input_size = 2
    batch_size = 100
    model = create_model(density)
    model.compile(input_size, batch_size)
    minibatch_gradient_descent_with_regrow(model, dataset, loss, learning_rate_scheduler, epochs=100, batch_size=batch_size, shuffle=True, statistics=True, zeta=0.1, weights_initializer=Xavier())
    print('')


def plot_dataset(X, T):
    import matplotlib.pyplot as plt

    rows, columns = X.shape
    x = [X[0][i] for i in range(columns)]
    y = [X[1][i] for i in range(columns)]
    c = ['lightblue' if T[0][i] == 1 else 'green' for i in range(columns)]
    s = 2
    plt.scatter(x, y, c=c, s=s)
    plt.show()


if __name__ == '__main__':
    n = 100000
    manual_seed(317822)
    Xtrain, Ttrain = make_dataset_chessboard(n)
    Xtest, Ttest = make_dataset_chessboard(n // 5)
    # plot_dataset(Xtrain, Ttrain)
    # plot_dataset(Xtest, Ttest)
    dataset = DataSet(Xtrain.T, Ttrain.T, Xtest.T, Ttest.T)
    density = 0.8
    train_sparse_model_with_regrow(dataset, density)
