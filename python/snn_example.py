#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

# This file contains the example code that was used in the SNN submission

import random
import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils
from nerva.activation import ReLU, NoActivation
from nerva.dataset import DataSet
from nerva.layers import Sequential, Dense, Dropout, Sparse, BatchNormalization
from nerva.learning_rate import ConstantScheduler
from nerva.loss import SoftmaxCrossEntropyLoss
from nerva.optimizers import GradientDescent, Momentum
from nerva.random import manual_seed
from nerva.weights import Xavier


def read_cifar10():
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # normalize data
    xTrainMean = np.mean(x_train, axis=0)
    xTtrainStd = np.std(x_train, axis=0)
    x_train = (x_train - xTrainMean) / xTtrainStd
    x_test = (x_test - xTrainMean) / xTtrainStd

    return x_train, x_test, y_train, y_test


def flatten(X: np.array):
    shape = X.shape
    return X.reshape(shape[0], -1)


def load_cifar10():
    x_train, x_test, y_train, y_test = read_cifar10()
    x_train_flattened = flatten(x_train)
    x_test_flattened = flatten(x_test)
    return DataSet(x_train_flattened, y_train, x_test_flattened, y_test)


def stochastic_gradient_descent(model, dataset, loss, learning_rate, epochs, batch_size, shuffle):
    N = dataset.Xtrain.shape[1]  # the number of examples
    I = list(range(N))
    K = N // batch_size  # the number of batches
    for epoch in range(epochs):
        if shuffle: random.shuffle(I)
        eta = learning_rate(epoch)  # update the learning rate at the start of each epoch
        for k in range(K):
            batch = I[k * batch_size: (k + 1) * batch_size]
            X = dataset.Xtrain[:, batch]
            T = dataset.Ttrain[:, batch]
            Y = model.feedforward(X)
            dY = loss.gradient(Y, T) / batch_size
            model.backpropagate(Y, dY)
            model.optimize(eta)


def snn_example():
    dataset = load_cifar10()
    loss = SoftmaxCrossEntropyLoss()
    learning_rate_scheduler = ConstantScheduler(0.01)
    manual_seed(1234567)
    density = 0.05

    model = Sequential()
    model.add(BatchNormalization())
    model.add(Sparse(1000, density, ReLU(), GradientDescent(), Xavier()))
    model.add(Dense(128, ReLU(), Momentum(0.9), Xavier()))
    model.add(Dense(64, ReLU(), GradientDescent(), Xavier()))
    model.add(Dropout(0.3))
    model.add(Dense(10, NoActivation(), GradientDescent(), Xavier()))

    model.compile(input_size=3072, batch_size=100)
    stochastic_gradient_descent(model, dataset, loss, learning_rate_scheduler,
                                epochs=10, batch_size=100, shuffle=True)


if __name__ == '__main__':
    snn_example()
