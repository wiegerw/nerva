#!/usr/bin/env python3

# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils
from nerva.activation import ReLU, NoActivation, AllReLU
from nerva.dataset import DataSetView
from nerva.layers import Sequential, Dense, Dropout, Sparse, BatchNormalization
from nerva.learning_rate import ConstantScheduler
from nerva.loss import SoftmaxCrossEntropyLoss
from nerva.optimizers import GradientDescent
from nerva.training import minibatch_gradient_descent, minibatch_gradient_descent_python, SGDOptions
from nerva.utilities import RandomNumberGenerator
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


def make_dataset(x_train, y_train, x_test, y_test) -> DataSetView:
    return DataSetView(x_train.T, y_train.T, x_test.T, y_test.T)  # N.B. the data sets must be transposed!


def make_sgd_options():
    options = SGDOptions()
    options.epochs = 10
    options.batch_size = 100
    options.shuffle = True
    options.statistics = True
    options.debug = False
    return options


def create_dense_model():
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Dense(128, activation=AllReLU(-0.5), optimizer=GradientDescent(), weight_initializer=Xavier()))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation=AllReLU(0.5), optimizer=GradientDescent(), weight_initializer=Xavier()))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation=NoActivation(), optimizer=GradientDescent(), weight_initializer=Xavier()))
    return model


def create_sparse_model(sparsity: float):
    model = Sequential()
    model.add(Sparse(128, sparsity, activation=ReLU(), optimizer=GradientDescent(), weight_initializer=Xavier()))
    model.add(Sparse(64, sparsity, activation=ReLU(), optimizer=GradientDescent(), weight_initializer=Xavier()))
    model.add(Sparse(10, sparsity, activation=NoActivation(), optimizer=GradientDescent(), weight_initializer=Xavier()))
    return model


def main():
    x_train, x_test, y_train, y_test = read_cifar10()

    # flatten data
    x_train = flatten(x_train)
    x_test = flatten(x_test)

    dataset = make_dataset(x_train, y_train, x_test, y_test)

    rng = RandomNumberGenerator(1234567)
    loss = SoftmaxCrossEntropyLoss()
    learning_rate_scheduler = ConstantScheduler(0.01)

    # train dense model
    model = create_dense_model()
    input_size = 3072
    batch_size = 100
    M = model.compile(input_size, batch_size, rng)
    minibatch_gradient_descent_python(M, dataset, loss, learning_rate_scheduler, epochs=10, batch_size=100, shuffle=True, statistics=True)
    print('')

    # train sparse model
    sparsity = 0.5
    model = create_sparse_model(sparsity)
    M = model.compile(input_size, batch_size, rng)
    minibatch_gradient_descent_python(M, dataset, loss, learning_rate_scheduler, epochs=10, batch_size=100, shuffle=True, statistics=True)
    print('')

    # train dense model using the c++ version of minibatch gradient descent (slightly faster)
    model = create_dense_model()
    M = model.compile(input_size, batch_size, rng)
    options = make_sgd_options()
    minibatch_gradient_descent(M, loss, dataset, options, learning_rate_scheduler, rng)


if __name__ == '__main__':
    main()
