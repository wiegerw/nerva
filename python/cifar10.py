#!/usr/bin/env python3

# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import random
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from nerva.activation import ReLU, NoActivation, AllReLU
from nerva.dataset import DataSetView
from nerva.layers import Sequential, Dense, Dropout, Sparse, BatchNormalization
from nerva.learning_rate import ConstantScheduler
from nerva.loss import SoftmaxCrossEntropyLoss
from nerva.optimizers import GradientDescent
from nerva.training import minibatch_gradient_descent, minibatch_gradient_descent_python, SGDOptions, compute_accuracy, compute_statistics
from nerva.utilities import RandomNumberGenerator, set_num_threads, StopWatch
from nerva.weights import Weights


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
    model.add(Dense(128, activation=AllReLU(-0.5), optimizer=GradientDescent(), weight_initializer=Weights.Xavier))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation=AllReLU(0.5), optimizer=GradientDescent(), weight_initializer=Weights.Xavier))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation=NoActivation(), optimizer=GradientDescent(), weight_initializer=Weights.Xavier))
    return model


def create_sparse_model(sparsity: float):
    model = Sequential()
    model.add(Sparse(128, sparsity, activation=ReLU(), optimizer=GradientDescent(), weight_initializer=Weights.Xavier))
    model.add(Sparse(64, sparsity, activation=ReLU(), optimizer=GradientDescent(), weight_initializer=Weights.Xavier))
    model.add(Sparse(10, sparsity, activation=NoActivation(), optimizer=GradientDescent(), weight_initializer=Weights.Xavier))
    return model


# Uses a data generator to generate batches.
# The parameter dataset is only used for computing statistics.
def minibatch_gradient_descent_with_augmentation(M, dataset, datagen, loss, learning_rate, epochs, batch_size, statistics=True):
    N = dataset.Xtrain.shape[1]  # the number of examples
    K = N // batch_size  # the number of batches

    compute_statistics(M, loss, dataset, batch_size, -1, statistics, 0.0)
    total_time = 0.0
    watch = StopWatch()

    for epoch in range(epochs):
        watch.reset()

        eta = learning_rate(epoch)  # update the learning rate at the start of each epoch

        for index, (Xbatch, Tbatch) in enumerate(datagen):
            if index == K:
                break
            X = flatten(Xbatch).T
            T = Tbatch.T
            Y = M.feedforward(X)
            dY = loss.gradient(Y, T) / batch_size  # pytorch uses this division
            M.backpropagate(Y, dY)
            M.optimize(eta)

        seconds = watch.seconds()
        compute_statistics(M, loss, dataset, batch_size, epoch, statistics, seconds)
        total_time += seconds

    print(f'Accuracy of the network on the {dataset.Xtest.shape[1]} test examples: {(100.0 * compute_accuracy(M, dataset.Xtest, dataset.Ttest, batch_size)):.2f}%')
    print(f'Total training time for the {epochs} epochs: {total_time:4.2f}s')


def train_dense_model(dataset):
    rng = RandomNumberGenerator(1234567)
    loss = SoftmaxCrossEntropyLoss()
    learning_rate_scheduler = ConstantScheduler(0.01)
    input_size = 3072
    batch_size = 100
    model = create_dense_model()
    M = model.compile(input_size, batch_size, rng)
    minibatch_gradient_descent_python(M, dataset, loss, learning_rate_scheduler, epochs=10, batch_size=100, shuffle=True, statistics=True)
    print('')


# Use the c++ version of minibatch_gradient_descent, which is slightly faster
def train_dense_model_cpp(dataset):
    rng = RandomNumberGenerator(1234567)
    loss = SoftmaxCrossEntropyLoss()
    learning_rate_scheduler = ConstantScheduler(0.01)
    input_size = 3072
    batch_size = 100
    model = create_dense_model()
    M = model.compile(input_size, batch_size, rng)
    options = make_sgd_options()
    minibatch_gradient_descent(M, loss, dataset, options, learning_rate_scheduler, rng)
    print('')


def train_sparse_model(dataset):
    rng = RandomNumberGenerator(1234567)
    loss = SoftmaxCrossEntropyLoss()
    learning_rate_scheduler = ConstantScheduler(0.01)
    input_size = 3072
    batch_size = 100
    sparsity = 0.5
    model = create_sparse_model(sparsity)
    M = model.compile(input_size, batch_size, rng)
    minibatch_gradient_descent_python(M, dataset, loss, learning_rate_scheduler, epochs=10, batch_size=100, shuffle=True, statistics=True)
    print('')


# Use on the fly data augmentation. Note that data augmentation is very expensive.
def train_dense_model_with_augmentation(x_train, x_test, y_train, y_test):
    rng = RandomNumberGenerator(1234567)
    loss = SoftmaxCrossEntropyLoss()
    learning_rate_scheduler = ConstantScheduler(0.01)

    # data augmentation
    image_data_generator = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    image_data_generator.fit(x_train)
    batch_size = 100
    datagen = image_data_generator.flow(x_train, y_train, batch_size=batch_size, shuffle=False)

    # flatten data
    x_train = flatten(x_train)
    x_test = flatten(x_test)

    dataset = make_dataset(x_train, y_train, x_test, y_test)

    model = create_dense_model()
    input_size = 3072
    batch_size = 100
    M = model.compile(input_size, batch_size, rng)
    minibatch_gradient_descent_with_augmentation(M, dataset, datagen, loss, learning_rate_scheduler, epochs=10, batch_size=100, statistics=True)
    print('')


def minibatch_gradient_descent_with_regrow(M, dataset, loss, learning_rate, epochs, batch_size, shuffle=True, statistics=True, zeta=0.3, regrow_weights=Weights.Zero, rng=RandomNumberGenerator(1234567), check_regrow=False):
    N = dataset.Xtrain.shape[1]  # the number of examples
    I = list(range(N))
    K = N // batch_size  # the number of batches
    compute_statistics(M, loss, dataset, batch_size, -1, statistics, 0.0)
    total_time = 0.0
    watch = StopWatch()

    for epoch in range(epochs):
        if epoch > 0:
            print('regrow')
            M.regrow(zeta, regrow_weights, rng, check_regrow)

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


def train_sparse_model_with_regrow(dataset):
    rng = RandomNumberGenerator(1234567)
    loss = SoftmaxCrossEntropyLoss()
    learning_rate_scheduler = ConstantScheduler(0.01)
    input_size = 3072
    batch_size = 100
    sparsity = 0.5
    model = create_sparse_model(sparsity)
    M = model.compile(input_size, batch_size, rng)
    minibatch_gradient_descent_with_regrow(M, dataset, loss, learning_rate_scheduler, epochs=100, batch_size=100, shuffle=True, statistics=True, zeta=0.3, regrow_weights=Weights.Xavier, rng=rng, check_regrow=False)
    print('')


if __name__ == '__main__':
    # set_num_threads(4)
    x_train, x_test, y_train, y_test = read_cifar10()
    x_train_flattened = flatten(x_train)
    x_test_flattened = flatten(x_test)
    dataset = make_dataset(x_train_flattened, y_train, x_test_flattened, y_test)

    train_dense_model(dataset)
    train_dense_model_cpp(dataset)
    train_sparse_model(dataset)
    train_sparse_model_with_regrow(dataset)
    train_dense_model_with_augmentation(x_train, x_test, y_train, y_test)
