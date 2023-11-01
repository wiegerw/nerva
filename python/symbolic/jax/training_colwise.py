#!/usr/bin/env python3

# Copyright 2022 - 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import List
from symbolic.learning_rate import parse_learning_rate, LearningRateScheduler
from symbolic.jax.datasets import DataLoader, create_npz_dataloaders
from symbolic.jax.loss_functions_colwise import *
from symbolic.jax.multilayer_perceptron_colwise import MultilayerPerceptron
from symbolic.jax.parse_mlp_colwise import parse_multilayer_perceptron, parse_loss_function
from symbolic.jax.utilities import pp, set_numpy_options
from symbolic.training import SGDOptions, print_epoch
from symbolic.utilities import StopWatch


def compute_accuracy(M: MultilayerPerceptron, data_loader: DataLoader):
    N = len(data_loader.dataset)  # N is the number of examples
    total_correct = 0
    for X, T in data_loader:
        Y = M.feedforward(X)
        predicted = Y.argmax(axis=0)  # the predicted classes for the batch
        targets = T.argmax(axis=0)    # the expected classes
        total_correct += (predicted == targets).sum().item()

    return total_correct / N


def compute_loss(M: MultilayerPerceptron, data_loader: DataLoader, loss: LossFunction):
    N = len(data_loader.dataset)  # N is the number of examples
    total_loss = 0.0
    for X, T in data_loader:
        Y = M.feedforward(X)
        total_loss += loss(Y, T)

    return total_loss / N


def compute_statistics(M, lr, loss, train_loader, test_loader, epoch, elapsed_seconds=0.0, print_statistics=True):
    if print_statistics:
        train_loss = compute_loss(M, train_loader, loss)
        train_accuracy = compute_accuracy(M, train_loader)
        test_accuracy = compute_accuracy(M, test_loader)
        print_epoch(epoch, lr, train_loss, train_accuracy, test_accuracy, elapsed_seconds)
    else:
        print(f'epoch {epoch:3}')


def sgd(M: MultilayerPerceptron,
        epochs: int,
        loss: LossFunction,
        learning_rate: LearningRateScheduler,
        train_loader: DataLoader,
        test_loader: DataLoader
       ):

    lr = learning_rate(0)
    compute_statistics(M, lr, loss, train_loader, test_loader, epoch=0)
    training_time = 0.0

    for epoch in range(epochs):
        timer = StopWatch()
        lr = learning_rate(epoch)  # update the learning at the start of each epoch

        for k, (X, T) in enumerate(train_loader):
            Y = M.feedforward(X)
            DY = loss.gradient(Y, T) / Y.shape[1]

            if SGDOptions.debug:
                print(f'epoch: {epoch} batch: {k}')
                M.info()
                pp("X", X.T)
                pp("Y", Y.T)
                pp("DY", DY.T)

            M.backpropagate(Y, DY)
            M.optimize(lr)

        seconds = timer.seconds()
        training_time += seconds
        compute_statistics(M, lr, loss, train_loader, test_loader, epoch=epoch + 1, elapsed_seconds=seconds)

    print(f'Total training time for the {epochs} epochs: {training_time:.8f}s\n')


def train(layer_specifications: List[str],
          linear_layer_sizes: List[int],
          linear_layer_optimizers: List[str],
          linear_layer_weight_initializers: List[str],
          batch_size: int,
          epochs: int,
          loss: str,
          learning_rate: str,
          weights_and_bias_file: str,
          dataset_file: str,
          debug: bool
         ):
    SGDOptions.debug = debug
    set_numpy_options()
    loss = parse_loss_function(loss)
    learning_rate = parse_learning_rate(learning_rate)
    M = parse_multilayer_perceptron(layer_specifications, linear_layer_sizes, linear_layer_optimizers, linear_layer_weight_initializers)
    M.load_weights_and_bias(weights_and_bias_file)
    train_loader, test_loader = create_npz_dataloaders(dataset_file, batch_size=batch_size, rowwise=False)
    sgd(M, epochs, loss, learning_rate, train_loader, test_loader)


if __name__ == '__main__':
    layer_specifications = ['ReLU', 'ReLU', 'Linear']
    linear_layer_sizes = [3072, 1024, 512, 10]
    linear_layer_optimizers = ['Momentum(0.9)', 'Momentum(0.9)', 'Momentum(0.9)']
    linear_layer_weight_initializers = ['Xavier', 'Xavier', 'Xavier']
    batch_size = 100
    epochs = 1
    loss = 'SoftmaxCrossEntropy'
    learning_rate = 'Constant(0.01)'
    weights_and_bias_file = '../../mlp-compare.npz'
    dataset_file = '../../cifar1/epoch0.npz'
    debug = False

    train(layer_specifications,
          linear_layer_sizes,
          linear_layer_optimizers,
          linear_layer_weight_initializers,
          batch_size,
          epochs,
          loss,
          learning_rate,
          weights_and_bias_file,
          dataset_file,
          debug
         )
