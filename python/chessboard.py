#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse
import math
import random
import sys

import numpy as np
import torch
from nerva.activation import ReLU, NoActivation
from nerva.dataset import DataSet
from nerva.layers import Sequential, Dense, Sparse
from nerva.learning_rate import ConstantScheduler
from nerva.loss import SquaredErrorLoss
from nerva.optimizers import GradientDescent
from nerva.random import manual_seed
from nerva.training import compute_accuracy, compute_statistics, compute_densities
from nerva.utilities import StopWatch
from nerva.weights import Xavier


def make_dataset_chessboard(n: int):
    X = np.zeros(shape=(n, 2), dtype=np.float32)
    T = np.zeros(shape=(n, 2), dtype=np.float32)
    N = 8

    for i in range(n):
        x = random.uniform(0.0, 1.0)
        y = random.uniform(0.0, 1.0)
        col = math.floor(x * N)
        row = math.floor(y * N)
        is_dark = (row + col) % 2 == 0
        X[i, 0] = 2 * x - 1  # normalize to the interval [-1, 1]
        X[i, 1] = 2 * y - 1  # normalize to the interval [-1, 1]
        T[i, 0] = 1 if is_dark else 0
        T[i, 1] = 0 if is_dark else 1

    return X, T


def stochastic_gradient_descent(model, dataset, loss, learning_rate, epochs, batch_size, shuffle=True, statistics=True):
    M = model.compiled_model
    N = dataset.Xtrain.shape[1]  # the number of examples
    I = list(range(N))
    K = N // batch_size  # the number of batches
    compute_statistics(M, learning_rate(0), loss, dataset, batch_size, -1, statistics, 0.0)
    total_time = 0.0
    watch = StopWatch()

    for epoch in range(epochs):
        watch.reset()
        if shuffle:
            random.shuffle(I)

        eta = learning_rate(epoch)  # update the learning rate at the start of each epoch

        for k in range(K):
            batch = I[k * batch_size: (k + 1) * batch_size]
            X = dataset.Xtrain[:, batch]
            T = dataset.Ttrain[:, batch]
            Y = M.feedforward(X)
            DY = loss.gradient(Y, T) / batch_size  # pytorch uses this division
            M.backpropagate(Y, DY)
            M.optimize(eta)

        seconds = watch.seconds()
        compute_statistics(M, eta, loss, dataset, batch_size, epoch, statistics, seconds)
        total_time += seconds

    print(f'Accuracy of the network on the {dataset.Xtest.shape[1]} test examples: {(100.0 * compute_accuracy(M, dataset.Xtest, dataset.Ttest, batch_size)):.2f}%')
    print(f'Total training time for the {epochs} epochs: {total_time:4.2f}s')


def create_model(overall_density: float):
    model = Sequential()
    layer_sizes = [2, 64, 64, 2]
    output_sizes = layer_sizes[1:]
    if overall_density == 1.0:
        model.add(Dense(output_sizes[0], activation=ReLU(), optimizer=GradientDescent(), weight_initializer=Xavier()))
        model.add(Dense(output_sizes[1], activation=ReLU(), optimizer=GradientDescent(), weight_initializer=Xavier()))
        model.add(Dense(output_sizes[2], activation=NoActivation(), optimizer=GradientDescent(), weight_initializer=Xavier()))
    else:
        densities = compute_densities(overall_density, layer_sizes)
        model.add(Sparse(output_sizes[0], densities[0], activation=ReLU(), optimizer=GradientDescent(), weight_initializer=Xavier()))
        model.add(Sparse(output_sizes[1], densities[1], activation=ReLU(), optimizer=GradientDescent(), weight_initializer=Xavier()))
        model.add(Sparse(output_sizes[2], densities[2], activation=NoActivation(), optimizer=GradientDescent(), weight_initializer=Xavier()))
    return model


def plot_dataset(X, T):
    import matplotlib.pyplot as plt

    rows, columns = X.shape
    x = [X[i][0] for i in range(rows)]
    y = [X[i][1] for i in range(rows)]
    c = ['green' if T[i][0] == 1 else 'lightblue' for i in range(rows)]
    s = 2
    plt.scatter(x, y, c=c, s=s)
    plt.show()


if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument("--seed", help="The initial seed of the random generator", type=int, default=random.randrange(0, sys.maxsize))
    cmdline_parser.add_argument('--sizes', type=str, default='2,64,64,2', help='A comma separated list of layer sizes, e.g. "2,64,64,2".')
    cmdline_parser.add_argument("--lr", help="The learning rate", type=float, default=0.1)
    cmdline_parser.add_argument("--epochs", help="The number of epochs", type=int, default=100)
    cmdline_parser.add_argument("--batch-size", help="The batch size", type=int, default=10)
    cmdline_parser.add_argument("--dataset-size", help="The dataset size", type=int, default=20000)
    cmdline_parser.add_argument("--plot", help="Plot the generated dataset", action="store_true")
    cmdline_parser.add_argument('--save-data', type=str, help='Save dataset to a file in .npz format')
    cmdline_parser.add_argument('--save-weights', type=str, help='Save weights to a file in .npz format')
    args = cmdline_parser.parse_args()

    print(f'seed = {args.seed}')
    manual_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_printoptions(precision=8, edgeitems=3, threshold=5, sci_mode=False, linewidth=160)

    Xtrain, Ttrain = make_dataset_chessboard(args.dataset_size)
    Xtest, Ttest = make_dataset_chessboard(args.dataset_size // 5)

    dataset = DataSet(Xtrain, Ttrain, Xtest, Ttest)

    if args.save_data:
        dataset.save(args.save_data)

    if args.plot:
        plot_dataset(Xtrain, Ttrain)
        plot_dataset(Xtest, Ttest)

    overall_density = 1.0
    loss = SquaredErrorLoss()
    learning_rate_scheduler = ConstantScheduler(args.lr)
    input_size = 2
    model = create_model(overall_density)
    model.compile(input_size, args.batch_size)

    if args.save_weights:
        model.compiled_model.export_weights(args.save_weights)

    print(model)
    stochastic_gradient_descent(model, dataset, loss, learning_rate_scheduler, epochs=args.epochs, batch_size=args.batch_size, shuffle=False, statistics=True)
