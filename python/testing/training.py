#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import List, Optional

import numpy as np

import nerva.weights
from nerva.layers import Sparse, Sequential
from nerva.utilities import StopWatch
from testing.datasets import create_npz_dataloaders
from testing.numpy_utils import to_numpy, to_one_hot_numpy, l1_norm
from testing.models import MLPPyTorch, MLPNerva


def compute_accuracy_torch(M: MLPPyTorch, data_loader):
    N = len(data_loader.dataset)  # N is the number of examples
    total_correct = 0
    for X, T in data_loader:
        Y = M(X)
        predicted = Y.argmax(axis=1)  # the predicted classes for the batch
        total_correct += (predicted == T).sum().item()
    return total_correct / N


def compute_loss_torch(M: MLPPyTorch, data_loader):
    N = len(data_loader.dataset)  # N is the number of examples
    batch_size = N // len(data_loader)
    total_loss = 0.0
    for X, T in data_loader:
        Y = M(X)
        total_loss += M.loss(Y, T).sum()
    return batch_size * total_loss / N


def compute_accuracy_nerva(M: MLPNerva, data_loader):
    N = len(data_loader.dataset)  # N is the number of examples
    total_correct = 0
    for X, T in data_loader:
        X = to_numpy(X)
        T = to_numpy(T)
        Y = M.feedforward(X)
        predicted = Y.argmax(axis=0)  # the predicted classes for the batch
        total_correct += (predicted == T).sum().item()
    return total_correct / N


def compute_loss_nerva(M: MLPNerva, data_loader):
    N = len(data_loader.dataset)  # N is the number of examples
    total_loss = 0.0
    for X, T in data_loader:
        X = to_numpy(X)
        T = to_one_hot_numpy(T, 10)
        Y = M.feedforward(X)
        total_loss += M.loss.value(Y, T)
    return total_loss / N


def compute_weight_difference(M1, M2):
    weights1 = M1.weights()
    weights2 = M2.weights()
    bias1 = M1.bias()
    bias2 = M2.bias()
    weightdiff = [l1_norm(W1 - W2) for W1, W2 in zip(weights1, weights2)]
    biasdiff = [l1_norm(b1 - b2) for b1, b2 in zip(bias1, bias2)]
    weightnorm1 = [l1_norm(W) for W in weights1]
    weightnorm2 = [l1_norm(W) for W in weights2]
    return weightdiff, biasdiff, weightnorm1, weightnorm2


def compute_matrix_difference(name, X1: np.ndarray, X2: np.ndarray):
    print(f'{name} difference: {l1_norm(X1 - X2)}')


def print_epoch(epoch, lr, loss, train_accuracy, test_accuracy, elapsed):
    print(f'epoch {epoch:3}  '
          f'lr: {lr:.8f}  '
          f'loss: {loss:.8f}  '
          f'train accuracy: {train_accuracy:.8f}  '
          f'test accuracy: {test_accuracy:.8f}  '
          f'time: {elapsed:.8f}s'
         )


def measure_inference_time_torch(M, train_loader, density, repetitions=100):
    M.train()  # Set model in training mode
    batch_size = len(train_loader.dataset) // len(train_loader)
    watch = StopWatch()

    total_time = 0.0
    for k, (X, T) in enumerate(train_loader):
        watch.reset()
        Y = M(X)
        elapsed = watch.seconds()
        if k > 0:  # skip the first batch, because it is slow (TODO: find out why)
            total_time += elapsed
        print(f'batch {k} took {elapsed:.8f} seconds')
        if k == repetitions:
            break
    print(f'Average PyTorch inference time for density={density} batch_size={batch_size}: {1000.0 * total_time/repetitions:.4f}ms')


def measure_inference_time_nerva(M, train_loader, density, repetitions=100):
    batch_size = len(train_loader.dataset) // len(train_loader)
    watch = StopWatch()

    total_time = 0.0
    for k, (X, T) in enumerate(train_loader):
        X = to_numpy(X)
        watch.reset()
        Y = M.feedforward(X)
        elapsed = watch.seconds()
        if k > 0:  # skip the first batch, because it is slow (TODO: find out why)
            total_time += elapsed
        print(f'batch {k} took {elapsed:.8f} seconds')
        if k == repetitions:
            break
    print(f'Average Nerva inference time for density={density} batch_size={batch_size}: {1000.0 * total_time/repetitions:.4f}ms')


class RegrowFunction(object):
    """
    Interface for regrowing the sparse layers of a neural network
    """
    def __call__(self, M: Sequential):
        raise NotImplementedError


class RegrowMagnitude(RegrowFunction):
    """
    Regrows a fraction zeta of the weights in the sparse layers. Weights
    are pruned according to their magnitude.
    """

    def __init__(self, zeta):
        self.zeta = zeta

    def __call__(self, M: Sequential):
        print('regrow weights based on magnitude')
        for layer in M.layers:
            if isinstance(layer, Sparse):
                weight_count = layer.weight_count()
                count = int(self.zeta * weight_count)
                print(f'regrowing {count}/{weight_count} weights')
                layer.prune_weights(count)
                layer.grow_weights(count, nerva.weights.XavierNormalized())


class RegrowMagnitudePosNeg(RegrowFunction):
    """
    Regrows a fraction zeta of the positive and a fraction zeta of the negative
    weights in the sparse layers. Weights are pruned according to their magnitude.
    """

    def __init__(self, zeta):
        self.zeta = zeta

    def __call__(self, M: Sequential):
        print('regrow positive and negative weights')
        for layer in M.layers:
            if isinstance(layer, Sparse):
                weight_count = layer.weight_count()
                negative_count = int(self.zeta * layer.negative_weight_count())
                positive_count = int(self.zeta * layer.positive_weight_count())
                print(f'regrowing {negative_count + positive_count}/{weight_count} weights')
                layer.prune_negative_weights(negative_count)
                layer.prune_positive_weights(positive_count)
                layer.grow_weights(negative_count + positive_count, nerva.weights.XavierNormalized())


def train_torch(M, train_loader, test_loader, epochs):
    M.train()  # Set model in training mode
    watch = StopWatch()

    print_epoch(epoch=0,
                lr=M.optimizer.param_groups[0]["lr"],
                loss=compute_loss_torch(M, train_loader),
                train_accuracy=compute_accuracy_torch(M, train_loader),
                test_accuracy=compute_accuracy_torch(M, test_loader),
                elapsed=0)

    for epoch in range(epochs):
        elapsed = 0.0
        for k, (X, T) in enumerate(train_loader):
            watch.reset()
            M.optimizer.zero_grad()
            Y = M(X)
            loss = M.loss(Y, T)
            loss.backward()
            M.optimize()
            elapsed += watch.seconds()

        print_epoch(epoch=epoch + 1,
                    lr=M.optimizer.param_groups[0]["lr"],
                    loss=compute_loss_torch(M, train_loader),
                    train_accuracy=compute_accuracy_torch(M, train_loader),
                    test_accuracy=compute_accuracy_torch(M, test_loader),
                    elapsed=elapsed)

        M.learning_rate.step()  # N.B. this updates the learning rate in M.optimizer


# At every epoch a new dataset in .npz format is read from datadir.
def train_torch_preprocessed(M, datadir, epochs, batch_size):
    M.train()  # Set model in training mode
    watch = StopWatch()

    train_loader, test_loader = create_npz_dataloaders(f'{datadir}/epoch0.npz', batch_size=batch_size)

    print_epoch(epoch=0,
                lr=M.optimizer.param_groups[0]["lr"],
                loss=compute_loss_torch(M, train_loader),
                train_accuracy=compute_accuracy_torch(M, train_loader),
                test_accuracy=compute_accuracy_torch(M, test_loader),
                elapsed=0)

    for epoch in range(epochs):
        if epoch > 0:
            train_loader, test_loader = create_npz_dataloaders(f'{datadir}/epoch{epoch}.npz', batch_size)

        elapsed = 0.0
        for k, (X, T) in enumerate(train_loader):
            watch.reset()
            M.optimizer.zero_grad()
            Y = M(X)
            loss = M.loss(Y, T)
            loss.backward()
            M.optimize()
            elapsed += watch.seconds()

        print_epoch(epoch=epoch + 1,
                    lr=M.optimizer.param_groups[0]["lr"],
                    loss=compute_loss_torch(M, train_loader),
                    train_accuracy=compute_accuracy_torch(M, train_loader),
                    test_accuracy=compute_accuracy_torch(M, test_loader),
                    elapsed=elapsed)

        M.learning_rate.step()  # N.B. this updates the learning rate in M.optimizer


def train_nerva(M, train_loader, test_loader, epochs, regrow: Optional[RegrowFunction] = None):
    n_classes = M.layer_sizes[-1]
    batch_size = len(train_loader.dataset) // len(train_loader)
    watch = StopWatch()

    print_epoch(epoch=0,
                lr=M.learning_rate(0),
                loss=compute_loss_nerva(M, train_loader),
                train_accuracy=compute_accuracy_nerva(M, train_loader),
                test_accuracy=compute_accuracy_nerva(M, test_loader),
                elapsed=0)

    for epoch in range(epochs):
        if epoch > 0 and regrow:
            regrow(M)

        lr = M.learning_rate(epoch)
        elapsed = 0.0
        for k, (X, T) in enumerate(train_loader):
            watch.reset()
            X = to_numpy(X)
            T = to_one_hot_numpy(T, n_classes)
            Y = M.feedforward(X)
            DY = M.loss.gradient(Y, T) / batch_size
            M.backpropagate(Y, DY)
            M.optimize(lr)
            elapsed += watch.seconds()

        print_epoch(epoch=epoch + 1,
                    lr=lr,
                    loss=compute_loss_nerva(M, train_loader),
                    train_accuracy=compute_accuracy_nerva(M, train_loader),
                    test_accuracy=compute_accuracy_nerva(M, test_loader),
                    elapsed=elapsed)


# At every epoch a new dataset in .npz format is read from datadir.
def train_nerva_preprocessed(M, datadir, epochs, batch_size, regrow: Optional[RegrowFunction] = None):
    train_loader, test_loader = create_npz_dataloaders(f'{datadir}/epoch0.npz', batch_size=batch_size)

    n_classes = M.layer_sizes[-1]
    batch_size = len(train_loader.dataset) // len(train_loader)
    watch = StopWatch()

    print_epoch(epoch=0,
                lr=M.learning_rate(0),
                loss=compute_loss_nerva(M, train_loader),
                train_accuracy=compute_accuracy_nerva(M, train_loader),
                test_accuracy=compute_accuracy_nerva(M, test_loader),
                elapsed=0)

    for epoch in range(epochs):
        if epoch > 0:
            train_loader, test_loader = create_npz_dataloaders(f'{datadir}/epoch{epoch}.npz', batch_size)

        if epoch > 0 and regrow:
            regrow(M)

        lr = M.learning_rate(epoch)
        elapsed = 0.0
        for k, (X, T) in enumerate(train_loader):
            watch.reset()
            X = to_numpy(X)
            T = to_one_hot_numpy(T, n_classes)
            Y = M.feedforward(X)
            DY = M.loss.gradient(Y, T) / batch_size
            M.backpropagate(Y, DY)
            M.optimize(lr)
            elapsed += watch.seconds()

        print_epoch(epoch=epoch + 1,
                    lr=lr,
                    loss=compute_loss_nerva(M, train_loader),
                    train_accuracy=compute_accuracy_nerva(M, train_loader),
                    test_accuracy=compute_accuracy_nerva(M, test_loader),
                    elapsed=elapsed)


def compute_densities(overall_density: float, sizes: List[int], erk_power_scale: float = 1.0) -> List[float]:
    layer_shapes = [(sizes[i], sizes[i+1]) for i in range(len(sizes) - 1)]
    n = len(layer_shapes)  # the number of layers

    if overall_density == 1.0:
        return [1.0] * n

    dense_layers = set()

    while True:
        divisor = 0
        rhs = 0
        raw_probabilities = [0.0] * n
        for i, (rows, columns) in enumerate(layer_shapes):
            n_param = rows * columns
            n_zeros = n_param * (1 - overall_density)
            n_ones = n_param * overall_density
            if i in dense_layers:
                rhs -= n_zeros
            else:
                rhs += n_ones
                raw_probabilities[i] = ((rows + columns) / (rows * columns)) ** erk_power_scale
                divisor += raw_probabilities[i] * n_param
        epsilon = rhs / divisor
        max_prob = max(raw_probabilities)
        max_prob_one = max_prob * epsilon
        if max_prob_one > 1:
            for j, mask_raw_prob in enumerate(raw_probabilities):
                if mask_raw_prob == max_prob:
                    dense_layers.add(j)
        else:
            break

    # Compute the densities
    densities = [0.0] * n
    total_nonzero = 0.0
    for i, (rows, columns) in enumerate(layer_shapes):
        n_param = rows * columns
        if i in dense_layers:
            densities[i] = 1.0
        else:
            probability_one = epsilon * raw_probabilities[i]
            densities[i] = probability_one
        total_nonzero += densities[i] * n_param

    return densities
