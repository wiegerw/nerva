# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import List
from nerva.learning_rate import LearningRateScheduler
from nerva.loss import LossFunction
import nerva.layers
from nerva.utilities import StopWatch
from testing.numpy_utils import to_numpy, to_one_hot_numpy, l1_norm


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


def compute_accuracy(M, data_loader):
    N = len(data_loader.dataset)  # N is the number of examples
    total_correct = 0
    for X, T in data_loader:
        X = to_numpy(X)
        T = to_numpy(T)
        Y = M.feedforward(X)
        predicted = Y.argmax(axis=0)  # the predicted classes for the batch
        total_correct += (predicted == T).sum().item()
    return total_correct / N


def compute_loss(M, data_loader, loss):
    N = len(data_loader.dataset)  # N is the number of examples
    total_loss = 0.0
    for X, T in data_loader:
        X = to_numpy(X)
        T = to_one_hot_numpy(T, 10)
        Y = M.feedforward(X)
        total_loss += loss.value(Y, T)
    return total_loss / N


def print_epoch(epoch, lr, loss, train_accuracy, test_accuracy, elapsed):
    print(f'epoch {epoch:3}  '
          f'lr: {lr:.8f}  '
          f'loss: {loss:.8f}  '
          f'train accuracy: {train_accuracy:.8f}  '
          f'test accuracy: {test_accuracy:.8f}  '
          f'time: {elapsed:.8f}s'
         )


def compute_statistics(M, lr, loss, train_loader, test_loader, epoch, print_statistics, elapsed_seconds):
    if print_statistics:
        training_loss = compute_loss(M, train_loader, loss)
        training_accuracy = compute_accuracy(M, train_loader)
        test_accuracy = compute_accuracy(M, test_loader)
        print_epoch(epoch, lr, training_loss, training_accuracy, test_accuracy, elapsed_seconds)
    else:
        print(f'epoch {epoch:3}')


class SGD_Options:
    def __init__(self):
        self.epochs = 100
        self.batch_size = 1
        self.shuffle = True
        self.statistics = True
        self.debug = False
        self.gradient_step = 0

    def info(self):
        pass # implementation of info() method goes here


class StochasticGradientDescentAlgorithm(object):
    def __init__(self,
                 M: nerva.layers.Sequential,
                 train_loader,
                 test_loader,
                 options: SGD_Options,
                 loss: LossFunction,
                 learning_rate: LearningRateScheduler
                ):
        self.M = M
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.options = options
        self.loss = loss
        self.learning_rate = learning_rate
        self.timer = nerva.utilities.MapTimer()

    def on_start_training(self) -> None:
        """
        Event function that is called at the start of training
        """
        pass

    def on_end_training(self) -> None:
        """
        Event function that is called at the end of training
        """
        pass

    def on_start_epoch(self, epoch: int) -> None:
        """
        Event function that is called at the start of each epoch
        """
        pass

    def on_end_epoch(self, epoch: int) -> None:
        """
        Event function that is called at the end of each epoch
        """

    def on_start_batch(self) -> None:
        """
        Event function that is called at the start of each batch
        """
        pass

    def on_end_batch(self) -> None:
        """
        Event function that is called at the end of each batch
        """
        pass

    def compute_training_time(self) -> float:
        """
        Returns the sum of the measured times for the epochs
        """
        result = 0.0
        for key, value in self.timer.values().items():
            if key.startswith("epoch"):
                result += self.timer.seconds(key)
        return result

    def run(self):
        M = self.M
        options = self.options
        n_classes = M.layers[-1].units

        self.on_start_training()

        lr = self.learning_rate(0)
        compute_statistics(M, lr, self.loss, self.train_loader, self.test_loader, 0, options.statistics, 0)

        for epoch in range(self.options.epochs):
            self.on_start_epoch(epoch)
            epoch_label = "epoch{}".format(epoch)
            self.timer.start(epoch_label)

            lr = self.learning_rate(epoch)  # update the learning at the start of each epoch

            for (X, T) in self.train_loader:
                self.on_start_batch()
                X = to_numpy(X)
                T = to_one_hot_numpy(T, n_classes)
                Y = M.feedforward(X)
                DY = self.loss.gradient(Y, T) / options.batch_size
                M.backpropagate(Y, DY)
                M.optimize(lr)
                self.on_end_batch()

            self.timer.stop(epoch_label)
            seconds = self.timer.seconds(epoch_label)
            compute_statistics(M, lr, self.loss, self.train_loader, self.test_loader, epoch + 1, options.statistics, seconds)

            self.on_end_epoch(epoch)

        self.on_end_training()
