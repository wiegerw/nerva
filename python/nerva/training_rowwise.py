# Copyright 2022 - 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import List

import torch

from nerva.datasets_rowwise import DataLoader
from nerva.learning_rate_rowwise import LearningRateScheduler
from nerva.loss_rowwise import LossFunction
from nerva.layers_rowwise import Sequential, print_model_info
from nerva.utilities import MapTimer, pp
import nervalibrowwise


def to_one_hot(x: torch.LongTensor, num_classes: int):
    one_hot = torch.zeros(len(x), num_classes, dtype=torch.float)
    one_hot.scatter_(1, x.unsqueeze(1), 1)
    return one_hot

def print_epoch(epoch, lr, loss, train_accuracy, test_accuracy, elapsed):
    print(f'epoch {epoch:3}  '
          f'lr: {lr:.8f}  '
          f'loss: {loss:.8f}  '
          f'train accuracy: {train_accuracy:.8f}  '
          f'test accuracy: {test_accuracy:.8f}  '
          f'time: {elapsed:.8f}s'
         )


def compute_accuracy(M, data_loader: DataLoader):
    nervalibrowwise.nerva_timer_suspend()
    N = len(data_loader.dataset)  # N is the number of examples
    total_correct = 0
    for X, T in data_loader:
        Y = torch.Tensor(M.feedforward(X))  # TODO: this conversion should be eliminated
        predicted = Y.argmax(dim=1)  # the predicted classes for the batch
        total_correct += (predicted == T).sum().item()

    nervalibrowwise.nerva_timer_resume()
    return total_correct / N


def compute_loss(M, data_loader: DataLoader, loss: LossFunction):
    nervalibrowwise.nerva_timer_suspend()
    N = len(data_loader.dataset)  # N is the number of examples
    total_loss = 0.0
    num_classes = M.layers[-1].output_size
    for X, T in data_loader:
        T = to_one_hot(T, num_classes)
        Y = M.feedforward(X)
        total_loss += loss.value(Y, T)

    nervalibrowwise.nerva_timer_resume()
    return total_loss / N


def compute_statistics(M, lr, loss, train_loader, test_loader, epoch, elapsed_seconds=0.0, print_statistics=True):
    if print_statistics:
        train_loss = compute_loss(M, train_loader, loss)
        train_accuracy = compute_accuracy(M, train_loader)
        test_accuracy = compute_accuracy(M, test_loader)
        print_epoch(epoch, lr, train_loss, train_accuracy, test_accuracy, elapsed_seconds)
    else:
        print(f'epoch {epoch:3}')


def compute_sparse_layer_densities(overall_density: float, sizes: List[int], erk_power_scale: float = 1.0) -> List[float]:
    """
    Computes suitable densities for a number of linear layers.
    :param overall_density: the overall density of the layers
    :param sizes: the input and output sizes of  the layers
    :param erk_power_scale:
    """
    return nervalibrowwise.compute_sparse_layer_densities(overall_density, sizes, erk_power_scale)


class SGDOptions(nervalibrowwise.sgd_options):
    def __init__(self):
        super().__init__()
        self.epochs = 100
        self.batch_size = 1
        self.shuffle = True
        self.statistics = True
        self.debug = False
        self.gradient_step = 0

    def info(self):
        super().info()


class StochasticGradientDescentAlgorithm(object):
    def __init__(self,
                 M: Sequential,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 options: SGDOptions,
                 loss: LossFunction,
                 learning_rate: LearningRateScheduler
                ):
        self.M = M
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.options = options
        self.loss = loss
        self.learning_rate = learning_rate
        self.timer = MapTimer()

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
        for key, value in self.timer.values.items():
            if key.startswith("epoch"):
                result += self.timer.seconds(key)
        return result

    def run(self):
        M = self.M
        options = self.options
        num_classes = M.layers[-1].output_size

        self.on_start_training()

        lr = self.learning_rate(0)
        compute_statistics(M, lr, self.loss, self.train_loader, self.test_loader, 0, 0.0, options.statistics)

        for epoch in range(self.options.epochs):
            self.on_start_epoch(epoch)
            epoch_label = "epoch{}".format(epoch)
            self.timer.start(epoch_label)

            lr = self.learning_rate(epoch)  # update the learning at the start of each epoch

            for k, (X, T) in enumerate(self.train_loader):
                self.on_start_batch()
                T = to_one_hot(T, num_classes)
                Y = M.feedforward(X)
                DY = self.loss.gradient(Y, T) / options.batch_size

                if options.debug:
                    print(f'epoch: {epoch} batch: {k}')
                    print_model_info(M)
                    pp("X", X)
                    pp("Y", Y)
                    pp("DY", DY)

                M.backpropagate(Y, DY)
                M.optimize(lr)

                self.on_end_batch()

            self.timer.stop(epoch_label)
            seconds = self.timer.seconds(epoch_label)
            compute_statistics(M, lr, self.loss, self.train_loader, self.test_loader, epoch + 1, seconds, options.statistics)

            self.on_end_epoch(epoch)

        training_time = self.compute_training_time()
        print(f'Total training time for the {options.epochs} epochs: {training_time:.8f}s\n')

        self.on_end_training()
