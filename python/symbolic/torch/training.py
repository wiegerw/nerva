# Copyright 2022 - 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import torch
from symbolic.torch.datasets import DataLoader
from symbolic.torch.multilayer_perceptron import MultilayerPerceptron
from symbolic.utilities import StopWatch


def to_one_hot(x: torch.LongTensor, n_classes: int):
    one_hot = torch.zeros(len(x), n_classes)
    one_hot.scatter_(1, x.unsqueeze(1), 1)
    return one_hot


def compute_accuracy(M: MultilayerPerceptron, data_loader: DataLoader):
    N = len(data_loader.dataset)  # N is the number of examples
    total_correct = 0
    for X, T in data_loader:
        Y = M.feedforward(X)
        predicted = Y.argmax(dim=0)  # the predicted classes for the batch
        total_correct += (predicted == T).sum().item()

    return total_correct / N


def compute_loss(M: MultilayerPerceptron, data_loader: DataLoader, loss: LossFunction):
    N = len(data_loader.dataset)  # N is the number of examples
    total_loss = 0.0
    for X, T in data_loader:
        T = to_one_hot(T, 10)
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
        train_loss = compute_loss(M, train_loader, loss)
        train_accuracy = compute_accuracy(M, train_loader)
        test_accuracy = compute_accuracy(M, test_loader)
        print_epoch(epoch, lr, train_loss, train_accuracy, test_accuracy, elapsed_seconds)
    else:
        print(f'epoch {epoch:3}')


def sgd(M: MultilayerPerceptron,
        epochs: int,
        loss,
        learning_rate,
        train_loader: DataLoader,
        test_loader: DataLoader,
        batch_size: int
       ):
    n_classes = M.layers[-1].units

    lr = learning_rate(0)
    compute_statistics(M, lr, loss, train_loader, test_loader, epoch=0, print_statistics=True, elapsed_seconds=0.0)
    training_time = 0.0

    for epoch in range(epochs):
        timer = StopWatch()
        lr = learning_rate(epoch)  # update the learning at the start of each epoch

        for (X, T) in train_loader:
            T = to_one_hot(T, n_classes)
            Y = M.feedforward(X)
            DY = loss.gradient(Y, T) / batch_size
            M.backpropagate(Y, DY)
            M.optimize(lr)

        seconds = timer.seconds()
        training_time += seconds
        compute_statistics(M, lr, loss, train_loader, test_loader, epoch + 1, print_statistics=True, elapsed_seconds=seconds)

    print(f'Total training time for the {epochs} epochs: {training_time:.8f}s\n')
