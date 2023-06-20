#!/usr/bin/env python3

# Copyright 2022 - 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from symbolic.learning_rate import ConstantScheduler
from symbolic.torch.datasets import DataLoader, create_npz_dataloaders
from symbolic.torch.loss_functions_rowwise import *
from symbolic.torch.multilayer_perceptron_rowwise import MultilayerPerceptron, parse_multilayer_perceptron
from symbolic.torch.utilities import set_torch_options, pp
from symbolic.training import SGDOptions, print_epoch
from symbolic.utilities import StopWatch


def to_one_hot(x: torch.LongTensor, n_classes: int):
    one_hot = torch.zeros(len(x), n_classes, dtype=torch.float)
    one_hot.scatter_(1, x.unsqueeze(1), 1)
    return one_hot


def compute_accuracy(M: MultilayerPerceptron, data_loader: DataLoader):
    N = len(data_loader.dataset)  # N is the number of examples
    total_correct = 0
    for X, T in data_loader:
        Y = M.feedforward(X)
        predicted = Y.argmax(dim=1)  # the predicted classes for the batch
        total_correct += (predicted == T).sum().item()

    return total_correct / N


def compute_loss(M: MultilayerPerceptron, data_loader: DataLoader, loss: LossFunction, num_classes: int):
    N = len(data_loader.dataset)  # N is the number of examples
    total_loss = 0.0
    for X, T in data_loader:
        T = to_one_hot(T, num_classes)
        Y = M.feedforward(X)
        total_loss += loss(Y, T)

    return total_loss / N


def compute_statistics(M, lr, loss, train_loader, test_loader, num_classes, epoch, print_statistics, elapsed_seconds):
    if print_statistics:
        train_loss = compute_loss(M, train_loader, loss, num_classes)
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
    num_classes = 10

    lr = learning_rate(0)
    compute_statistics(M, lr, loss, train_loader, test_loader, num_classes, epoch=0, print_statistics=True, elapsed_seconds=0.0)
    training_time = 0.0

    for epoch in range(epochs):
        timer = StopWatch()
        lr = learning_rate(epoch)  # update the learning at the start of each epoch

        for k, (X, T) in enumerate(train_loader):
            T = to_one_hot(T, num_classes)
            Y = M.feedforward(X)
            DY = loss.gradient(Y, T) / batch_size

            if SGDOptions.debug:
                print(f'epoch: {epoch} batch: {k}')
                M.info()
                pp("X", X)
                pp("Y", Y)
                pp("DY", DY)

            M.backpropagate(Y, DY)
            M.optimize(lr)

        seconds = timer.seconds()
        training_time += seconds
        compute_statistics(M, lr, loss, train_loader, test_loader, num_classes, epoch=epoch + 1, print_statistics=True, elapsed_seconds=seconds)

    print(f'Total training time for the {epochs} epochs: {training_time:.8f}s\n')


def main():
    layer_specifications = ['ReLU', 'ReLU', 'Linear']
    linear_layer_sizes = [3072, 1024, 512, 10]
    linear_layer_optimizers = ['Momentum(0.9)', 'Momentum(0.9)', 'Momentum(0.9)']
    linear_layer_weight_initializers = ['Xavier', 'Xavier', 'Xavier']
    batch_size = 100
    epochs = 1
    loss = SoftmaxCrossEntropyLossFunction()
    learning_rate = ConstantScheduler(0.01)
    SGDOptions.debug = True

    M = parse_multilayer_perceptron(layer_specifications, linear_layer_sizes, linear_layer_optimizers, linear_layer_weight_initializers, batch_size)
    M.load_weights_and_bias('../../mlp-compare.npz')
    train_loader, test_loader = create_npz_dataloaders('../../cifar1/epoch0.npz', batch_size=batch_size)
    sgd(M, epochs, loss, learning_rate, train_loader, test_loader, batch_size)


if __name__ == '__main__':
    set_torch_options()
    main()
