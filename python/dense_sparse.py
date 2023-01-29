#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from timeit import default_timer as timer
from testing.datasets import create_cifar10_dataloaders, load_cifar10_data, TorchDataLoader
from testing.nerva_models import make_nerva_optimizer, make_nerva_scheduler
from testing.models import MLP2, copy_weights_and_biases
from testing.numpy_utils import to_numpy, to_one_hot_numpy
from testing.training import compute_accuracy_nerva, compute_loss_nerva, compute_weight_difference, \
    compute_matrix_difference, compute_densities
import nerva.loss
from nervalib import MLPMasking
from snn import make_argument_parser, initialize_frameworks


# M1 is a dense model
# M2 is a sparse model with the same structure
def train_dense_sparse(M1: MLP2, M2: MLP2, train_loader, test_loader, epochs, show: bool):
    n_classes = M2.sizes[-1]
    batch_size = len(train_loader.dataset) // len(train_loader)

    compute_weight_difference(M1, M2)
    masking = MLPMasking(M2.compiled_model)

    for epoch in range(epochs):
        start = timer()
        lr = M1.learning_rate(epoch)

        for k, (X, T) in enumerate(train_loader):
            X = to_numpy(X)
            T = to_one_hot_numpy(T, n_classes)

            Y1 = M1.feedforward(X)
            DY1 = M1.loss.gradient(Y1, T) / batch_size
            M1.backpropagate(Y1, DY1)
            M1.optimize(lr)

            Y2 = M2.feedforward(X)
            DY2 = M2.loss.gradient(Y2, T) / batch_size
            M2.backpropagate(Y2, DY2)
            M2.optimize(lr)

            # compute_weight_difference(M1, M2)
            masking.apply(M1.compiled_model)

            print(f'epoch: {epoch} batch: {k}')
            compute_matrix_difference('Y', Y1, Y2)
            compute_matrix_difference('DY', DY1, DY2)
            # pp('Y', Y1)
            # pp('DY', DY1)
            # pp('Y', Y2)
            # pp('DY', DY2)
            compute_weight_difference(M1, M2)

            elapsed = timer() - start

        print(f'epoch {epoch + 1:3}  '
              f'lr: {lr:.4f}  '
              f'loss: {compute_loss_nerva(M1, train_loader):.3f}  '
              f'train accuracy: {compute_accuracy_nerva(M1, train_loader):.3f}  '
              f'test accuracy: {compute_accuracy_nerva(M1, test_loader):.3f}  '
              f'time: {elapsed:.3f}'
             )

        print(f'epoch {epoch + 1:3}  '
              f'lr: {lr:.4f}  '
              f'loss: {compute_loss_nerva(M2, train_loader):.3f}  '
              f'train accuracy: {compute_accuracy_nerva(M2, train_loader):.3f}  '
              f'test accuracy: {compute_accuracy_nerva(M2, test_loader):.3f}  '
              f'time: {elapsed:.3f}'
             )


def make_sparse_model(args, sizes, densities) -> MLP2:
    optimizer = make_nerva_optimizer(args.momentum, args.nesterov)
    M = MLP2(sizes, densities, optimizer, args.batch_size)
    M.loss = nerva.loss.SoftmaxCrossEntropyLoss()
    M.learning_rate = make_nerva_scheduler(args)
    return M


def make_dense_copy(M: MLP2, sizes, densities, batch_size) -> MLP2:
    full_densities = [1.0 for _ in densities]
    M1 = MLP2(sizes, full_densities, M.optimizer, batch_size)
    M1.loss = M.loss
    M1.learning_rate = M.learning_rate
    copy_weights_and_biases(M, M1)
    return M1


def main():
    cmdline_parser = make_argument_parser()
    args = cmdline_parser.parse_args()

    print('=== Command line arguments ===')
    print(args)

    initialize_frameworks(args)

    if args.augmented:
        train_loader, test_loader = create_cifar10_dataloaders(args.batch_size, args.batch_size, args.datadir)
    else:
        Xtrain, Ttrain, Xtest, Ttest = load_cifar10_data(args.datadir)
        train_loader = TorchDataLoader(Xtrain, Ttrain, args.batch_size)
        test_loader = TorchDataLoader(Xtest, Ttest, args.batch_size)

    sizes = [int(s) for s in args.sizes.split(',')]
    densities = compute_densities(args.density, sizes)

    M1 = make_sparse_model(args, sizes, densities)
    M2 = make_dense_copy(M1, sizes, densities, args.batch_size)

    print('\n=== Nerva sparse model ===')
    print(M1)
    print(M1.loss)
    print(M1.learning_rate)

    print('\n=== Nerva dense model ===')
    print(M2)
    print(M2.loss)
    print(M2.learning_rate)

    print('\n=== Training dense Nerva and sparse Nerva model ===')
    train_dense_sparse(M2, M1, train_loader, test_loader, args.epochs, args.show)
    print(f'Accuracy of the network M1 on the 10000 test images: {100 * compute_accuracy_nerva(M1, test_loader):.3f} %')
    print(f'Accuracy of the network M2 on the 10000 test images: {100 * compute_accuracy_nerva(M2, test_loader):.3f} %')


if __name__ == '__main__':
    main()
