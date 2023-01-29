#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import nerva.dataset
import nerva.layers
import nerva.learning_rate
import nerva.loss
import nerva.optimizers
import nerva.random
from testing.datasets import create_cifar10_dataloaders, load_cifar10_data, TorchDataLoader
from testing.nerva_models import make_nerva_optimizer, make_nerva_scheduler
from testing.torch_models import make_torch_mask, make_torch_scheduler
from testing.models import MLP1, MLP2, copy_weights_and_biases, print_model_info
from testing.training import train_torch, compute_accuracy_torch, compute_accuracy_nerva, train_nerva, \
    train_both, compute_densities


def make_models(args, sizes, densities) -> Tuple[MLP1, MLP2]:
    # create PyTorch model
    M1 = MLP1(sizes)
    M1.optimizer = optim.SGD(M1.parameters(), lr=args.lr, momentum=args.momentum, nesterov=args.nesterov)
    if args.density != None:
        mask = make_torch_mask(M1, args.density)
        mask.add_module(M1, density=args.density, sparse_init='ER')
        M1.mask = mask
    M1.loss = nn.CrossEntropyLoss()
    M1.learning_rate = make_torch_scheduler(args, M1.optimizer)

    # create Nerva model
    optimizer2 = make_nerva_optimizer(args.momentum, args.nesterov)
    M2 = MLP2(sizes, densities, optimizer2, args.batch_size)
    M2.loss = nerva.loss.SoftmaxCrossEntropyLoss()
    M2.learning_rate = make_nerva_scheduler(args)

    print('\n=== PyTorch model ===')
    print(M1)
    print(M1.loss)
    print(M1.learning_rate)

    print('\n=== Nerva model ===')
    print(M2)
    print(M2.loss)
    print(M2.learning_rate)

    return M1, M2


def make_argument_parser():
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument("--show", help="Show data and intermediate results", action="store_true")
    cmdline_parser.add_argument("--batch-size", help="The batch size", type=int, default=1)
    cmdline_parser.add_argument("--seed", help="The initial seed of the random generator", type=int)
    cmdline_parser.add_argument("--precision", help="The precision used for printing", type=int, default=4)
    cmdline_parser.add_argument("--edgeitems", help="The edgeitems used for printing matrices", type=int, default=3)
    cmdline_parser.add_argument("--epochs", help="The number of epochs", type=int, default=100)
    cmdline_parser.add_argument("--lr", help="The learning rate", type=float, default=0.1)
    cmdline_parser.add_argument('--momentum', type=float, default=0.9, help='the momentum value (default: off)')
    cmdline_parser.add_argument('--gamma', type=float, default=0.1, help='the learning rate decay (default: 0.1)')
    cmdline_parser.add_argument("--nesterov", help="apply nesterov", action="store_true")
    cmdline_parser.add_argument('--datadir', type=str, default='./data', help='the data directory (default: ./data)')
    cmdline_parser.add_argument("--augmented", help="use data loaders with augmentation", action="store_true")
    cmdline_parser.add_argument('--density', type=float, default=1.0, help='The density of the overall sparse network.')
    cmdline_parser.add_argument('--sizes', type=str, default='3072,128,64,10', help='A comma separated list of layer sizes, e.g. "3072,128,64,10".')
    cmdline_parser.add_argument("--copy", help="copy weights and biases from the PyTorch model to the Nerva model", action="store_true")
    cmdline_parser.add_argument("--nerva", help="Train using a Nerva model", action="store_true")
    cmdline_parser.add_argument("--torch", help="Train using a PyTorch model", action="store_true")
    #cmdline_parser.add_argument("--dense-sparse", help="Train using a dense and sparse Nerva model", action="store_true")
    cmdline_parser.add_argument("--info", help="Print detailed info about the models", action="store_true")
    cmdline_parser.add_argument("--scheduler", type=str, help="the learning rate scheduler (constant,multistep)", default="multistep")
    return cmdline_parser


def initialize_frameworks(args):
    if args.seed:
        torch.manual_seed(args.seed)
        nerva.random.manual_seed(args.seed)

    torch.set_printoptions(precision=args.precision, edgeitems=args.edgeitems, threshold=5, sci_mode=False, linewidth=120)

    # avoid 'Too many open files' error when using data loaders
    torch.multiprocessing.set_sharing_strategy('file_system')


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

    M1, M2 = make_models(args, sizes, densities)

    if args.copy:
        copy_weights_and_biases(M1, M2)

    if args.info:
        print('\n=== PyTorch info ===')
        print_model_info(M1)
        print('\n=== Nerva info ===')
        print_model_info(M2)

    if args.torch and args.nerva:
        print('\n=== Training PyTorch and Nerva model ===')
        train_both(M1, M2, train_loader, test_loader, args.epochs, args.show)
        print(f'Accuracy of the network M1 on the 10000 test images: {100 * compute_accuracy_torch(M1, test_loader):.3f} %')
        print(f'Accuracy of the network M2 on the 10000 test images: {100 * compute_accuracy_nerva(M2, test_loader):.3f} %')
    elif args.torch:
        print('\n=== Training PyTorch model ===')
        train_torch(M1, train_loader, test_loader, args.epochs, args.show)
        print(f'Accuracy of the network on the 10000 test images: {100 * compute_accuracy_torch(M1, test_loader):.3f} %')
    elif args.nerva:
        print('\n=== Training Nerva model ===')
        train_nerva(M2, train_loader, test_loader, args.epochs, args.show)
        print(f'Accuracy of the network on the 10000 test images: {100 * compute_accuracy_nerva(M2, test_loader):.3f} %')


if __name__ == '__main__':
    main()
