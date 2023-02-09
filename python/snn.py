#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import nerva.dataset
import nerva.layers
import nerva.learning_rate
import nerva.loss
import nerva.optimizers
import nerva.random
import nervalib
from testing.datasets import create_cifar10_augmented_dataloaders, create_cifar10_dataloaders
from testing.nerva_models import make_nerva_optimizer, make_nerva_scheduler
from testing.torch_models import make_torch_mask, make_torch_scheduler
from testing.models import MLP1, MLP1a, MLP2
from testing.training import train_nerva, train_torch, compute_accuracy_torch, compute_accuracy_nerva, \
    compute_densities, train_torch_preprocessed, train_nerva_preprocessed, measure_inference_time_nerva, measure_inference_time_torch


def make_torch_model_new(args, sizes, densities):
    print('Use MLP1 variant with custom masking')
    M1 = MLP1a(sizes, densities)
    M1.optimizer = optim.SGD(M1.parameters(), lr=args.lr, momentum=args.momentum, nesterov=args.nesterov)
    M1.loss = nn.CrossEntropyLoss()
    M1.learning_rate = make_torch_scheduler(args, M1.optimizer)
    for layer in M1.layers:
        nn.init.xavier_uniform_(layer.weight)
    M1.apply_masks()
    return M1


def make_torch_model(args, sizes):
    M1 = MLP1(sizes)
    M1.optimizer = optim.SGD(M1.parameters(), lr=args.lr, momentum=args.momentum, nesterov=args.nesterov)
    if args.density != None:
        mask = make_torch_mask(M1, args.density)
        mask.add_module(M1, density=args.density, sparse_init='ER')
        M1.mask = mask
    M1.loss = nn.CrossEntropyLoss()
    M1.learning_rate = make_torch_scheduler(args, M1.optimizer)
    return M1


def make_nerva_model(args, sizes, densities):
    optimizer2 = make_nerva_optimizer(args.momentum, args.nesterov)
    M2 = MLP2(sizes, densities, optimizer2, args.batch_size)
    M2.loss = nerva.loss.SoftmaxCrossEntropyLoss()
    M2.learning_rate = make_nerva_scheduler(args)
    return M2


def make_argument_parser():
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument("--batch-size", help="The batch size", type=int, default=1)
    cmdline_parser.add_argument("--seed", help="The initial seed of the random generator", type=int)
    cmdline_parser.add_argument("--precision", help="The precision used for printing", type=int, default=4)
    cmdline_parser.add_argument("--edgeitems", help="The edgeitems used for printing matrices", type=int, default=3)
    cmdline_parser.add_argument('--density', type=float, default=1.0, help='The density of the overall sparse network.')
    cmdline_parser.add_argument('--sizes', type=str, default='3072,128,64,10', help='A comma separated list of layer sizes, e.g. "3072,128,64,10".')
    cmdline_parser.add_argument("--epochs", help="The number of epochs", type=int, default=100)
    cmdline_parser.add_argument("--lr", help="The learning rate", type=float, default=0.1)
    cmdline_parser.add_argument('--momentum', type=float, default=0.9, help='the momentum value (default: off)')
    cmdline_parser.add_argument('--gamma', type=float, default=0.1, help='the learning rate decay (default: 0.1)')
    cmdline_parser.add_argument("--nesterov", help="apply nesterov", action="store_true")
    cmdline_parser.add_argument('--datadir', type=str, default='./data', help='the data directory (default: ./data)')
    cmdline_parser.add_argument("--augmented", help="use data loaders with augmentation", action="store_true")
    cmdline_parser.add_argument("--preprocessed", help="folder with preprocessed datasets for each epoch")
    cmdline_parser.add_argument("--nerva", help="Train using a Nerva model", action="store_true")
    cmdline_parser.add_argument("--torch", help="Train using a PyTorch model", action="store_true")
    cmdline_parser.add_argument("--inference", help="Estimate inference time", action="store_true")
    cmdline_parser.add_argument("--storage", type=str, help="Save the model in .npy format (N.B. this is only used for measuring disk sizes!)")
    cmdline_parser.add_argument("--scheduler", type=str, help="the learning rate scheduler (constant,multistep)", default="multistep")
    cmdline_parser.add_argument('--export-weights-npz', type=str, help='Export weights to a file in .npz format')
    cmdline_parser.add_argument('--import-weights-npz', type=str, help='Import weights from a file in .npz format')
    cmdline_parser.add_argument("--custom-masking", help="Use a custom variant of masking in the PyTorch models", action="store_true")
    return cmdline_parser


def initialize_frameworks(args):
    if args.seed:
        torch.manual_seed(args.seed)
        nerva.random.manual_seed(args.seed)

    torch.set_printoptions(precision=args.precision, edgeitems=args.edgeitems, threshold=5, sci_mode=False, linewidth=160)

    # avoid 'Too many open files' error when using data loaders
    torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    cmdline_parser = make_argument_parser()
    args = cmdline_parser.parse_args()

    print('=== Command line arguments ===')
    print(args)

    initialize_frameworks(args)

    if args.epochs <= 10:
        print('Setting gamma to 1.0')
        args.gamma = 1.0

    if args.augmented:
        train_loader, test_loader = create_cifar10_augmented_dataloaders(args.batch_size, args.batch_size, args.datadir)
    else:
        train_loader, test_loader = create_cifar10_dataloaders(args.batch_size, args.batch_size, args.datadir)

    sizes = [int(s) for s in args.sizes.split(',')]
    densities = compute_densities(args.density, sizes)

    M1 = make_torch_model(args, sizes)
    M2 = make_nerva_model(args, sizes, densities)

    if args.augmented and args.preprocessed:
        raise RuntimeError('the combination of --augmented and --preprocessed is unsupported')

    if args.custom_masking:
        M1 = make_torch_model_new(args, sizes, densities)

    if args.torch:
        print('\n=== PyTorch model ===')
        print(M1)
        print(M1.loss)
        print(M1.learning_rate)

        if args.export_weights_npz:
            M1.export_weights_npz(args.export_weights_npz)

        print('\n=== Training PyTorch model ===')
        if args.preprocessed:
            train_torch_preprocessed(M1, args.preprocessed, args.epochs, args.batch_size)
        else:
            train_torch(M1, train_loader, test_loader, args.epochs)
        print(f'Accuracy of the network on the 10000 test images: {100 * compute_accuracy_torch(M1, test_loader):.3f} %')
    elif args.nerva:
        print('\n=== Nerva model ===')
        print(M2)
        print(M2.loss)
        print(M2.learning_rate)

        if args.import_weights_npz:
            M2.import_weights_npz(args.import_weights_npz)

        print('\n=== Training Nerva model ===')
        if args.preprocessed:
            train_nerva_preprocessed(M2, args.preprocessed, args.epochs, args.batch_size)
        else:
            train_nerva(M2, train_loader, test_loader, args.epochs)
        print(f'Accuracy of the network on the 10000 test images: {100 * compute_accuracy_nerva(M2, test_loader):.3f} %')
    elif args.inference:
        measure_inference_time_torch(M1, train_loader, args.density, repetitions=10)
        measure_inference_time_nerva(M2, train_loader, args.density, repetitions=10)
    elif args.storage:
        print(M2)
        nervalib.save_model_weights_to_npy(args.save_model_npy, M2.compiled_model)


if __name__ == '__main__':
    main()
