#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse
import shlex
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from nerva.training import compute_densities
from testing.datasets import create_cifar10_augmented_dataloaders, create_cifar10_dataloaders
from testing.torch_models import make_torch_scheduler
from testing.models import MLPPyTorch, MLPPyTorchTRelu
from testing.training import train_torch, train_torch_preprocessed


def make_torch_model(args, linear_layer_sizes, densities):
    M = MLPPyTorch(linear_layer_sizes, densities) if args.trim_relu == 0 else MLPPyTorchTRelu(linear_layer_sizes, densities, args.trim_relu)
    M.optimizer = optim.SGD(M.parameters(), lr=args.lr, momentum=args.momentum, nesterov=args.nesterov)
    M.loss = nn.CrossEntropyLoss()
    M.learning_rate = make_torch_scheduler(args, M.optimizer)
    for layer in M.layers:
        nn.init.xavier_uniform_(layer.weight)
    M.apply_masks()
    return M


def make_argument_parser():
    cmdline_parser = argparse.ArgumentParser()

    # randomness
    cmdline_parser.add_argument("--seed", help="The initial seed of the random generator", type=int)

    # model parameters
    cmdline_parser.add_argument('--sizes', type=str, default='3072,128,64,10', help='A comma separated list of layer sizes, e.g. "3072,128,64,10".')
    cmdline_parser.add_argument('--densities', type=str, help='A comma separated list of layer densities, e.g. "0.05,0.05,1.0".')
    cmdline_parser.add_argument('--overall-density', type=float, default=1.0, help='The overall density of the layers.')
    cmdline_parser.add_argument('--trim-relu', type=float, default=0.0, help='The threshold for trimming ReLU outputs.')
    cmdline_parser.add_argument('--layers', type=str, help='A semi-colon separated lists of layers.')

    # optimizer
    cmdline_parser.add_argument('--momentum', type=float, default=0.9, help='the momentum value (default: off)')
    cmdline_parser.add_argument("--nesterov", help="apply nesterov", action="store_true")

    # learning rate
    cmdline_parser.add_argument("--lr", help="The initial learning rate", type=float, default=0.1)
    cmdline_parser.add_argument("--scheduler", type=str, help="The learning rate scheduler (constant,multistep)", default="multistep")
    cmdline_parser.add_argument('--gamma', type=float, default=0.1, help='The learning rate decay (default: 0.1)')

    # training
    cmdline_parser.add_argument("--epochs", help="The number of epochs", type=int, default=100)
    cmdline_parser.add_argument("--batch-size", help="The batch size", type=int, default=1)

    # dataset
    cmdline_parser.add_argument('--datadir', type=str, default='', help='the data directory (default: ./data)')
    cmdline_parser.add_argument("--augmented", help="use data loaders with augmentation", action="store_true")
    cmdline_parser.add_argument("--preprocessed", help="folder with preprocessed datasets for each epoch")

    # load/save weights
    cmdline_parser.add_argument('--init-weights', type=str, default='None', help='The weights for initalizing the layers')
    cmdline_parser.add_argument('--save-weights', type=str, help='Save weights and bias to a file in .npz format')
    cmdline_parser.add_argument('--load-weights', type=str, help='Load weights and bias from a file in .npz format')

    # print options
    cmdline_parser.add_argument("--precision", help="The precision used for printing matrices", type=int, default=8)
    cmdline_parser.add_argument("--edgeitems", help="The edgeitems used for printing matrices", type=int, default=3)

    # multi-threading
    cmdline_parser.add_argument("--threads", help="The number of threads being used", type=int)

    # timer
    cmdline_parser.add_argument("--timer", help="Enable timer messages", action="store_true")

    return cmdline_parser


def check_command_line_arguments(args):
    if args.augmented and args.preprocessed:
        raise RuntimeError('the combination of --augmented and --preprocessed is unsupported')

    if args.densities and args.overall_density:
        raise RuntimeError('the options --densities and --overall-density cannot be used simultaneously')

    if not args.datadir and not args.preprocessed:
        raise RuntimeError('at least one of the options --datadir and --preprocessed must be set')


def print_command_line_arguments(args):
    print("python3 " + " ".join(shlex.quote(arg) if " " in arg else arg for arg in sys.argv) + '\n')


def initialize_frameworks(args):
    if args.seed:
        torch.manual_seed(args.seed)

    torch.set_printoptions(precision=args.precision, edgeitems=args.edgeitems, threshold=5, sci_mode=False, linewidth=160)

    # avoid 'Too many open files' error when using data loaders
    torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    cmdline_parser = make_argument_parser()
    args = cmdline_parser.parse_args()
    check_command_line_arguments(args)
    print_command_line_arguments(args)

    initialize_frameworks(args)

    if args.scheduler == 'multistep' and args.epochs <= 10:
        print('Setting gamma to 1.0')
        args.gamma = 1.0

    if args.datadir:
        if args.augmented:
            train_loader, test_loader = create_cifar10_augmented_dataloaders(args.batch_size, args.batch_size, args.datadir)
        else:
            train_loader, test_loader = create_cifar10_dataloaders(args.batch_size, args.batch_size, args.datadir)
    else:
        train_loader, test_loader = None, None

    linear_layer_sizes = [int(s) for s in args.sizes.split(',')]

    if args.densities:
        linear_layer_densities = list(float(d) for d in args.densities.split(','))
    elif args.overall_density:
        linear_layer_densities = compute_densities(args.overall_density, linear_layer_sizes)
    else:
        linear_layer_densities = [1.0] * (len(linear_layer_sizes) - 1)

    M = make_torch_model(args, linear_layer_sizes, linear_layer_densities)
    print('=== PyTorch model ===')
    print(M)

    if args.load_weights:
        M.load_weights_and_bias(args.load_weights)
    if args.save_weights:
        M.save_weights_and_bias(args.save_weights)

    if args.epochs > 0:
        print('\n=== Training PyTorch model ===')
        if args.preprocessed:
            train_torch_preprocessed(M, args.preprocessed, args.epochs, args.batch_size)
        else:
            train_torch(M, train_loader, test_loader, args.epochs)


if __name__ == '__main__':
    main()
