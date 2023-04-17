#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse
import shlex
import sys
from pathlib import Path
from typing import List

import torch

import nerva.activation
import nerva.dataset
import nerva.layers
import nerva.learning_rate
import nerva.loss
import nerva.optimizers
import nerva.random
import nerva.utilities
import nerva.weights
import nervalib
from nerva.pruning import PruneFunction, GrowFunction, PruneGrow, parse_prune_function, parse_grow_function
from nerva.training import StochasticGradientDescentAlgorithm, SGD_Options, compute_densities
from testing.datasets import create_cifar10_augmented_dataloaders, create_cifar10_dataloaders, create_npz_dataloaders
from testing.models import MLPNerva


def parse_init_weights(text: str, linear_layer_count: int) -> List[nerva.weights.WeightInitializer]:
    words = text.strip().split(';')
    n = linear_layer_count

    if len(words) == 1:
        init = nerva.weights.parse_weight_initializer(words[0])
        return [init] * n

    if len(words) != n:
        raise RuntimeError(f'the number of weight initializers ({len(words)}) does not match with the number of linear layers ({n})')

    return [nerva.weights.parse_weight_initializer(word) for word in words]


def parse_optimizers(text: str, linear_layer_count: int) -> List[nerva.optimizers.Optimizer]:
    words = text.strip().split(';')
    n = linear_layer_count

    if len(words) == 0:
        optimizer = nerva.optimizers.GradientDescent()
        return [optimizer] * n

    if len(words) == 1:
        optimizer = nerva.optimizers.parse_optimizer(words[0])
        return [optimizer] * n

    if len(words) != n:
        raise RuntimeError(f'the number of weight initializers ({len(words)}) does not match with the number of linear layers ({n})')

    return [nerva.optimizers.parse_optimizer(word) for word in words]


def make_argument_parser():
    cmdline_parser = argparse.ArgumentParser()

    # randomness
    cmdline_parser.add_argument("--seed", help="The initial seed of the random generator", type=int)

    # model parameters
    cmdline_parser.add_argument('--sizes', type=str, default='3072,128,64,10', help='A comma separated list of layer sizes, e.g. "3072,128,64,10".')
    cmdline_parser.add_argument('--densities', type=str, help='A comma separated list of layer densities, e.g. "0.05,0.05,1.0".')
    cmdline_parser.add_argument('--overall-density', type=float, default=1.0, help='The overall density of the layers.')
    cmdline_parser.add_argument('--layers', type=str, help='A semi-colon separated lists of layers.')

    # learning rate
    cmdline_parser.add_argument("--learning-rate", type=str, help="The learning rate scheduler")

    # loss function
    cmdline_parser.add_argument('--loss', type=str, help='The loss function')

    # training
    cmdline_parser.add_argument("--epochs", help="The number of epochs", type=int, default=100)
    cmdline_parser.add_argument("--batch-size", help="The batch size", type=int, default=1)

    # optimizer
    cmdline_parser.add_argument("--optimizers", type=str, help="The optimizer (GradientDescent, Momentum(<mu>), Nesterov(<mu>))", default="GradientDescent")

    # dataset
    cmdline_parser.add_argument('--datadir', type=str, default='', help='the data directory (default: ./data)')
    cmdline_parser.add_argument("--augmented", help="use data loaders with augmentation", action="store_true")
    cmdline_parser.add_argument("--preprocessed", help="folder with preprocessed datasets for each epoch")

    # load/save weights
    cmdline_parser.add_argument('--init-weights', type=str, default='None', help='The initial weights for the layers')
    cmdline_parser.add_argument('--save-weights', type=str, help='Save weights and bias to a file in .npz format')
    cmdline_parser.add_argument('--load-weights', type=str, help='Load weights and bias from a file in .npz format')

    # print options
    cmdline_parser.add_argument("--precision", help="The precision used for printing matrices", type=int, default=8)
    cmdline_parser.add_argument("--edgeitems", help="The edgeitems used for printing matrices", type=int, default=3)
    cmdline_parser.add_argument("--debug", help="print debug information", action="store_true")

    # pruning + growing (experimental!)
    cmdline_parser.add_argument("--prune", help="The pruning strategy: Magnitude(<rate>), SET(<rate>) or Threshold(<value>)", type=str)
    cmdline_parser.add_argument("--grow", help="The growing strategy: (default: Random)", type=str)
    cmdline_parser.add_argument('--grow-weights', type=str, help='The function used for growing weigths: Xavier, XavierNormalized, He, PyTorch, Zero')

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
        nerva.random.manual_seed(args.seed)

    if args.timer:
        nerva.utilities.global_timer_enable()

    torch.set_printoptions(precision=args.precision, edgeitems=args.edgeitems, threshold=5, sci_mode=False, linewidth=160)

    # avoid 'Too many open files' error when using data loaders
    torch.multiprocessing.set_sharing_strategy('file_system')


class SGD(StochasticGradientDescentAlgorithm):
    def __init__(self,
                 M: nerva.layers.Sequential,
                 train_loader,
                 test_loader,
                 options: SGD_Options,
                 loss: nerva.loss.LossFunction,
                 learning_rate: nerva.learning_rate.LearningRateScheduler,
                 preprocessed_dir: str,
                 prune: PruneFunction,
                 grow: GrowFunction
                ):
        super().__init__(M, train_loader, test_loader, options, loss, learning_rate)
        self.preprocessed_dir = preprocessed_dir
        self.regrow = PruneGrow(prune, grow) if prune else None

    def reload_data(self, epoch) -> None:
        """
        Reloads the dataset if a directory with preprocessed data was specified.
        """
        if self.preprocessed_dir:
            path = Path(self.preprocessed_dir) / f'epoch{epoch}.npz'
            self.train_loader, self.test_loader = create_npz_dataloaders(str(path), self.options.batch_size)

    def on_start_training(self) -> None:
        self.reload_data(0)

    def on_start_epoch(self, epoch):
        if epoch > 0:
            self.reload_data(epoch)

        if epoch > 0 and self.regrow:
            self.regrow(self.M)

        # TODO: renew dropout masks


def main():
    cmdline_parser = make_argument_parser()
    args = cmdline_parser.parse_args()
    check_command_line_arguments(args)
    print_command_line_arguments(args)

    initialize_frameworks(args)

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

    layer_specifications = args.layers.split(';')
    linear_layer_specifications = [spec for spec in layer_specifications if nervalib.is_linear_layer(spec)]
    linear_layer_weights = parse_init_weights(args.init_weights, len(linear_layer_sizes) - 1)
    linear_layer_optimizers = parse_optimizers(args.optimizers, len(linear_layer_sizes) - 1)

    loss = nerva.loss.parse_loss_function(args.loss)
    learning_rate = nerva.learning_rate.parse_learning_rate(args.learning_rate)
    activations = [nerva.activation.parse_activation(text) for text in linear_layer_specifications]

    M = MLPNerva(linear_layer_sizes,
                 linear_layer_densities,
                 linear_layer_optimizers,
                 linear_layer_weights,
                 activations,
                 loss,
                 learning_rate,
                 args.batch_size
                )

    print('=== Nerva python model ===')
    print(M)

    if args.load_weights:
        M.load_weights_and_bias(args.load_weights)

    if args.save_weights:
        M.save_weights_and_bias(args.save_weights)

    if args.epochs > 0:
        print('\n=== Training Nerva model ===')
        options = SGD_Options()
        options.epochs = args.epochs
        options.batch_size = args.batch_size
        options.shuffle = False
        options.statistics = True
        options.debug = args.debug
        options.gradient_step = 0
        prune = parse_prune_function(args.prune) if args.prune else None
        grow = parse_grow_function(args.grow, nerva.weights.parse_weight_initializer(args.grow_weights)) if args.grow else None
        algorithm = SGD(M, train_loader, test_loader, options, M.loss, M.learning_rate, args.preprocessed, prune, grow)
        algorithm.run()


if __name__ == '__main__':
    main()
