#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse
import symbolic.numpy.training_colwise
import symbolic.numpy.training_rowwise
import symbolic.tensorflow.training_colwise
import symbolic.tensorflow.training_rowwise
import symbolic.torch.training_colwise
import symbolic.torch.training_rowwise
import symbolic.jax.training_colwise
import symbolic.jax.training_rowwise
from symbolic.numpy.utilities import set_numpy_options
from symbolic.tensorflow.utilities import set_tensorflow_options
from symbolic.torch.utilities import set_torch_options


def make_argument_parser():
    cmdline_parser = argparse.ArgumentParser()

    # model parameters
    cmdline_parser.add_argument('--sizes', type=str, default='3072,128,64,10', help='A comma separated list of layer sizes, e.g. "3072,128,64,10".')
    cmdline_parser.add_argument('--layers', type=str, help='A semi-colon separated lists of layer specifications.')

    # learning rate
    cmdline_parser.add_argument("--learning-rate", type=str, help="The learning rate scheduler")

    # loss function
    cmdline_parser.add_argument('--loss', type=str, help='The loss function')

    # training parameters
    cmdline_parser.add_argument("--epochs", help="The number of epochs", type=int, default=100)
    cmdline_parser.add_argument("--batch-size", help="The batch size", type=int, default=1)

    # optimizer
    cmdline_parser.add_argument("--optimizers", type=str, help="The optimizers (GradientDescent, Momentum(<mu>), Nesterov(<mu>))", default="GradientDescent")

    # dataset
    cmdline_parser.add_argument('--dataset', type=str, help='The .npz file containing train and test data')

    # weights
    cmdline_parser.add_argument('--weights', type=str, help='The .npz file containing weights and bias values')
    cmdline_parser.add_argument('--init-weights', type=str, help='The weight initializers (Xavier, XavierNormalized, He)')

    # logging
    cmdline_parser.add_argument("--debug", help="Log intermediate values", action="store_true")

    # framework
    cmdline_parser.add_argument("--numpy", help="Train using NumPy", action="store_true")
    cmdline_parser.add_argument("--torch", help="Train using PyTorch", action="store_true")
    cmdline_parser.add_argument("--tensorflow", help="Train using Tensorflow", action="store_true")
    cmdline_parser.add_argument("--jax", help="Train using JAX", action="store_true")

    # layout
    cmdline_parser.add_argument("--colwise", help="Train using data with column layout", action="store_true")
    cmdline_parser.add_argument("--rowwise", help="Train using data with row layout", action="store_true")

    return cmdline_parser


def print_header(header: str):
    print('===========================================')
    print(f'               {header}')
    print('===========================================')


def main():
    cmdline_parser = make_argument_parser()
    args = cmdline_parser.parse_args()

    set_tensorflow_options()
    set_torch_options()
    set_numpy_options()

    linear_layer_sizes = [int(s) for s in args.sizes.split(',')]
    layer_specifications = args.layers.split(';')
    linear_layer_weight_initializers = args.init_weights.split(',')
    linear_layer_optimizers = args.optimizers.split(';')

    train_functions = {}
    if args.numpy and args.colwise:
        train_functions['numpy-colwise'] = symbolic.numpy.training_colwise.train
    if args.numpy and args.rowwise:
        train_functions['numpy-rowwise'] = symbolic.numpy.training_rowwise.train
    if args.tensorflow and args.colwise:
        train_functions['tensorflow-colwise'] = symbolic.tensorflow.training_colwise.train
    if args.tensorflow and args.rowwise:
        train_functions['tensorflow-rowwise'] = symbolic.tensorflow.training_rowwise.train
    if args.torch and args.colwise:
        train_functions['torch-colwise'] = symbolic.torch.training_colwise.train
    if args.torch and args.rowwise:
        train_functions['torch-rowwise'] = symbolic.torch.training_rowwise.train
    if args.jax and args.colwise:
        train_functions['jax-colwise'] = symbolic.jax.training_colwise.train
    if args.jax and args.rowwise:
        train_functions['jax-rowwise'] = symbolic.jax.training_rowwise.train

    for header, train in train_functions.items():
        print_header(header)
        train(layer_specifications,
              linear_layer_sizes,
              linear_layer_optimizers,
              linear_layer_weight_initializers,
              args.batch_size,
              args.epochs,
              args.loss,
              args.learning_rate,
              args.weights,
              args.dataset,
              args.debug
             )

if __name__ == '__main__':
    main()
