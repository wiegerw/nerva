# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse


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

    return cmdline_parser
