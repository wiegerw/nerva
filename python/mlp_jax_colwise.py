#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from mlp_utilities import make_argument_parser
from mlps.nerva_jax.training_colwise import train


def main():
    cmdline_parser = make_argument_parser()
    args = cmdline_parser.parse_args()

    linear_layer_sizes = [int(s) for s in args.sizes.split(',')]
    layer_specifications = args.layers.split(';')
    linear_layer_weight_initializers = args.init_weights.split(',')
    linear_layer_optimizers = args.optimizers.split(';')

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
