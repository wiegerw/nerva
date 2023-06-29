# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from symbolic.optimizers import Optimizer, CompositeOptimizer, GradientDescentOptimizer
from symbolic.sympy.activation_functions import ActivationFunction, ReLUActivation, HyperbolicTangentActivation, \
    AllReLUActivation, LeakyReLUActivation, SReLUActivation
from symbolic.sympy.optimizers import MomentumOptimizer, NesterovOptimizer
from symbolic.utilities import parse_function_call


def parse_activation(text: str) -> ActivationFunction:
    try:
        name, args = parse_function_call(text)
        if name == 'ReLU':
            return ReLUActivation()
        elif name == 'HyperbolicTangent':
            return HyperbolicTangentActivation()
        elif name == 'AllReLU':
            alpha = args['alpha']
            return AllReLUActivation(alpha)
        elif name == 'LeakyReLU':
            alpha = args['alpha']
            return LeakyReLUActivation(alpha)
        elif name == 'SReLU':
            al = float(args.get('al', 0))
            tl = float(args.get('tl', 0))
            ar = float(args.get('ar', 0))
            tr = float(args.get('tr', 1))
            return SReLUActivation(al, tl, ar, tr)
    except:
        pass
    raise RuntimeError(f'Could not parse activation "{text}"')


def parse_optimizer(text: str, layer) -> Optimizer:
    try:
        name, args = parse_function_call(text)
        if name == 'GradientDescent':
            return CompositeOptimizer([GradientDescentOptimizer(layer.W, layer.DW), GradientDescentOptimizer(layer.b, layer.Db)])
        elif name == 'Momentum':
            mu = float(args['mu'])
            return CompositeOptimizer([MomentumOptimizer(layer.W, layer.DW, mu), MomentumOptimizer(layer.b, layer.Db, mu)])
        elif name == 'Nesterov':
            mu = float(args['mu'])
            return CompositeOptimizer([NesterovOptimizer(layer.W, layer.DW, mu), NesterovOptimizer(layer.b, layer.Db, mu)])
    except:
        pass
    raise RuntimeError(f'Could not parse optimizer "{text}"')
