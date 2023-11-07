# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import Any, Callable

from mlps.nerva_jax.activation_functions import ActivationFunction, AllReLUActivation, HyperbolicTangentActivation, \
    LeakyReLUActivation, ReLUActivation, SReLUActivation, SigmoidActivation
from mlps.nerva_jax.optimizers import GradientDescentOptimizer, MomentumOptimizer, NesterovOptimizer, Optimizer
from mlps.nerva_jax.utilities import parse_function_call


def parse_activation(text: str) -> ActivationFunction:
    try:
        name, args = parse_function_call(text)
        if name == 'ReLU':
            return ReLUActivation()
        elif name == 'Sigmoid':
            return SigmoidActivation()
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


def parse_optimizer(text: str) -> Callable[[Any, str, str], Optimizer]:
    try:
        name, args = parse_function_call(text)
        if name == 'GradientDescent':
            return lambda obj, attr_x, attr_Dx: GradientDescentOptimizer(obj, attr_x, attr_Dx)
        elif name == 'Momentum':
            mu = float(args['mu'])
            return lambda obj, attr_x, attr_Dx: MomentumOptimizer(obj, attr_x, attr_Dx, mu)
        elif name == 'Nesterov':
            mu = float(args['mu'])
            return lambda obj, attr_x, attr_Dx: NesterovOptimizer(obj, attr_x, attr_Dx, mu)
    except:
        pass
    raise RuntimeError(f'Could not parse optimizer "{text}"')
