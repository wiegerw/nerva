import re
from typing import Callable, Any

from symbolic.jax.activation_functions import SReLUActivation, ActivationFunction, ReLUActivation, \
    HyperbolicTangentActivation, AllReLUActivation, LeakyReLUActivation
from symbolic.jax.optimizers import Optimizer, GradientDescentOptimizer, MomentumOptimizer, NesterovOptimizer
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
        elif text.startswith('LeakyReLU'):
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
