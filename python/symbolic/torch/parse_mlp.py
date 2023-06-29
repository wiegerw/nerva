import re
from typing import Callable, Any

from symbolic.optimizers import Optimizer, GradientDescentOptimizer
from symbolic.torch.activation_functions import ActivationFunction, ReLUActivation, HyperbolicTangentActivation, \
    AllReLUActivation, LeakyReLUActivation, SReLUActivation
from symbolic.torch.optimizers import MomentumOptimizer, NesterovOptimizer
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


def parse_optimizer(text: str) -> Callable[[Any, Any], Optimizer]:
    try:
        name, args = parse_function_call(text)
        if name == 'GradientDescent':
            return lambda x, Dx: GradientDescentOptimizer(x, Dx)
        elif name == 'Momentum':
            mu = float(args['mu'])
            return lambda x, Dx: MomentumOptimizer(x, Dx, mu)
        elif name == 'Nesterov':
            mu = float(args['mu'])
            return lambda x, Dx: NesterovOptimizer(x, Dx, mu)
    except:
        pass
    raise RuntimeError(f'Could not parse optimizer "{text}"')
