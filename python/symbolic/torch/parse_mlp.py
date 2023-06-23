import re
from typing import Callable, Any

from symbolic.optimizers import Optimizer, GradientDescentOptimizer
from symbolic.torch.activation_functions import ActivationFunction, ReLUActivation, HyperbolicTangentActivation, \
    AllReLUActivation, LeakyReLUActivation, SReLUActivation
from symbolic.torch.optimizers import MomentumOptimizer, NesterovOptimizer


def parse_activation(text: str) -> ActivationFunction:
    try:
        if text == 'ReLU':
            return ReLUActivation()
        elif text == 'HyperbolicTangent':
            return HyperbolicTangentActivation()
        elif text.startswith('AllReLU'):
            m = re.match(r'AllReLU\((.*)\)$', text)
            alpha = float(m.group(1))
            return AllReLUActivation(alpha)
        elif text.startswith('LeakyReLU'):
            m = re.match(r'LeakyReLU\((.*)\)$', text)
            alpha = float(m.group(1))
            return LeakyReLUActivation(alpha)
    except:
        pass
    raise RuntimeError(f'Could not parse activation "{text}"')


def parse_srelu_activation(text: str) -> SReLUActivation:
    try:
        if text == 'SReLU':
            return SReLUActivation()
        else:
            m = re.match(r'SReLU\(([^,]*),([^,]*),([^,]*),([^,]*)\)$', text)
            al = float(m.group(1))
            tl = float(m.group(2))
            ar = float(m.group(3))
            tr = float(m.group(4))
            return SReLUActivation(al, tl, ar, tr)
    except:
        pass
    raise RuntimeError(f'Could not parse SReLU activation "{text}"')


def parse_optimizer(text: str) -> Callable[[Any, Any], Optimizer]:
    try:
        if text == 'GradientDescent':
            return lambda x, Dx: GradientDescentOptimizer(x, Dx)
        elif text.startswith('Momentum'):
            m = re.match(r'Momentum\((.*)\)$', text)
            mu = float(m.group(1))
            return lambda x, Dx: MomentumOptimizer(x, Dx, mu)
        elif text.startswith('Nesterov'):
            m = re.match(r'Nesterov\((.*)\)$', text)
            mu = float(m.group(1))
            return lambda x, Dx: NesterovOptimizer(x, Dx, mu)
    except:
        pass
    raise RuntimeError(f'Could not parse optimizer "{text}"')
