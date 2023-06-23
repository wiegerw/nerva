import re

from symbolic.optimizers import Optimizer, CompositeOptimizer, GradientDescentOptimizer
from symbolic.sympy.activation_functions import ActivationFunction, ReLUActivation, HyperbolicTangentActivation, \
    AllReLUActivation, LeakyReLUActivation, SReLUActivation
from symbolic.sympy.optimizers import MomentumOptimizer, NesterovOptimizer


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


def parse_optimizer(text: str,
                    layer
                   ) -> Optimizer:
    try:
        if text == 'GradientDescent':
            return CompositeOptimizer([GradientDescentOptimizer(layer.W, layer.DW), GradientDescentOptimizer(layer.b, layer.Db)])
        elif text.startswith('Momentum'):
            m = re.match(r'Momentum\((.*)\)$', text)
            mu = float(m.group(1))
            return CompositeOptimizer([MomentumOptimizer(layer.W, layer.DW, mu), MomentumOptimizer(layer.b, layer.Db, mu)])
        elif text.startswith('Nesterov'):
            m = re.match(r'Nesterov\((.*)\)$', text)
            mu = float(m.group(1))
            return CompositeOptimizer([NesterovOptimizer(layer.W, layer.DW, mu), NesterovOptimizer(layer.b, layer.Db, mu)])
    except:
        pass
    raise RuntimeError(f'Could not parse optimizer "{text}"')
