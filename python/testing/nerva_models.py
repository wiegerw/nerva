import nerva.layers
import nerva.learning_rate
import nerva.optimizers


def make_nerva_optimizer(momentum=0.0, nesterov=False) -> nerva.optimizers.Optimizer:
    if nesterov:
        return nerva.optimizers.Nesterov(momentum)
    elif momentum > 0.0:
        return nerva.optimizers.Momentum(momentum)
    else:
        return nerva.optimizers.GradientDescent()


def make_nerva_scheduler(args):
    if args.scheduler == 'constant':
        return nerva.learning_rate.ConstantScheduler(args.lr)
    elif args.scheduler == 'multistep':
        milestones = [int(args.epochs / 2), int(args.epochs * 3 / 4)]
        return nerva.learning_rate.MultiStepLRScheduler(args.lr, milestones, args.gamma)
    else:
        raise RuntimeError(f'Unknown scheduler {args.scheduler}')
