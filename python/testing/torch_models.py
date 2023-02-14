import numpy as np
import torch
from sparselearning.core import Masking


def flatten_torch(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    return x.reshape(shape[0], -1)


def to_one_hot_torch(x: torch.Tensor) -> np.ndarray:
    return to_numpy(torch.nn.functional.one_hot(x, num_classes = 10).float())


def make_torch_scheduler(args, optimizer):
    if args.scheduler == 'constant':
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    elif args.scheduler == 'multistep':
        milestones = [int(args.epochs / 2), int(args.epochs * 3 / 4)]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma, last_epoch=-1)
    else:
        raise RuntimeError(f'Unknown scheduler {args.scheduler}')


def make_torch_mask(model: torch.nn.Module,
                    density,
                    sparse_init='ER',
                    ) -> Masking:
    mask = Masking(model.optimizer,
                   prune_rate=0.5,
                   prune_mode='none',
                   prune_rate_decay=None,
                   prune_interval=0,
                   growth_mode='random',
                   redistribution_mode='none',
                   )
    mask.add_module(model, sparse_init=sparse_init, density=density)
    return mask
