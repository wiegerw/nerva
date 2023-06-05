#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import numpy as np
import torch


def flatten_torch(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    return x.reshape(shape[0], -1)


def to_one_hot_torch(x: torch.Tensor) -> np.ndarray:
    return torch.nn.functional.one_hot(x, num_classes = 10).float()


def make_torch_scheduler(args, optimizer):
    if args.scheduler == 'constant':
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    elif args.scheduler == 'multistep':
        milestones = [int(args.epochs / 2), int(args.epochs * 3 / 4)]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma, last_epoch=-1)
    else:
        raise RuntimeError(f'Unknown scheduler {args.scheduler}')

