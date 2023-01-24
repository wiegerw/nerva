from __future__ import print_function

import os
import time
from typing import Tuple, Optional

import torch
from torch import nn as nn, optim as optim

from sparselearning.core import Masking, CosineDecay
from sparselearning.logger import Logger
from sparselearning.models import MLP_CIFAR10
from sparselearning.train_nerva import log_training_results, log_test_results


def test_model(model, loss_fn, device, test_loader, log: Logger):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            model.t = target
            output = model(data)
            # test_loss += loss_fn(output, target, reduction='sum').item() # sum up batch loss
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

    test_loss /= float(n)

    N = len(test_loader.dataset)
    log_test_results(log, n, correct, test_loss * N)
    return correct / float(n)


def train_model(model, mask, loss_fn, train_loader, test_loader, lr_scheduler, optimizer, device, epochs, batch_size, log_interval, log):
    model.train()  # Set model in training mode

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = 0
        correct = 0
        n = 0

        # global gradient_norm
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

            loss.backward()

            if mask is not None:
                mask.step()
            else:
                optimizer.step()

            k = batch_idx + 1
            K = len(train_loader)
            N = len(train_loader.dataset)
            if k != 0 and k % log_interval == 0:
                log_training_results(log, epoch, n, N, k, K, correct, train_loss * batch_size)

        log(f'Current learning rate: {optimizer.param_groups[0]["lr"]:.4f}. Time taken for epoch: {time.time() - t0:.2f} seconds.')
        test_model(model, loss_fn, device, test_loader, log)

        lr_scheduler.step()


def make_model(name: str, device) -> torch.nn.Module:
    if name == 'mlp_cifar10':
        return MLP_CIFAR10().to(device)
    raise RuntimeError(f'Unknown model {name}')


def make_optimizer(name, model: nn.Module, lr: float, momentum: float, weight_decay: float, nesterov: bool) -> torch.optim.Optimizer:
    if name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    elif name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise RuntimeError(f'Unknown optimizer: {name}')


def make_mask(args,
              model: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              train_loader
             ) -> Optional[Masking]:
    mask = None
    if args.sparse:
        prune_interval = args.update_frequency if not args.fix else 0
        decay = CosineDecay(args.prune_rate, len(train_loader) * (args.epochs * args.multiplier))
        mask = Masking(optimizer, prune_rate=args.prune_rate, prune_mode=args.prune, prune_rate_decay=decay,
                       growth_mode=args.growth,
                       redistribution_mode=args.redistribution, train_loader=train_loader,
                       prune_interval=prune_interval)
        mask.add_module(model, sparse_init=args.sparse_init, density=args.density)
    return mask


def train_and_test(i, args, device, train_loader, test_loader, log):
    log("\nIteration start: {0}/{1}\n".format(i + 1, args.iters))
    model = make_model(args.model, device)
    optimizer = make_optimizer(args.optimizer, model, args.lr, args.momentum, args.l2, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[int(args.epochs / 2) * args.multiplier, int(args.epochs * 3 / 4) * args.multiplier],
                                                        last_epoch=-1)
    mask = make_mask(args, model, optimizer, train_loader)
    log(model)
    log('=' * 60)
    log(args.model)
    log('=' * 60)
    log('Prune mode: {0}'.format(args.prune))
    log('Growth mode: {0}'.format(args.growth))
    log('Redistribution mode: {0}'.format(args.redistribution))
    log('=' * 60)
    # create output folder
    output_path = './save/' + str(args.model) + '/' + str(args.data) + '/' + str(args.sparse_init) + '/' + str(args.seed)
    output_folder = os.path.join(output_path, f'sparsity{1 - args.density}' if args.sparse else 'dense')
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    epochs = args.epochs * args.multiplier
    loss_fn = nn.CrossEntropyLoss()
    train_model(model, mask, loss_fn, train_loader, test_loader, lr_scheduler, optimizer, device, epochs, args.batch_size, args.log_interval, log)
    model.load_state_dict(torch.load(os.path.join(output_folder, 'model_final.pth'))['state_dict'])
    log("\nIteration end: {0}/{1}\n".format(i + 1, args.iters))
