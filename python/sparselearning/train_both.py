import torch
import torch.nn as nn
from sparselearning import train_pytorch


def train_and_test(i, args, device, train_loader, test_loader, log):
    log("\nIteration start: {0}/{1}\n".format(i + 1, args.iters))
    model = train_pytorch.make_model(args.model, device)
    optimizer = train_pytorch.make_optimizer(args.optimizer, model, args.lr, args.momentum, args.l2, nesterov=True)
    milestones = [int(args.epochs / 2) * args.multiplier, int(args.epochs * 3 / 4) * args.multiplier]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, last_epoch=-1)
    mask = train_pytorch.make_mask(args, model, optimizer, train_loader)
    log(model)
    log('=' * 60)
    log(args.model)
    log('=' * 60)
    log('Prune mode: {0}'.format(args.prune))
    log('Growth mode: {0}'.format(args.growth))
    log('Redistribution mode: {0}'.format(args.redistribution))
    log('=' * 60)
    epochs = args.epochs * args.multiplier
    loss_fn = nn.CrossEntropyLoss()
    train_pytorch.train_model(model, mask, loss_fn, train_loader, test_loader, lr_scheduler, optimizer, device, epochs, args.batch_size, args.log_interval, log)
    log("\nIteration end: {0}/{1}\n".format(i + 1, args.iters))
