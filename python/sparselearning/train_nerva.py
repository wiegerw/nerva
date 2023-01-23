import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from sparselearning.logger import Logger
from sparselearning.train_pytorch import evaluate
from nerva.activation import ReLU, NoActivation, AllReLU
from nerva.dataset import DataSet
from nerva.layers import Sequential, Dense, Dropout, Sparse, BatchNormalization
from nerva.learning_rate import ConstantScheduler
from nerva.loss import SoftmaxCrossEntropyLoss
from nerva.optimizers import Optimizer, GradientDescent, Momentum, Nesterov
from nerva.training import minibatch_gradient_descent, minibatch_gradient_descent_python, SGDOptions, compute_accuracy, compute_statistics
from nerva.utilities import RandomNumberGenerator, set_num_threads, StopWatch
from nerva.weights import Xavier


def make_optimizer(momentum=0.0, nesterov=True) -> Optimizer:
    if nesterov:
        return Nesterov(momentum)
    elif momentum > 0.0:
        return Momentum(momentum)
    else:
        return GradientDescent()


class MLP_CIFAR10(Sequential):
    def __init__(self, sparsity, optimizer: Optimizer):
        super().__init__()
        self.add(Sparse(1024, sparsity, activation=ReLU(), optimizer=optimizer, weight_initializer=Xavier()))
        self.add(Sparse(512, sparsity, activation=ReLU(), optimizer=optimizer, weight_initializer=Xavier()))
        self.add(Sparse(10, sparsity, activation=NoActivation(), optimizer=optimizer, weight_initializer=Xavier()))


def make_model(name: str, sparsity, optimizer: Optimizer) -> Sequential:
    if name == 'mlp_cifar10':
        return MLP_CIFAR10(sparsity, optimizer)
    raise RuntimeError(f'Unknown model {name}')


def flatten(X: torch.Tensor):
    shape = X.shape
    return X.reshape(shape[0], -1)


def train_model(model, loss_fn, train_loader, lr_scheduler, device, epochs, batch_size, log_interval, log):
    for epoch in range(1, epochs + 1):
        eta = lr_scheduler(epoch)  # update the learning rate at the start of each epoch

        t0 = time.time()
        train_loss = 0
        correct = 0
        n = 0

        # global gradient_norm
        for batch_idx, (data, target) in enumerate(train_loader):
            X = flatten(data.to(device)).T  # torch.Tensor
            T = F.one_hot(target.to(device) % 10).T  # torch.Tensor
            Y = torch.Tensor(model.feedforward(X))
            # print('X', X.shape, X)
            # print('Y', Y.shape, Y)
            # print('T', T.shape, T)
            loss = loss_fn.value(Y, T)

            train_loss += loss
            pred = Y.argmax(dim=0, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

            dY = loss_fn.gradient(Y, T) / batch_size  # pytorch uses this division
            # print('dY', dY.shape, dY)
            model.backpropagate(Y, dY)
            model.optimize(eta)

            if batch_idx % log_interval == 0:
                log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {}/{} ({:.3f}% '.format(
                    epoch, batch_idx * len(data), len(train_loader) * batch_size,
                           100. * batch_idx / len(train_loader), loss, correct, n, 100. * correct / float(n)))

        # training summary
        log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format('Training summary', train_loss / batch_idx, correct, n, 100. * correct / float(n)))

        # lr_scheduler.step()
        log('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.format(eta, time.time() - t0))


def train_and_test(i, args, device, train_loader, test_loader, log: Logger):
    rng = RandomNumberGenerator(123)
    sparsity = 1.0 - args.density
    optimizer = make_optimizer(args.momentum, nesterov=True)
    model = make_model(args.model, sparsity, optimizer)
    model.compile(3072, args.batch_size, rng)
    lr_scheduler = ConstantScheduler(args.lr)
    log(str(model))
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
    loss_fn = SoftmaxCrossEntropyLoss()
    train_model(model, loss_fn, train_loader, lr_scheduler, device, epochs, args.batch_size, args.log_interval, log)
    # print('Testing model')
    # model.load_state_dict(torch.load(os.path.join(output_folder, 'model_final.pth'))['state_dict'])
    # evaluate(model, loss_fn, device, test_loader, is_test_set=True)
    # log("\nIteration end: {0}/{1}\n".format(i + 1, args.iters))
