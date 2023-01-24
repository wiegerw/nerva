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
from nerva.utilities import set_num_threads, StopWatch
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


# TODO: find an efficient implementation for this
def correct_predictions(Y, T):
    # https://stackoverflow.com/questions/74501160/error-using-np-argmax-when-applying-keepdims
    # unfortunately the suggested solutions do not work
    a = Y.argmax(axis=0)

    total_correct = 0
    for i, value in enumerate(a):
        if T[value, i] == 1:
            total_correct += 1

    return total_correct


def train_model(model, loss_fn, dataset, lr_scheduler, device, epochs, batch_size, log_interval, log):
    def log_results(n, N, k, K, correct, train_loss):
        log(f'Train Epoch: {epoch} [{n}/{N} ({float(100 * k / K):.0f}%)]\tLoss: {train_loss / n:.6f} Accuracy: {correct}/{n} ({100. * correct / float(n):.3f}%)')

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        N = dataset.Xtrain.shape[1]  # the number of examples
        I = list(range(N))
        K = N // batch_size  # the number of batches
        # if shuffle: random.shuffle(I)

        train_loss = 0
        correct = 0
        n = 0

        eta = lr_scheduler(epoch)  # update the learning rate at the start of each epoch
        for k in range(1, K + 1):
            n += batch_size
            batch = I[(k - 1) * batch_size: k * batch_size]
            X = dataset.Xtrain[:, batch]
            T = dataset.Ttrain[:, batch]
            Y = model.feedforward(X)
            correct += correct_predictions(Y, T)
            train_loss += loss_fn.value(Y, T)
            dY = loss_fn.gradient(Y, T) / batch_size
            model.backpropagate(Y, dY)
            model.optimize(eta)

            if k != 0 and k % log_interval == 0:
                log_results(n, N, k, K, correct, train_loss)

        log(f'Current learning rate: {eta:.4f}. Time taken for epoch: {time.time() - t0:.2f} seconds.\n')


def evaluate_model(model, loss_fn, dataset, batch_size, log: Logger):
    test_loss = 0
    correct = 0
    n = 0

    N = dataset.Xtest.shape[1]  # the number of examples
    I = list(range(N))
    K = N // batch_size  # the number of batches

    for k in range(1, K + 1):
        n += batch_size
        batch = I[(k - 1) * batch_size: k * batch_size]
        X = dataset.Xtest[:, batch]
        T = dataset.Ttest[:, batch]
        Y = model.feedforward(X)
        correct += correct_predictions(Y, T)
        test_loss += loss_fn.value(Y, T)

    test_loss /= float(n)

    log(f'\nTest evaluation: Average loss: {test_loss:.4f}, Accuracy: {correct}/{n} ({100. * correct / float(n):.3f}%)\n')
    return correct / float(n)


def train_and_test(i, args, device, Xtrain, Ttrain, Xtest, Ttest, log: Logger):
    dataset = DataSet(Xtrain, Ttrain, Xtest, Ttest)
    sparsity = 1.0 - args.density
    optimizer = make_optimizer(args.momentum, nesterov=True)
    model = make_model(args.model, sparsity, optimizer)
    model.compile(3072, args.batch_size)
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
    train_model(model, loss_fn, dataset, lr_scheduler, device, epochs, args.batch_size, args.log_interval, log)
    evaluate_model(model, loss_fn, dataset, args.batch_size, log)
