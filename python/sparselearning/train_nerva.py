import os
from sparselearning.logger import Logger
from nerva.activation import ReLU, NoActivation, AllReLU
from nerva.dataset import DataSet
from nerva.layers import Sequential, Dense, Dropout, Sparse, BatchNormalization
from nerva.learning_rate import ConstantScheduler
from nerva.loss import SoftmaxCrossEntropyLoss
from nerva.optimizers import Optimizer, GradientDescent, Momentum, Nesterov
from nerva.training import minibatch_gradient_descent, minibatch_gradient_descent_python, SGDOptions, compute_accuracy, compute_statistics
from nerva.utilities import RandomNumberGenerator, set_num_threads, StopWatch
from nerva.weights import Weights


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
        self.add(Sparse(1024, sparsity, activation=ReLU(), optimizer=optimizer, weight_initializer=Weights.Xavier))
        self.add(Sparse(512, sparsity, activation=ReLU(), optimizer=optimizer, weight_initializer=Weights.Xavier))
        self.add(Sparse(10, sparsity, activation=NoActivation(), optimizer=optimizer, weight_initializer=Weights.Xavier))


def make_model(name: str, sparsity, optimizer: Optimizer) -> Sequential:
    if name == 'mlp_cifar10':
        return MLP_CIFAR10(sparsity, optimizer)
    raise RuntimeError(f'Unknown model {name}')


def train_and_test(i, args, device, train_loader, test_loader, log: Logger):
    sparsity = 1.0 - args.density
    optimizer = make_optimizer(args.momentum, args.nesterov)
    model = make_model(args.model, sparsity, optimizer)
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
    train_model(model, loss_fn, train_loader, lr_scheduler, optimizer, device, epochs, args.batch_size, args.log_interval)
    print('Testing model')
    model.load_state_dict(torch.load(os.path.join(output_folder, 'model_final.pth'))['state_dict'])
    evaluate(model, loss_fn, device, test_loader, is_test_set=True)
    log("\nIteration end: {0}/{1}\n".format(i + 1, args.iters))
