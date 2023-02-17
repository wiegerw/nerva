import time
from typing import List, Tuple
from sparselearning.logger import Logger
from nerva.activation import ReLU, NoActivation
from nerva.dataset import DataSet
from nerva.layers import Sequential, Dense, Sparse
from nerva.learning_rate import MultiStepLRScheduler
from nerva.loss import SoftmaxCrossEntropyLoss
from nerva.optimizers import Optimizer, GradientDescent, Momentum, Nesterov
from nerva.weights import Xavier


def log_test_results(log, n, correct, test_loss):
    log(f'Test evaluation: Loss: {test_loss / n:.6f}, Accuracy: {correct}/{n} ({100. * correct / float(n):.3f}%)\n')


def log_training_results(log, epoch, n, N, k, K, correct, train_loss):
    log(f'Train Epoch: {epoch} [{n}/{N} ({float(100 * k / K):.0f}%)]\tLoss: {train_loss / n:.6f} Accuracy: {correct}/{n} ({100. * correct / float(n):.3f}%)')


def log_model_parameters(log, model, args):
    log(str(model))
    log('=' * 60)
    log(args.model)
    log('=' * 60)
    log('Prune mode: {0}'.format(args.prune))
    log('Growth mode: {0}'.format(args.growth))
    log('Redistribution mode: {0}'.format(args.redistribution))
    log('=' * 60)


def make_optimizer(momentum=0.0, nesterov=True) -> Optimizer:
    if nesterov:
        return Nesterov(momentum)
    elif momentum > 0.0:
        return Momentum(momentum)
    else:
        return GradientDescent()


def compute_sparse_layer_densities(density: float, layer_shapes: List[Tuple[int, int]], erk_power_scale: float = 1.0):
    n = len(layer_shapes)  # the number of layers
    total_params = sum(rows * columns for (rows, columns) in layer_shapes)

    dense_layers = set()
    while True:
        divisor = 0
        rhs = 0
        raw_probabilities = [0.0] * n
        for i, (rows, columns) in enumerate(layer_shapes):
            n_param = rows * columns
            n_zeros = n_param * (1 - density)
            n_ones = n_param * density
            if i in dense_layers:
                rhs -= n_zeros
            else:
                rhs += n_ones
                raw_probabilities[i] = ((rows + columns) / (rows * columns)) ** erk_power_scale
                divisor += raw_probabilities[i] * n_param
        epsilon = rhs / divisor
        max_prob = max(raw_probabilities)
        max_prob_one = max_prob * epsilon
        if max_prob_one > 1:
            for j, mask_raw_prob in enumerate(raw_probabilities):
                if mask_raw_prob == max_prob:
                    #print(f"Sparsity of layer:{j} had to be set to 0.")
                    dense_layers.add(j)
        else:
            break

    # Compute the densities
    densities = [0.0] * n
    total_nonzero = 0.0
    for i, (rows, columns) in enumerate(layer_shapes):
        n_param = rows * columns
        if i in dense_layers:
            densities[i] = 1.0
        else:
            probability_one = epsilon * raw_probabilities[i]
            densities[i] = probability_one
        #print(f"layer: {i}, shape: {(rows,columns)}, density: {densities[i]}")
        total_nonzero += densities[i] * n_param
    print(f"Overall sparsity {total_nonzero / total_params:.4f}")
    return densities


class MLP_CIFAR10(Sequential):
    def __init__(self, density, optimizer: Optimizer):
        super().__init__()
        shapes = [(3072, 1024), (1024, 512), (512, 10)]

        densities = compute_sparse_layer_densities(density, shapes)
        sparsities = [1.0 - x for x in densities]
        layer_sizes = [1024, 512, 10]
        activations = [ReLU(), ReLU(), NoActivation()]

        for (sparsity, size, activation) in zip(sparsities, layer_sizes, activations):
            if sparsity == 0.0:
                 self.add(Dense(size, activation=activation, optimizer=optimizer, weight_initializer=Xavier()))
            else:
                self.add(Sparse(size, sparsity, activation=activation, optimizer=optimizer, weight_initializer=Xavier()))

def make_model(name: str, sparsity, optimizer: Optimizer) -> Sequential:
    if name == 'mlp_cifar10':
        return MLP_CIFAR10(1.0 - sparsity, optimizer)
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


def test_model(model, loss_fn, dataset, batch_size, log: Logger):
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

    log_test_results(log, n, correct, test_loss)
    return correct / float(n)


def train_model(model, loss_fn, dataset, lr_scheduler, device, epochs, batch_size, log_interval, log):
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
                log_training_results(log, epoch, n, N, k, K, correct, train_loss)

        log(f'Current learning rate: {eta:.4f}. Time taken for epoch: {time.time() - t0:.2f} seconds.')
        test_model(model, loss_fn, dataset, batch_size, log)


def train_and_test(i, args, device, Xtrain, Ttrain, Xtest, Ttest, log: Logger):
    dataset = DataSet(Xtrain, Ttrain, Xtest, Ttest)
    sparsity = 1.0 - args.density
    optimizer = make_optimizer(args.momentum, nesterov=True)
    model = make_model(args.model, sparsity, optimizer)
    model.compile(3072, args.batch_size)
    milestones = [int(args.epochs / 2) * args.multiplier, int(args.epochs * 3 / 4) * args.multiplier]
    lr_scheduler = MultiStepLRScheduler(args.lr, milestones, 0.1)
    log_model_parameters(log, model, args)
    epochs = args.epochs * args.multiplier
    loss_fn = SoftmaxCrossEntropyLoss()
    train_model(model, loss_fn, dataset, lr_scheduler, device, epochs, args.batch_size, args.log_interval, log)
    log("\nIteration end: {0}/{1}\n".format(i + 1, args.iters))
