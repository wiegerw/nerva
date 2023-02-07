from timeit import default_timer as timer
from typing import List, Union
import numpy as np

from testing.datasets import TorchDataLoader, create_npz_dataloaders
from testing.numpy_utils import to_numpy, to_one_hot_numpy, l1_norm, pp, load_eigen_array
from testing.models import MLP1, MLP1a, MLP2, print_model_info


def compute_accuracy_torch(M: Union[MLP1, MLP1a], data_loader):
    N = len(data_loader.dataset)  # N is the number of examples
    total_correct = 0
    for X, T in data_loader:
        Y = M(X)
        predicted = Y.argmax(axis=1)  # the predicted classes for the batch
        total_correct += (predicted == T).sum().item()
    return total_correct / N


def compute_loss_torch(M: Union[MLP1, MLP1a], data_loader):
    N = len(data_loader.dataset)  # N is the number of examples
    batch_size = N // len(data_loader)
    total_loss = 0.0
    for X, T in data_loader:
        Y = M(X)
        total_loss += M.loss(Y, T).sum()
    return batch_size * total_loss / N


def compute_accuracy_nerva(M: MLP2, data_loader):
    N = len(data_loader.dataset)  # N is the number of examples
    total_correct = 0
    for X, T in data_loader:
        X = to_numpy(X)
        T = to_numpy(T)
        Y = M.feedforward(X)
        predicted = Y.argmax(axis=0)  # the predicted classes for the batch
        total_correct += (predicted == T).sum().item()
    return total_correct / N


def compute_loss_nerva(M: MLP2, data_loader):
    N = len(data_loader.dataset)  # N is the number of examples
    total_loss = 0.0
    for X, T in data_loader:
        X = to_numpy(X)
        T = to_one_hot_numpy(T, 10)
        Y = M.feedforward(X)
        total_loss += M.loss.value(Y, T)
    return total_loss / N


def compute_weight_difference(M1, M2):
    wdiff = [l1_norm(W1 - W2) for W1, W2 in zip(M1.weights(), M2.weights())]
    bdiff = [l1_norm(b1 - b2) for b1, b2 in zip(M1.bias(), M2.bias())]
    w1 = [l1_norm(W) for W in M1.weights()]
    w2 = [l1_norm(W) for W in M2.weights()]
    print(f'weight differences: {wdiff} bias differences: {bdiff} |W1|={w1} |W2|={w2}')


def compute_matrix_difference(name, X1: np.ndarray, X2: np.ndarray):
    print(f'{name} difference: {l1_norm(X1 - X2)}')


def print_epoch(epoch, lr, loss, train_accuracy, test_accuracy, elapsed):
    print(f'epoch {epoch:3}  '
          f'lr: {lr:.8f}  '
          f'loss: {loss:.8f}  '
          f'train accuracy: {train_accuracy:.8f}  '
          f'test accuracy: {test_accuracy:.8f}  '
          f'time: {elapsed:.8f}s'
         )


def train_torch(M, train_loader, test_loader, epochs, debug: bool):
    M.train()  # Set model in training mode

    print_epoch(epoch=0,
                lr=M.optimizer.param_groups[0]["lr"],
                loss=compute_loss_torch(M, train_loader),
                train_accuracy=compute_accuracy_torch(M, train_loader),
                test_accuracy=compute_accuracy_torch(M, test_loader),
                elapsed=0)

    for epoch in range(epochs):
        elapsed = 0.0
        for k, (X, T) in enumerate(train_loader):
            start = timer()
            M.optimizer.zero_grad()
            Y = M(X)
            # Y.retain_grad()
            loss = M.loss(Y, T)
            loss.backward()
            M.optimize()
            elapsed += (timer() - start)

            if debug:
                print(f'epoch: {epoch} batch: {k}')
                print_model_info(M)
                pp('X', X)
                pp('Y', Y)
                #pp('DY', Y.grad.detach())

        print_epoch(epoch=epoch + 1,
                    lr=M.optimizer.param_groups[0]["lr"],
                    loss=compute_loss_torch(M, train_loader),
                    train_accuracy=compute_accuracy_torch(M, train_loader),
                    test_accuracy=compute_accuracy_torch(M, test_loader),
                    elapsed=elapsed)

        M.learning_rate.step()  # N.B. this updates the learning rate in M.optimizer


def measure_inference_time_torch(M, train_loader, repetitions=10):
    M.train()  # Set model in training mode
    batch_size = len(train_loader.dataset) // len(train_loader)

    total_time = 0.0
    for k, (X, T) in enumerate(train_loader):
        start = timer()
        Y = M(X)
        elapsed = (timer() - start)
        total_time += elapsed
        print(f'batch {k} took {elapsed} seconds')
        if k == repetitions:
            break
    print(f'Average PyTorch inference time for batch_size {batch_size}: {1000*total_time/repetitions}ms')


# At every epoch a new dataset in .npz format is read from datadir.
def train_torch_preprocessed(M, datadir, epochs, batch_size, debug: bool):
    M.train()  # Set model in training mode

    train_loader, test_loader = create_npz_dataloaders(f'{datadir}/epoch0.npz', batch_size=batch_size)

    print_epoch(epoch=0,
                lr=M.optimizer.param_groups[0]["lr"],
                loss=compute_loss_torch(M, train_loader),
                train_accuracy=compute_accuracy_torch(M, train_loader),
                test_accuracy=compute_accuracy_torch(M, test_loader),
                elapsed=0)

    for epoch in range(epochs):
        if epoch > 0:
            train_loader, test_loader = create_npz_dataloaders(f'{datadir}/epoch{epoch}.npz', batch_size)

        elapsed = 0.0
        for k, (X, T) in enumerate(train_loader):
            start = timer()
            M.optimizer.zero_grad()
            Y = M(X)
            # Y.retain_grad()
            loss = M.loss(Y, T)
            loss.backward()
            M.optimize()
            elapsed += (timer() - start)

            if debug:
                print(f'epoch: {epoch} batch: {k}')
                print_model_info(M)
                pp('X', X)
                pp('Y', Y)
                # pp('DY', Y.grad.detach())

        print_epoch(epoch=epoch + 1,
                    lr=M.optimizer.param_groups[0]["lr"],
                    loss=compute_loss_torch(M, train_loader),
                    train_accuracy=compute_accuracy_torch(M, train_loader),
                    test_accuracy=compute_accuracy_torch(M, test_loader),
                    elapsed=elapsed)

        M.learning_rate.step()  # N.B. this updates the learning rate in M.optimizer


def train_nerva(M, train_loader, test_loader, epochs, debug: bool):
    n_classes = M.sizes[-1]
    batch_size = len(train_loader.dataset) // len(train_loader)

    print_epoch(epoch=0,
                lr=M.learning_rate(0),
                loss=compute_loss_nerva(M, train_loader),
                train_accuracy=compute_accuracy_nerva(M, train_loader),
                test_accuracy=compute_accuracy_nerva(M, test_loader),
                elapsed=0)

    for epoch in range(epochs):
        lr = M.learning_rate(epoch)
        elapsed = 0.0
        for k, (X, T) in enumerate(train_loader):
            start = timer()
            X = to_numpy(X)
            T = to_one_hot_numpy(T, n_classes)
            Y = M.feedforward(X)
            DY = M.loss.gradient(Y, T) / batch_size
            M.backpropagate(Y, DY)
            M.optimize(lr)
            elapsed += (timer() - start)

            if debug:
                print(f'epoch: {epoch} batch: {k}')
                print_model_info(M)
                pp('X', X)
                pp('Y', Y)
                # pp('DY', DY)

        print_epoch(epoch=epoch + 1,
                    lr=lr,
                    loss=compute_loss_nerva(M, train_loader),
                    train_accuracy=compute_accuracy_nerva(M, train_loader),
                    test_accuracy=compute_accuracy_nerva(M, test_loader),
                    elapsed=elapsed)


def measure_inference_time_nerva(M, train_loader, repetitions=10):
    batch_size = len(train_loader.dataset) // len(train_loader)

    total_time = 0.0
    for k, (X, T) in enumerate(train_loader):
        start = timer()
        Y = M.feedforward(X)
        elapsed = (timer() - start)
        total_time += elapsed
        print(f'batch {k} took {elapsed} seconds')
        if k == repetitions:
            break
    print(f'Average Nerva inference time for batch_size {batch_size}: {1000*total_time/repetitions}ms')


# TODO: use classes to reuse code
# At every epoch a new dataset in .npz format is read from datadir.
def train_nerva_preprocessed(M, datadir, epochs, batch_size, debug: bool):
    train_loader, test_loader = create_npz_dataloaders(f'{datadir}/epoch0.npz', batch_size=batch_size)

    n_classes = M.sizes[-1]
    batch_size = len(train_loader.dataset) // len(train_loader)

    print_epoch(epoch=0,
                lr=M.learning_rate(0),
                loss=compute_loss_nerva(M, train_loader),
                train_accuracy=compute_accuracy_nerva(M, train_loader),
                test_accuracy=compute_accuracy_nerva(M, test_loader),
                elapsed=0)

    for epoch in range(epochs):
        if epoch > 0:
            train_loader, test_loader = create_npz_dataloaders(f'{datadir}/epoch{epoch}.npz', batch_size)

        lr = M.learning_rate(epoch)
        elapsed = 0.0
        for k, (X, T) in enumerate(train_loader):
            start = timer()
            X = to_numpy(X)
            T = to_one_hot_numpy(T, n_classes)
            Y = M.feedforward(X)
            DY = M.loss.gradient(Y, T) / batch_size
            M.backpropagate(Y, DY)
            M.optimize(lr)
            elapsed += (timer() - start)

            if debug:
                print(f'epoch: {epoch} batch: {k}')
                print_model_info(M)
                pp('X', X)
                pp('Y', Y)
                # pp('DY', DY)

        print_epoch(epoch=epoch + 1,
                    lr=lr,
                    loss=compute_loss_nerva(M, train_loader),
                    train_accuracy=compute_accuracy_nerva(M, train_loader),
                    test_accuracy=compute_accuracy_nerva(M, test_loader),
                    elapsed=elapsed)


def train_both(M1: MLP1, M2: MLP2, train_loader, test_loader, epochs, debug: bool):
    M1.train()  # Set model in training mode

    n_classes = M2.sizes[-1]
    batch_size = len(train_loader.dataset) // len(train_loader)

    compute_weight_difference(M1, M2)

    for epoch in range(epochs):
        start = timer()
        lr = M2.learning_rate(epoch)

        for k, (X1, T1) in enumerate(train_loader):
            M1.optimizer.zero_grad()
            Y1 = M1(X1)
            Y1.retain_grad()
            loss = M1.loss(Y1, T1)
            loss.backward()
            M1.optimize()

            if debug:
                print(f'epoch: {epoch} batch: {k}')
                pp('Y', Y1)
                pp('DY', Y1.grad.detach())

            X2 = to_numpy(X1)
            T2 = to_one_hot_numpy(T1, n_classes)
            Y2 = M2.feedforward(X2)
            DY2 = M2.loss.gradient(Y2, T2) / batch_size
            M2.backpropagate(Y2, DY2)
            M2.optimize(lr)

            if debug:
                print(f'epoch: {epoch} batch: {k}')
                pp('Y', Y2)
                pp('DY', DY2)

            if debug:
                print(f'epoch: {epoch} batch: {k}')
                compute_matrix_difference('Y', Y1.detach().numpy().T, Y2)
                compute_matrix_difference('DY', Y1.grad.detach().numpy().T, DY2)
            compute_weight_difference(M1, M2)

            elapsed = timer() - start

        print_epoch(epoch=epoch + 1,
                    lr=M1.optimizer.param_groups[0]["lr"],
                    loss=compute_loss_torch(M1, train_loader),
                    train_accuracy=compute_accuracy_torch(M1, train_loader),
                    test_accuracy=compute_accuracy_torch(M1, test_loader),
                    elapsed=elapsed)

        print_epoch(epoch=epoch + 1,
                    lr=lr,
                    loss=compute_loss_nerva(M2, train_loader),
                    train_accuracy=compute_accuracy_nerva(M2, train_loader),
                    test_accuracy=compute_accuracy_nerva(M2, test_loader),
                    elapsed=elapsed)

        M1.learning_rate.step()


def compute_densities(density: float, sizes: List[int], erk_power_scale: float = 1.0) -> List[float]:
    layer_shapes = [(sizes[i], sizes[i+1]) for i in range(len(sizes) - 1)]
    n = len(layer_shapes)  # the number of layers

    if density == 1.0:
        return [1.0] * n

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
    #print(f"Overall sparsity {total_nonzero / total_params:.4f}")
    return densities
