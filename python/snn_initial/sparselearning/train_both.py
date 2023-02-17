import tempfile
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sparselearning import train_pytorch, train_nerva
from sparselearning.train_nerva import log_model_parameters, log_test_results, log_training_results
from sparselearning.logger import Logger
import nerva.dataset
import nerva.layers
import nerva.learning_rate
import nerva.loss
import nerva.optimizers


class MLP_CIFAR10_pytorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.mask = None
        self.optimizer = None

    def optimize(self):
        if self.mask is not None:
            self.mask.step()
        else:
            self.optimizer.step()

    def learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]

    def forward(self, x):
        x0 = F.relu(self.fc1(x.view(-1, 3072)))
        x1 = F.relu(self.fc2(x0))
        # return F.log_softmax(self.fc3(x1), dim=1)
        return self.fc3(x1)

    def export_weights(self, filename: str):
        with open(filename, "wb") as f:
            W1 = self.fc1.weight.detach().numpy()
            W2 = self.fc2.weight.detach().numpy()
            W3 = self.fc3.weight.detach().numpy()
            np.save(f, np.asfortranarray(W1))
            np.save(f, np.asfortranarray(W2))
            np.save(f, np.asfortranarray(W3))

    def export_bias(self, filename: str):
        with open(filename, "wb") as f:
            b1 = self.fc1.bias.detach().numpy()
            b2 = self.fc2.bias.detach().numpy()
            b3 = self.fc3.bias.detach().numpy()
            np.save(f, np.asfortranarray(b1))
            np.save(f, np.asfortranarray(b2))
            np.save(f, np.asfortranarray(b3))

    def info(self):
        torch.set_printoptions(threshold=10)
        W1 = self.fc1.weight
        W2 = self.fc2.weight
        W3 = self.fc3.weight
        b1 = self.fc1.bias
        b2 = self.fc2.bias
        b3 = self.fc3.bias
        print(f'W1 = {W1.shape}')
        print(W1)
        print(f'b1 = {b1.shape}')
        print(b1)
        print(f'W2 = {W2.shape}')
        print(W2)
        print(f'b2 = {b2.shape}')
        print(b2)
        print(f'W3 = {W3.shape}')
        print(W3)
        print(f'b3 = {b3.shape}')
        print(b3)


class MLP_CIFAR10_nerva(nerva.layers.Sequential):
    def __init__(self, density, optimizer: nerva.optimizers.Optimizer):
        super().__init__()
        shapes = [(3072, 1024), (1024, 512), (512, 10)]

        densities = train_nerva.compute_sparse_layer_densities(density, shapes)
        sparsities = [1.0 - x for x in densities]
        layer_sizes = [1024, 512, 10]
        activations = [nerva.layers.ReLU(), nerva.layers.ReLU(), nerva.layers.NoActivation()]
        # activations = [nerva.layers.ReLU(), nerva.layers.ReLU(), nerva.layers.LogSoftmax()]

        for (sparsity, size, activation) in zip(sparsities, layer_sizes, activations):
            if sparsity == 0.0:
                 self.add(nerva.layers.Dense(size, activation=activation, optimizer=optimizer))
            else:
                self.add(nerva.layers.Sparse(size, sparsity, activation=activation, optimizer=optimizer))


def flatten_numpy(X: np.array):
    shape = X.shape
    return X.reshape(shape[0], -1)


def flatten_torch(X: torch.Tensor):
    shape = X.shape
    return X.reshape(shape[0], -1)


# Copies models and weights from model1 to model2
def copy_weights_and_biases(model1: nn.Module, model2: nerva.layers.Sequential):
    name = tempfile.NamedTemporaryFile().name
    filename1 =  name + '_weights.npy'
    filename2 =  name + '_bias.npy'
    print('saving weights to', filename1)
    print('saving bias to', filename2)
    model1.export_weights(filename1)
    model2.import_weights(filename1)
    model1.export_bias(filename2)
    model2.import_bias(filename2)
    #model1.info()
    #model2.info()


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
            test_loss += loss_fn(output, target).item() * len(data)
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


def to_numpy(x: torch.Tensor):
    return np.asfortranarray(flatten_torch(x).detach().numpy().T)


def train_and_test(i, args, device, train_loader, test_loader, dataset, log):
    log("\nIteration start: {0}/{1}\n".format(i + 1, args.iters))

    model1 = MLP_CIFAR10_pytorch().to(device)
    model1.optimizer = train_pytorch.make_optimizer(args.optimizer, model1, args.lr, args.momentum, args.l2, nesterov=True)
    if args.density < 1.0:
        model1.mask = train_pytorch.make_mask(args, model1, model1.optimizer, train_loader)
    loss_fn1 = nn.CrossEntropyLoss()
    log_model_parameters(log, model1, args)

    optimizer2 = train_nerva.make_optimizer(args.momentum, nesterov=True)
    model2 = MLP_CIFAR10_nerva(args.density, optimizer2)
    model2.compile(3072, args.batch_size)
    loss_fn2 = nerva.loss.SoftmaxCrossEntropyLoss()

    copy_weights_and_biases(model1, model2)
    log_model_parameters(log, model2, args)

    for batch_idx, (data, target) in enumerate(train_loader):
        X1, T1 = data.to(device), target.to(device)

        model1.optimizer.zero_grad()
        Y1 = model1(X1)
        Y1.retain_grad()
        loss = loss_fn1(Y1, T1)
        loss1 = loss.item()
        pred = Y1.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct1 = pred.eq(T1.view_as(pred)).sum().item()
        loss.backward()
        DY1 = Y1.grad.detach()
        lr = model1.learning_rate()
        model1.optimize()  # this will update the learning rate
        X2 = to_numpy(X1)
        T2 = to_numpy(torch.nn.functional.one_hot(T1, num_classes = 10).float())

        Y2 = model2.feedforward(X2)
        correct2 = train_nerva.correct_predictions(Y2, T2)
        loss2 = loss_fn2.value(Y2, T2) / args.batch_size
        DY2 = loss_fn2.gradient(Y2, T2) / args.batch_size
        model2.backpropagate(Y2, DY2)
        model2.optimize(lr)

        print(f'--- batch {batch_idx} ---')
        print('loss1', loss1)
        print('loss2', loss2)
        print('correct1', correct1)
        print('correct2', correct2)
        print(f'|Y1 - Y2| = {np.linalg.norm(to_numpy(Y1) - Y2)}')
        print(f'|DY1 - DY2| = {np.linalg.norm(to_numpy(DY1) - DY2)}')
