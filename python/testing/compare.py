import pathlib
import sys
import tempfile

from nerva.utilities import StopWatch
from testing.models import MLP1a, MLP2
from testing.numpy_utils import pp, to_numpy, to_one_hot_numpy
from testing.training import compute_weight_difference, compute_matrix_difference, print_epoch, compute_loss_torch, \
    compute_accuracy_torch, compute_loss_nerva, compute_accuracy_nerva


# Contains global options for the compare functions
class CompareOptions(object):
    print_weight_info = False
    print_weight_norm_info = False
    print_batch_info = False
    batch_limit = sys.maxsize


def print_weight_difference(M1, M2):
    weightdiff, biasdiff, weightnorm1, weightnorm2 = compute_weight_difference(M1, M2)
    if CompareOptions.print_weight_info and CompareOptions.print_weight_norm_info:
        print(f'weight differences: {weightdiff} bias differences: {biasdiff} |W1|={weightnorm1} |W2|={weightnorm2}')
    elif CompareOptions.print_weight_info:
        print(f'weight differences: {weightdiff}')


def print_batch_info(epoch, k, Y1, Y2, DY2):
    if CompareOptions.print_batch_info:
        print(f'epoch: {epoch} batch: {k}')
        pp('Y1', Y1)
        pp('DY1', Y1.grad.detach())
        pp('Y2', Y2.T)
        pp('DY2', DY2.T)
        # compute_matrix_difference('Y', Y1.detach().numpy().T, Y2)
        # compute_matrix_difference('DY', Y1.grad.detach().numpy().T, DY2)


def copy_weights_and_bias(M1: MLP1a, M2: MLP2):
    filename = tempfile.NamedTemporaryFile().name + '.npz'
    M1.export_weights_npz(filename)
    M2.import_weights_npz(filename)
    pathlib.Path(filename).unlink()


def compare_pytorch_nerva(M1: MLP1a, M2: MLP2, train_loader, test_loader, epochs: int):
    M1.train()  # Set model in training mode
    watch = StopWatch()

    n_classes = M2.sizes[-1]
    batch_size = len(train_loader.dataset) // len(train_loader)

    print_weight_difference(M1, M2)
    for epoch in range(epochs):
        watch.reset()
        lr = M2.learning_rate(epoch)

        for k, (X1, T1) in enumerate(train_loader):
            if k > CompareOptions.batch_limit:
                break

            M1.optimizer.zero_grad()
            Y1 = M1(X1)
            Y1.retain_grad()
            loss = M1.loss(Y1, T1)
            loss.backward()
            M1.optimize()

            X2 = to_numpy(X1)
            T2 = to_one_hot_numpy(T1, n_classes)
            Y2 = M2.feedforward(X2)
            DY2 = M2.loss.gradient(Y2, T2) / batch_size
            M2.backpropagate(Y2, DY2)
            M2.optimize(lr)

            print_batch_info(epoch, k, Y1, Y2, DY2)
            print_weight_difference(M1, M2)

            elapsed = watch.seconds()

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
