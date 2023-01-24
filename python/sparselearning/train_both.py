import tempfile
import time
import torch
import torch.nn as nn
from sparselearning import train_pytorch, train_nerva
from sparselearning.train_nerva import log_model_parameters, log_test_results, log_training_results
from sparselearning.logger import Logger
import nerva.layers
import nerva.learning_rate
import nerva.loss
from nerva.learning_rate import MultiStepLRScheduler


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
    model1.info()
    model2.info()


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


def train_and_test(i, args, device, train_loader, test_loader, log):
    log("\nIteration start: {0}/{1}\n".format(i + 1, args.iters))

    milestones = [int(args.epochs / 2) * args.multiplier, int(args.epochs * 3 / 4) * args.multiplier]

    model1 = train_pytorch.make_model(args.model, device)
    optimizer1 = train_pytorch.make_optimizer(args.optimizer, model1, args.lr, args.momentum, args.l2, nesterov=True)
    lr_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones, last_epoch=-1)
    mask = train_pytorch.make_mask(args, model1, optimizer1, train_loader)
    log_model_parameters(log, model1, args)

    sparsity = 1.0 - args.density
    optimizer2 = train_nerva.make_optimizer(args.momentum, nesterov=True)
    model2 = train_nerva.make_model(args.model, sparsity, optimizer2)
    model2.compile(3072, args.batch_size)
    lr_scheduler2 = nerva.learning_rate.MultiStepLRScheduler(args.lr, milestones, 0.1)
    loss_fn2 = nerva.loss.SoftmaxCrossEntropyLoss()

    copy_weights_and_biases(model1, model2)

    # log_model_parameters(log, model1, args)
    # epochs = args.epochs * args.multiplier
    # loss_fn2 = nn.CrossEntropyLoss()
    # train_pytorch.train_model(model1, mask, loss_fn2, train_loader, test_loader, lr_scheduler1, optimizer1, device, epochs, args.batch_size, args.log_interval, log)
    # log("\nIteration end: {0}/{1}\n".format(i + 1, args.iters))
