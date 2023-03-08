#!/bin/bash
source utilities.sh

# This experiment demonstrates unexpected timing results in a Nerva model with low density when weights are imported.
#
# epoch   0 lr: 0.10000000  loss: 2.30272428  train accuracy: 0.10000000  test accuracy: 0.10000000 time: 0.00000000s
# epoch   1 lr: 0.10000000  loss: 2.30258383  train accuracy: 0.09518000  test accuracy: 0.09800000 time: 3.33798666s
# epoch   2 lr: 0.10000000  loss: 2.30258377  train accuracy: 0.10266000  test accuracy: 0.10530000 time: 16.47344112s
# epoch   3 lr: 0.10000000  loss: 2.30258374  train accuracy: 0.11114000  test accuracy: 0.11670000 time: 25.86790928s
#
# It is unknown what causes the slowdown in epoch 2 and epoch 3

print_header "PyTorch export weights"
python3 -u mlp.py --torch --seed=1 --overall-density=0.001 --lr=0.1 --sizes=3072,1024,1024,1024,10 --batch-size=100 --epochs=3 --momentum=0.9 --nesterov --datadir=./data --precision=8 --save-weights=weights-pytorch-0.001.npz 2>&1 | tee log1

print_header "Nerva export weights"
../tools/dist/mlpf --seed=1 --overall-density=0.001 --sizes=3072,1024,1024,1024,10 --batch-size=100 --epochs=3 '--learning-rate=constant(0.1)' '--optimizer=nesterov(0.9)' --architecture=RRRL --weights=XXXX --loss=softmax-cross-entropy --threads=4 --no-shuffle --verbose --dataset=cifar10 --save-weights=weights-nerva-0.001.npz 2>&1 | tee log2

print_header "Nerva import Nerva weights"
../tools/dist/mlpf --seed=1 --overall-density=0.001 --sizes=3072,1024,1024,1024,10 --batch-size=100 --epochs=3 '--learning-rate=constant(0.1)' '--optimizer=nesterov(0.9)' --architecture=RRRL --weights=XXXX --loss=softmax-cross-entropy --threads=4 --no-shuffle --verbose --dataset=cifar10 --load-weights=weights-nerva-0.001.npz 2>&1 | tee log4

print_header "Nerva import PyTorch weights"
../tools/dist/mlpf --seed=1 --overall-density=0.001 --sizes=3072,1024,1024,1024,10 --batch-size=100 --epochs=3 '--learning-rate=constant(0.1)' '--optimizer=nesterov(0.9)' --architecture=RRRL --weights=XXXX --loss=softmax-cross-entropy --threads=4 --no-shuffle --verbose --dataset=cifar10 --load-weights=weights-pytorch-0.001.npz 2>&1 | tee log3

print_header "PyTorch import Nerva weights"
python3 -u mlp.py --torch --seed=1 --overall-density=0.001 --lr=0.1 --sizes=3072,1024,1024,1024,10 --batch-size=100 --epochs=3 --momentum=0.9 --nesterov --datadir=./data --precision=8 --load-weights=weights-nerva-0.001.npz 2>&1 | tee log6

print_header "PyTorch import PyTorch weights"
python3 -u mlp.py --torch --seed=1 --overall-density=0.001 --lr=0.1 --sizes=3072,1024,1024,1024,10 --batch-size=100 --epochs=3 --momentum=0.9 --nesterov --datadir=./data --precision=8 --load-weights=weights-pytorch-0.001.npz 2>&1 | tee log5
