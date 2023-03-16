#!/bin/bash
source ../python/utilities.sh
PYTHONPATH=../python

# This experiment demonstrates unexpected timing results in a Nerva model with very low density and nonzero bias vectors.
#
# epoch   0 lr: 0.10000000  loss: 2.30352549  train accuracy: 0.10000000  test accuracy: 0.10000000 time: 0.00000000s
# epoch   1 lr: 0.10000000  loss: 2.30258388  train accuracy: 0.09716000  test accuracy: 0.09490000 time: 3.81212472s
# epoch   2 lr: 0.10000000  loss: 2.30258380  train accuracy: 0.10380000  test accuracy: 0.10190000 time: 16.66320519s
# epoch   3 lr: 0.10000000  loss: 2.30258369  train accuracy: 0.10946000  test accuracy: 0.10710000 time: 26.92191094s
#
# The slowdown in epoch 2 and epoch 3 is caused by small floating point values (~ 1e-40). Trimmed ReLU can be
# used to solve this problem.

print_header "Nerva zero bias"
../tools/dist/mlpf --seed=1 --overall-density=0.001 --sizes=3072,1024,1024,1024,10 --batch-size=100 --epochs=3 '--learning-rate=constant(0.1)' '--optimizer=nesterov(0.9)' --layers="ReLU;ReLU;ReLU;Linear" --weights=XXXX --loss=softmax-cross-entropy --threads=4 --no-shuffle --verbose --dataset=cifar10 2>&1 | tee issue1a.log

print_header "Nerva nonzero bias"
../tools/dist/mlpf --seed=1 --overall-density=0.001 --sizes=3072,1024,1024,1024,10 --batch-size=100 --epochs=3 '--learning-rate=constant(0.1)' '--optimizer=nesterov(0.9)' --layers="ReLU;ReLU;ReLU;Linear" --weights=pppp --loss=softmax-cross-entropy --threads=4 --no-shuffle --verbose --dataset=cifar10 2>&1 | tee issue1b.log

print_header "Nerva nonzero bias + trimmed ReLU"
../tools/dist/mlpf --seed=1 --overall-density=0.001 --sizes=3072,1024,1024,1024,10 --batch-size=100 --epochs=3 '--learning-rate=constant(0.1)' '--optimizer=nesterov(0.9)' --layers="TrimmedReLU(1e-30);TrimmedReLU(1e-30);TrimmedReLU(1e-30);Linear" --weights=pppp --loss=softmax-cross-entropy --threads=4 --no-shuffle --verbose --dataset=cifar10 2>&1 | tee issue1c.log
