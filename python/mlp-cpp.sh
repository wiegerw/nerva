#!/bin/bash
source utilities.sh

# N.B. For this script the mlpf executable must be built first.

print_header "Train CIFAR10 using the mlpf tool"
../tools/dist/mlpf --seed=1 --overall-density=0.05 --sizes='3072,1024,512,10' --batch-size=100 --epochs=1 --learning-rate='constant(0.1)' --optimizer='nesterov(0.9)' --architecture=RRL --weights=xxx --dataset=cifar10 --size=50000 --loss=softmax-cross-entropy --algorithm=sgd --threads=4 --no-shuffle -v

print_header "Train CIFAR10 with preprocessed data using the mlpf tool"
../tools/dist/mlpf --seed=1 --overall-density=0.05 --sizes='3072,1024,512,10' --batch-size=100 --epochs=1 --learning-rate='constant(0.1)' --optimizer='nesterov(0.9)' --architecture=RRL --weights=xxx --dataset=cifar10 --preprocessed=./cifar1 --size=50000 --loss=softmax-cross-entropy --algorithm=sgd --threads=4 --no-shuffle -v
