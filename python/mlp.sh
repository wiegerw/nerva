#!/bin/bash

# PyTorch
python3 snn.py --torch --seed=1 --overall-density=0.05 --lr=0.1 --sizes=3072,1024,512,10 --batch-size=100 --epochs=1 --momentum=0.9 --nesterov --datadir=./data --custom-masking

# PyTorch preprocessed
python3 snn.py --torch --seed=1 --overall-density=0.05 --lr=0.1 --sizes=3072,1024,512,10 --batch-size=100 --epochs=1 --momentum=0.9 --nesterov --preprocessed=./cifar1 --custom-masking

# Nerva-python
python3 mlp.py --nerva --seed=1 --overall-density=0.05 --lr=0.1 --sizes=3072,1024,512,10 --batch-size=100 --epochs=1 --momentum=0.9 --nesterov --datadir=./data

# Nerva-python preprocessed
python3 mlp.py --nerva --seed=1 --overall-density=0.05 --lr=0.1 --sizes=3072,1024,512,10 --batch-size=100 --epochs=1 --momentum=0.9 --nesterov --preprocessed=./cifar1

# Nerva-c++
../tools/dist/mlpf --seed=1 --overall-density=0.05 --sizes='3072,1024,512,10' --batch-size=100 --epochs=1 --learning-rate='constant(0.1)' --optimizer='nesterov(0.9)' --architecture=RRL --weights=xxx --dataset=cifar10 --size=50000 --loss=softmax-cross-entropy --algorithm=minibatch --threads=4 --no-shuffle -v

# Nerva-c++ preprocessed
../tools/dist/mlpf --seed=1 --overall-density=0.05 --sizes='3072,1024,512,10' --batch-size=100 --epochs=1 --learning-rate='constant(0.1)' --optimizer='nesterov(0.9)' --architecture=RRL --weights=xxx --dataset=cifar10 --preprocessed=./cifar1 --size=50000 --loss=softmax-cross-entropy --algorithm=minibatch --threads=4 --no-shuffle -v
