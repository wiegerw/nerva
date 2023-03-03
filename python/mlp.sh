#!/bin/bash
source utilities.sh

print_header "Train CIFAR10 using a sparse PyTorch model with binary masks"
python3 mlp.py --torch --seed=1 --overall-density=0.05 --lr=0.1 --sizes=3072,1024,512,10 --batch-size=100 --epochs=1 --momentum=0.9 --nesterov --datadir=./data

print_header "Train CIFAR10 using a sparse Nerva model"
python3 mlp.py --nerva --seed=1 --overall-density=0.05 --lr=0.1 --sizes=3072,1024,512,10 --batch-size=100 --epochs=1 --momentum=0.9 --nesterov --datadir=./data
