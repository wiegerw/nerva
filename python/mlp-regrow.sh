#!/bin/bash
source utilities.sh

# N.B. This script is still very experimental!

print_header "Nerva-python with regrow"
python3 mlp.py --nerva --seed=1 --overall-density=0.01 --lr=0.1 --sizes=3072,1024,512,10 --batch-size=100 --epochs=10 --momentum=0.9 --nesterov --datadir=./data --zeta=0.2

print_header "Nerva-python preprocessed with regrow"
python3 mlp.py --nerva --seed=1 --overall-density=0.05 --lr=0.1 --sizes=3072,1024,512,10 --batch-size=100 --epochs=10 --momentum=0.9 --nesterov --preprocessed=./cifar1 --zeta=0.2

print_header "Nerva-c++ with regrow"
../tools/dist/mlpf --seed=1 --overall-density=0.05 --sizes='3072,1024,512,10' --batch-size=100 --epochs=10 --learning-rate='constant(0.1)' --optimizer='nesterov(0.9)' --layers=ReLU;ReLU;Linear --weights=xxx --dataset=cifar10 --size=50000 --loss=softmax-cross-entropy --threads=4 --no-shuffle -v --zeta=0.2

print_header "Nerva-c++ preprocessed with regrow"
../tools/dist/mlpf --seed=1 --overall-density=0.05 --sizes='3072,1024,512,10' --batch-size=100 --epochs=10 --learning-rate='constant(0.1)' --optimizer='nesterov(0.9)' --layers=ReLU;ReLU;Linear --weights=xxx --preprocessed=./cifar1 --size=50000 --loss=softmax-cross-entropy --threads=4 --no-shuffle -v --zeta=0.2
