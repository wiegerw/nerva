#!/bin/bash
source utilities.sh

print_header "Train CIFAR10 using a sparse Nerva model"
python3 mlp.py --seed=1 \
               --overall-density=0.05 \
               --learning-rate="Constant(0.1)" \
               --sizes=3072,1024,512,10 \
               --layers="ReLU;ReLU;Linear" \
               --batch-size=100 \
               --epochs=1 \
               --optimizers="Nesterov(0.9)" \
               --loss=SoftmaxCrossEntropy \
               --datadir=./data \
               --init-weights=XavierNormalized

print_header "Train CIFAR10 with preprocessed data using a sparse Nerva model"
python3 mlp.py --seed=1 \
               --overall-density=0.05 \
               --learning-rate="Constant(0.1)" \
               --sizes=3072,1024,512,10 \
               --layers="ReLU;ReLU;Linear" \
               --batch-size=100 \
               --epochs=1 \
               --optimizers="Nesterov(0.9)" \
               --loss=SoftmaxCrossEntropy \
               --preprocessed=./cifar1 \
               --init-weights=XavierNormalized

print_header "Train CIFAR10 using the mlp tool"
../tools/dist/mlp --seed=1 \
                  --overall-density=0.05 \
                  --sizes='3072,1024,512,10' \
                  --batch-size=100 \
                  --epochs=1 \
                  --learning-rate='Constant(0.1)' \
                  --optimizers='Nesterov(0.9)' \
                  --layers='ReLU;ReLU;Linear' \
                  --init-weights=XavierNormalized \
                  --dataset=cifar10 \
                  --size=50000 \
                  --loss=SoftmaxCrossEntropy \
                  --threads=4 \
                  --no-shuffle \
                  --verbose

print_header "Train CIFAR10 with preprocessed data using the mlp tool"
../tools/dist/mlp --seed=1 \
                  --overall-density=0.05 \
                  --sizes='3072,1024,512,10' \
                  --batch-size=100 \
                  --epochs=1 \
                  --learning-rate='Constant(0.1)' \
                  --optimizers='Nesterov(0.9)' \
                  --layers='ReLU;ReLU;Linear' \
                  --init-weights=XavierNormalized \
                  --preprocessed=./cifar1 \
                  --loss=SoftmaxCrossEntropy \
                  --threads=4 \
                  --no-shuffle \
                  --verbose

