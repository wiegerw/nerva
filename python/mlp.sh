#!/bin/bash
source utilities.sh

print_header "Train CIFAR10 using a sparse Nerva model"
python3 mlp.py --seed=1 \
               --overall-density=0.05 \
               --sizes=3072,1024,512,10 \
               --layers="ReLU;ReLU;Linear" \
               --optimizers="Nesterov(0.9)" \
               --init-weights=XavierNormalized \
               --learning-rate="Constant(0.1)" \
               --loss=SoftmaxCrossEntropy \
               --batch-size=100 \
               --epochs=1 \
               --datadir=./data

print_header "Train CIFAR10 using the mlp tool"
../tools/dist/mlp --seed=1 \
                  --overall-density=0.05 \                  --loss=SoftmaxCrossEntropy \                  --loss=SoftmaxCrossEntropy \


                  --sizes=3072,1024,512,10 \
                  --layers="ReLU;ReLU;Linear" \
                  --optimizers="Nesterov(0.9)" \
                  --init-weights=XavierNormalized \
                  --learning-rate="Constant(0.1)" \
                  --loss=SoftmaxCrossEntropy \
                  --batch-size=100 \
                  --epochs=1 \
                  --dataset=cifar10 \
                  --threads=4 \
                  --no-shuffle \
                  --verbose
