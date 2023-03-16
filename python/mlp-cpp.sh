#!/bin/bash
source utilities.sh

# N.B. For this script the mlp executable must be built first.

print_header "Train CIFAR10 using the mlp tool"
../tools/dist/mlp --seed=1 \
                  --overall-density=0.05 \
                  --sizes='3072,1024,512,10' \
                  --batch-size=100 \
                  --epochs=1 \
                  --learning-rate='constant(0.1)' \
                  --optimizer='nesterov(0.9)' \
                  --layers='ReLU;ReLU;Linear' \
                  --weights=xxx \
                  --dataset=cifar10 \
                  --size=50000 \
                  --loss=softmax-cross-entropy \
                  --threads=4 \
                  --no-shuffle \
                  --verbose

print_header "Train CIFAR10 with preprocessed data using the mlp tool"
../tools/dist/mlp --seed=1 \
                  --overall-density=0.05 \
                  --sizes='3072,1024,512,10' \
                  --batch-size=100 \
                  --epochs=1 \
                  --learning-rate='constant(0.1)' \
                  --optimizer='nesterov(0.9)' \
                  --layers='ReLU;ReLU;Linear' \
                  --weights=xxx \
                  --preprocessed=./cifar1 \
                  --size=50000 \
                  --loss=softmax-cross-entropy \
                  --threads=4 \
                  --no-shuffle \
                  --verbose
