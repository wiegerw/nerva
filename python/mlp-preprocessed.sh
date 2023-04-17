#!/bin/bash
source utilities.sh

# This script assumes that preprocessed data has been created using the snn_preprocess_data.py script.

print_header "Train CIFAR10 with preprocessed data using a sparse Nerva model"
python3 -u mlp.py --seed=1 \
                  --overall-density=0.05 \
                  --sizes=3072,1024,512,10 \
                  --layers="ReLU;ReLU;Linear" \
                  --optimizers="Nesterov(0.9)" \
                  --init-weights=XavierNormalized \
                  --loss=SoftmaxCrossEntropy \
                  --learning-rate="Constant(0.1)" \
                  --batch-size=100 \
                  --epochs=1 \
                  --preprocessed=./cifar1 \
                  --precision=8 \
                  --edgeitems=3 \
                  2>&1 | tee mlp1.log

print_header "Train CIFAR10 with preprocessed data using the mlp tool"
../tools/dist/mlp --seed=1 \
                  --overall-density=0.05 \
                  --sizes=3072,1024,512,10 \
                  --layers="ReLU;ReLU;Linear" \
                  --optimizers="Nesterov(0.9)" \
                  --init-weights=XavierNormalized \
                  --loss=SoftmaxCrossEntropy \
                  --learning-rate='Constant(0.1)' \
                  --batch-size=100 \
                  --epochs=1 \
                  --preprocessed=./cifar1 \
                  --threads=4 \
                  --no-shuffle \
                  --verbose \
                  2>&1 | tee mlp2.log
