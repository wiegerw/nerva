#!/bin/bash
source utilities.sh

# This script assumes that preprocessed data has been created using the snn_preprocess_data.py script.

print_header "Train CIFAR10 PyTorch with preprocessed data"
python3 mlp.py \
        --torch \
        --seed=1 \
        --overall-density=0.05 \
        --lr=0.1 \
        --sizes=3072,1024,512,10 \
        --init-weights=Xavier \
        --batch-size=100 \
        --epochs=1 \
        --momentum=0.9 \
        --nesterov \
        --preprocessed=./cifar1

print_header "Train CIFAR10 Nerva with preprocessed data"
python3 mlp.py \
        --nerva \
        --seed=1 \
        --overall-density=0.05 \
        --lr=0.1 \
        --sizes=3072,1024,512,10 \
        --init-weights=Xavier \
        --batch-size=100 \
        --epochs=1 \
        --momentum=0.9 \
        --nesterov \
        --preprocessed=./cifar1
