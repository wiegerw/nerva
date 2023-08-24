#!/bin/bash

# This script compares the computations of Nerva C++ and Nerva python by
# logging intermediate results.

source utilities.sh

seed=1
init_weights=XavierNormalized
density=1.0
sizes="3072,1024,512,10"
layers="ReLU;ReLU;Linear"
optimizers="Nesterov(mu=0.9)"
learning_rate="Constant(lr=0.1)"
loss=SoftmaxCrossEntropy
batch_size=100
epochs=1
optimizers="GradientDescent"

function train()
{
  local weight_file="mlp-compare.npz"

  print_header "Train CIFAR10 using mlptorch.py"
  python3 -u mlptorch.py \
    --seed=$seed \
    --overall-density=$density \
    --batch-size=$batch_size \
    --epochs=$epochs \
    --sizes=$sizes \
    --layers=$layers \
    --optimizers=$optimizers \
    --save-weights=$weight_file \
    --learning-rate=$learning_rate \
    --loss=$loss \
    --preprocessed=cifar$seed \
    --debug \
    2>&1 | tee "logs/mlp-compare-${optimizers}-torch.log"

  print_header "Train CIFAR10 using mlp.cpp"
  ../tools/dist/mlp \
    --seed=$seed \
    --overall-density=$density \
    --batch-size=$batch_size \
    --epochs=$epochs \
    --sizes=$sizes \
    --layers=$layers \
    --optimizers=$optimizers \
    --init-weights=$init_weights \
    --load-weights=$weight_file \
    --learning-rate=$learning_rate \
    --loss=$loss \
    --preprocessed=cifar$seed \
    --threads=4 \
    --no-shuffle \
    --debug \
    2>&1 | tee "logs/mlp-compare-${optimizers}-cpp.log"

  print_header "Train CIFAR10 using mlp.py"
  python3 -u mlp.py \
    --seed=$seed \
    --overall-density=$density \
    --batch-size=$batch_size \
    --epochs=$epochs \
    --sizes=$sizes \
    --layers=$layers \
    --optimizers=$optimizers \
    --load-weights=$weight_file \
    --learning-rate=$learning_rate \
    --loss=$loss \
    --preprocessed=cifar$seed \
    --debug \
    2>&1 | tee "logs/mlp-compare-${optimizers}-python.log"
}

optimizers="GradientDescent"
train

optimizers="Momentum(0.9)"
learning_rate="Constant(0.01)"
train

learning_rate="Constant(0.01)"
optimizers="Nesterov(0.9)"
train
