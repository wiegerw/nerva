#!/bin/bash
source utilities.sh

seed=1
init_weights=XavierNormalized
density=0.05
sizes="3072,1024,512,10"
layers="ReLU;ReLU;Linear"
optimizers="Momentum(0.9)"
learning_rate="Constant(0.1)"
loss=SoftmaxCrossEntropy
batch_size=100
epochs=3
toolname=mlp_rowwise
tool="../tools/dist/$toolname"

function train_cpp()
{
  print_header "Train CIFAR10 using $toolname"
  $tool \
      --seed=$seed \
      --overall-density=$density \
      --batch-size=$batch_size \
      --epochs=$epochs \
      --sizes=$sizes \
      --layers=$layers \
      --optimizers=$optimizers \
      --init-weights=$init_weights \
      --learning-rate=$learning_rate \
      --loss=$loss \
      --dataset=cifar10 \
      --threads=4 \
      --no-shuffle \
      --verbose \
      2>&1 | tee logs/mlp-$toolname.log
}

function train_python()
{
  print_header "Train CIFAR10 using $tool"
  python3 -u $tool \
        --seed=$seed \
        --overall-density=$density \
        --batch-size=$batch_size \
        --epochs=$epochs \
        --sizes=$sizes \
        --layers=$layers \
        --optimizers=$optimizers \
        --init-weights=$init_weights \
        --learning-rate=$learning_rate \
        --loss=$loss \
        --manual \
        --datadir=./data \
        2>&1 | tee logs/mlpcolwise.py.log
}

toolname=mlp_rowwise
tool="../tools/dist/$toolname"
train_cpp

toolname=mlp_rowwise_mkl
tool="../tools/dist/$toolname"
train_cpp

toolname=mlp_colwise
tool="../tools/dist/$toolname"
train_cpp

toolname=mlp_colwise_mkl
tool="../tools/dist/$toolname"
train_cpp

tool=mlprowwise.py
train_python

tool=mlpcolwise.py
train_python
