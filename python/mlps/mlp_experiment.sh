#!/bin/bash

# This script is used to compare 7 MLP implementations.

source ../utilities.sh

PYTHONPATH=..
seed=1
density=1
epochs=1
batch_size=100

function prepare_cifar10()
{
  sizes="3072,1024,512,10"
  layers="ReLU;ReLU;Linear"
  optimizer="Momentum(0.9)"
  optimizers="Momentum(0.9);Momentum(0.9);Momentum(0.9)"
  init_weight="Xavier"
  init_weights="Xavier,Xavier,Xavier"
  learning_rate="Constant(0.01)"
  loss=SoftmaxCrossEntropy
  dataset="../data/cifar10.npz"
  logfile="cifar10.log"
  weights="cifar10-weights.npz"
}

function prepare_mnist()
{
  sizes="784,1024,512,10"
  layers="ReLU;ReLU;Linear"
  optimizer="Momentum(0.9)"
  optimizers="Momentum(0.9);Momentum(0.9);Momentum(0.9)"
  init_weight="Xavier"
  init_weights="Xavier,Xavier,Xavier"
  learning_rate="Constant(0.01)"
  loss=SoftmaxCrossEntropy
  dataset="../data/mnist.npz"
  logfile="mnist.log"
  weights="mnist-weights.npz"
}

function train_pytorch()
{
  python3 -u ../mlptorch.py \
    --seed=$seed \
    --layers=$layers \
    --sizes=$sizes \
    --optimizers=$optimizer \
    --init-weights=$init_weight \
    --save-weights=$weights \
    --overall-density=$density \
    --batch-size=$batch_size \
    --epochs=$epochs \
    --loss=$loss \
    --learning-rate=$learning_rate \
    --dataset=$dataset \
    2>&1 | tee -a $logfile
}

function train_python()
{
  tool=$1
  python3 -u $tool \
          --layers=$layers \
          --sizes=$sizes \
          --optimizers=$optimizers \
          --init-weights=$init_weights \
          --weights=$weights \
          --batch-size=$batch_size \
          --epochs=$epochs \
          --loss=$loss \
          --learning-rate=$learning_rate \
          --dataset=$dataset \
          2>&1 | tee -a $logfile
}

function train_nerva_cpp()
{
  ../../tools/dist/mlp_rowwise \
      --computation=mkl \
      --seed=$seed \
      --overall-density=$density \
      --batch-size=$batch_size \
      --epochs=$epochs \
      --sizes=$sizes \
      --layers=$layers \
      --optimizers=$optimizer \
      --init-weights=$init_weight \
      --load-weights=$weights \
      --learning-rate=$learning_rate \
      --loss=$loss \
      --threads=4 \
      --no-shuffle \
      --verbose \
      --load-dataset=$dataset \
      2>&1 | tee -a $logfile
}

function train_nerva_python()
{
  ../mlprowwise.py \
      --computation=mkl \
      --seed=$seed \
      --overall-density=$density \
      --batch-size=$batch_size \
      --epochs=$epochs \
      --sizes=$sizes \
      --layers=$layers \
      --optimizers=$optimizer \
      --init-weights=$init_weight \
      --load-weights=$weights \
      --learning-rate=$learning_rate \
      --loss=$loss \
      --dataset=$dataset \
      --manual \
      2>&1 | tee -a $logfile
}

function train_all()
{
  rm $logfile
  touch $logfile
  print_header "tool: PyTorch-Native" $logfile
  train_pytorch  # N.B. this must be the first one, since it generates initial weights
  print_header "tool: Nerva-C++" $logfile
  train_nerva_cpp
  print_header "tool: Nerva-Python" $logfile
  train_nerva_python
  print_header "tool: PyTorch" $logfile
  train_python mlp_torch_rowwise.py
  print_header "tool: TensorFlow" $logfile
  train_python mlp_tensorflow_rowwise.py
  print_header "tool: JAX" $logfile
  train_python mlp_jax_rowwise.py
  print_header "tool: NumPy" $logfile
  train_python mlp_numpy_rowwise.py
}

prepare_cifar10
train_all

prepare_mnist
train_all
