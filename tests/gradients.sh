#!/bin/bash
source ../python/utilities.sh

seed=12345
dataset=cifar10
sizes="3072,8,4,10"
init_weights=XavierNormalized
layers="ReLU;ReLU;Linear"
epochs=1
learning_rate="Constant(0.001)"
size=50
loss="SoftmaxCrossEntropy"
batch_size=5
gradient_step=0.00001
optimizers="GradientDescent"

function run()
{
  ../tools/dist/mlpd \
               --epochs=$epochs \
               --layers="$layers" \
               --sizes=$sizes \
               --dataset=$dataset \
               --init-weights=$init_weights \
               --batch-size=$batch_size \
               --learning-rate="$learning_rate" \
               --optimizers=$optimizers \
               --size=$size \
               --loss=$loss \
               --normalize \
               --threads=4 \
               --verbose \
               --no-shuffle \
               --seed=$seed \
               --gradient-step=$gradient_step
}

function run_default()
{
  print_header "default"
  layers="ReLU;ReLU;Linear"
  run
}

function run_srelu()
{
  print_header "srelu"
  layers="SReLU;ReLU;Linear"
  run
}

function run_softmax()
{
  print_header "softmax layer"
  layers="Softmax;ReLU;Linear"
  run
}

function run_log_softmax()
{
  print_header "logsoftmax layer"
  layers="LogSoftmax;ReLU;Linear"
  run
}

function run_momentum()
{
  print_header "momentum"
  layers="ReLU;ReLU;Linear"
  optimizers="Momentum(0.8)"
  run
}

function run_nesterov()
{
  print_header "nesterov"
  layers="ReLU;ReLU;Linear"
  optimizers="Nesterov(0.8)"
  run
}

function run_batchnorm()
{
  print_header "batch normalization"
  layers="BatchNorm;ReLU;BatchNorm;ReLU;Linear"
  optimizers="GradientDescent"
  run
}

function run_dropout()
{
  print_header "dropout"
  layers="ReLU(dropout=0.3);ReLU;Linear"
  run
}

function run_all()
{
  run_default
  run_srelu
  run_softmax
  run_log_softmax
  run_momentum
  run_nesterov
  run_batchnorm
  run_dropout
}
