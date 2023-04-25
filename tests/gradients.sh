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
dropout=0

function run()
{
  extra_args=$1

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
               --gradient-step=$gradient_step \
               $1
}

function run_dataset()
{
  print_header "default"
  layers="ReLU;ReLU;Linear"
  run

  print_header "srelu"
  layers="SReLU(1,0,1,0);ReLU;Linear"
  run

  print_header "softmax layer"
  layers="Softmax;ReLU;Linear"
  run

  print_header "momentum"
  layers="ReLU;ReLU;Linear"
  optimizer="momentum(0.8)"
  run

  print_header "nesterov"
  layers="ReLU;ReLU;Linear"
  optimizer="nesterov(0.8)"
  run

  print_header "batch normalization 1"
  layers="BatchNorm;ReLU;BatchNorm;ReLU;Linear"
  optimizer="gradient-descent"
  run

  print_header "dropout"
  layers="Dropout(0.3);ReLU;ReLU;Linear"
  run
}
