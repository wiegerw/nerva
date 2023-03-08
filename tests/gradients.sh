#!/bin/bash
source ../python/utilities.sh

dataset=cifar10
sizes="3072,8,4,10"

weights=xxx
architecture=RRL

seed=12345
epochs=1
lr=0.001
size=50
loss="softmax-cross-entropy"
algorithm=sgd
batch_size=5
gradient_step=0.00001
optimizer="gradient-descent"
dropout=0

function run()
{
  extra_args=$1

  ../tools/dist/mlpd --epochs=$epochs \
               --architecture=$architecture \
               --sizes=$sizes \
               --dataset=$dataset \
               --weights=$weights \
               --batch-size=$batch_size \
               --learning-rate="constant($lr)" \
               --optimizer=$optimizer \
               --dropout=$dropout \
               --size=$size \
               --loss="softmax-cross-entropy" \
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
  architecture=RRL
  run

  print_header "softmax layer"
  architecture=ZRL
  run

  print_header "momentum"
  architecture=RRL
  optimizer="momentum(0.8)"
  run

  print_header "nesterov"
  architecture=RRL
  optimizer="nesterov(0.8)"
  run

  print_header "batch normalization 1"
  architecture=BRBRL
  optimizer="gradient-descent"
  run

  print_header "dropout"
  architecture=RRL
  dropout=0.3
  run
}
