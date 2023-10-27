#!/bin/bash
source ../utilities.sh

PYTHONPATH=..
seed=1
density=1
sizes="3072,1024,512,10"
layers="ReLU;ReLU;Linear"
optimizer="Momentum(0.9)"
init_weight="Xavier"
optimizers="Momentum(mu=0.9);Momentum(mu=0.9);Momentum(mu=0.9)"
init_weights="Xavier,Xavier,Xavier"
learning_rate="Constant(0.01)"
loss=SoftmaxCrossEntropy
batch_size=100
epochs=1
CURRENTDIR=`pwd`
dataset="${CURRENTDIR}/../cifar1/epoch0.npz"
logfile="${CURRENTDIR}/experiment.log"
weights="${CURRENTDIR}/experiment-weights.npz"

function train_pytorch()
{
  print_header "Train CIFAR10 using mlptorch.py"
  python3 -u mlptorch.py \
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
    2>&1 | tee $logfile
}

function train_python()
{
  shift
  extra_args=$*
  python3 -u mlp.py \
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
          $extra_args \
          2>&1 | tee -a $logfile
}

function train_cpp()
{
  print_header "Train CIFAR10 using mlp_rowwise"
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

# create directory logs if it does not exist
mkdir -p logs

# run the experiments
cd ..
train_pytorch  # N.B. this must be the first one, since it generates initial weights
cd "$CURRENTDIR" || exit
train_cpp
train_python "numpy-rowwise"      --numpy      --rowwise
train_python "tensorflow-rowwise" --tensorflow --rowwise
train_python "torch-rowwise"      --torch      --rowwise
train_python "jax-rowwise"        --jax        --rowwise
