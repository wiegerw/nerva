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
dataset="../data/cifar10.npz"
logfile="experiment.log"
weights="experiment-weights.npz"

function train_pytorch()
{
  print_header "Train CIFAR10 using mlptorch.py"
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
    2>&1 | tee $logfile
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
  print_header "Train Nerva-c++ (rowwise)"
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
  print_header "Train Nerva-python (rowwise)"
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

# create directory logs if it does not exist
mkdir -p logs

# run the experiments
train_pytorch  # N.B. this must be the first one, since it generates initial weights
train_nerva_python
train_nerva_cpp
print_header "numpy-colwise"
train_python mlp_numpy_colwise.py
print_header "numpy-rowwise"
train_python mlp_numpy_rowwise.py
print_header "tensorflow-colwise"
train_python mlp_tensorflow_colwise.py
print_header "tensorflow-rowwise"
train_python mlp_tensorflow_rowwise.py
print_header "torch-colwise"
train_python mlp_torch_colwise.py
print_header "torch-rowwise"
train_python mlp_torch_rowwise.py
print_header "jax-colwise"
train_python mlp_jax_colwise.py
print_header "jax-rowwise"
train_python mlp_jax_rowwise.py
