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
dropouts=""
epochs=3
tool="../tools/dist/mlp_rowwise"
name="mlp"
computation="eigen"

function train_cpp()
{
  toolname=$(basename -- "$tool")
  toolname="${toolname}_$computation"
  print_header "experiment=$name   tool=$toolname"
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
      --dropouts=$dropouts \
      --threads=4 \
      --computation=$computation \
      --no-shuffle \
      --verbose \
      --timer \
      "$@" \
      2>&1 | tee logs/$name-$toolname.log
}

function train_python()
{
  toolname=$(basename -- "$tool")
  toolname="${toolname}_$computation"
  print_header "experiment=$name   tool=$toolname"
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
      --dropouts=$dropouts \
      --computation=$computation \
      --timer \
      "$@" \
      2>&1 | tee logs/$name-$tool.log
}

function train_torch()
{
  toolname="mlptorch.py"
  print_header "experiment=$name   tool=$toolname"
  python3 -u mlptorch.py \
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
    "$@" \
    2>&1 | tee logs/$name-$tool.log
}
