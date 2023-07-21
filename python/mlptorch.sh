#!/bin/bash
source utilities.sh

seed=1
init_weights=XavierNormalized
density=0.05
sizes="3072,1024,512,10"
layers="ReLU;ReLU;Linear"
optimizers="Nesterov(0.9)"
learning_rate="Constant(0.1)"
loss=SoftmaxCrossEntropy
batch_size=100
epochs=5

print_header "Train CIFAR10 using mlptorch.py"
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
	--datadir=./data \
	2>&1 | tee mlptorch.log
