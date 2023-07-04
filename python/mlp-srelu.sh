#!/bin/bash
source utilities.sh

seed=1
init_weights=Xavier
density=0.05
sizes="3072,1024,512,10"
layers="SReLU;SReLU;Linear"
optimizers="Nesterov(0.9)"
learning_rate="Constant(0.01)"
loss=SoftmaxCrossEntropy
batch_size=100
epochs=5

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
	--learning-rate=$learning_rate \
	--loss=$loss \
	--preprocessed=cifar$seed \
	--threads=4 \
	--no-shuffle \
	--verbose \
	2>&1 | tee mlp-srelu1.log

print_header "Train CIFAR10 using mlp.py"
python3 -u mlp.py \
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
	--preprocessed=cifar$seed \
	2>&1 | tee mlp-srelu2.log
