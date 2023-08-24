#!/bin/bash
source utilities.sh

# This script assumes that preprocessed data has been created using the snn_preprocess_data.py script.

seed=1
init_weights=XavierNormalized
density=0.05
sizes="3072,1024,512,10"
layers="ReLU;ReLU;Linear"
optimizers="Nesterov(momentum=0.9)"
learning_rate="Constant(lr=0.1)"
loss=SoftmaxCrossEntropy
batch_size=100
epochs=5

print_header "Train CIFAR10 using mlp.py with preprocessed data"
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
	2>&1 | tee logs/mlp-preprocessed-python.log

print_header "Train CIFAR10 using mlp.cpp with preprocessed data"
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
	2>&1 | tee logs/mlp-preprocessed-cpp.log
