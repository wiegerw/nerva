#!/bin/bash
source utilities.sh

seed=1
init_weights=XavierNormalized
density=0.01
sizes="3072,1024,512,10"
layers="ReLU;ReLU;Linear"
optimizers="Nesterov(0.9)"
learning_rate="Constant(0.1)"
loss=SoftmaxCrossEntropy
batch_size=100
epochs=3
prune="Magnitude(0.2)"
grow=Random
grow_weights=XavierNormalized

print_header "Nerva-python with regrow"
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
	--prune=$prune \
	--grow=$grow \
	--grow-weights=$grow_weights \
	--datadir=./data \
	2>&1 | tee mlp-regrow1.log

print_header "Nerva-python preprocessed with regrow"
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
	--prune=$prune \
	--grow=$grow \
	--grow-weights=$grow_weights \
	--preprocessed=./cifar1 \
	2>&1 | tee mlp-regrow2.log

print_header "Nerva-c++ with regrow"
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
	--prune=$prune \
	--grow=$grow \
	--grow-weights=$grow_weights \
	--dataset=cifar10 \
	--threads=4 \
	--no-shuffle \
	--verbose \
	2>&1 | tee mlp-regrow3.log

print_header "Nerva-c++ preprocessed with regrow"
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
	--prune=$prune \
	--grow=$grow \
	--grow-weights=$grow_weights \
	--preprocessed=./cifar1 \
	--threads=4 \
	--no-shuffle \
	--verbose \
	2>&1 | tee mlp-regrow4.log
