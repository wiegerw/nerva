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

print_header "mlprowwise.py with regrow"
python3 -u mlprowwise.py \
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
	2>&1 | tee logs/regrow-mlprowwise.py.log

print_header "mlpcolwise.py with regrow"
python3 -u mlpcolwise.py \
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
	2>&1 | tee logs/regrow-mlpcolwise.py.log

print_header "mlprowwise.cpp with regrow"
../tools/dist/mlprowwise \
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
	2>&1 | tee logs/regrow-mlprowwise.cpp.log

print_header "mlpcolwise.cpp with regrow"
../tools/dist/mlpcolwise \
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
	2>&1 | tee logs/regrow-mlpcolwise.cpp-preprocessed.log
