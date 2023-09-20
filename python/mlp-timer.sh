#!/bin/bash
source utilities.sh

seed=3
init_weights=XavierNormalized
density=0.05
sizes="3072,1024,512,10"
layers="ReLU;ReLU;Linear"
optimizers="Nesterov(0.9)"
learning_rate="Constant(0.1)"
loss=SoftmaxCrossEntropy
batch_size=100
epochs=5

print_header "Train CIFAR10 using mlpcolwise.cpp"
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
	--preprocessed=./cifar$seed \
	--threads=4 \
	--no-shuffle \
	--verbose \
	--timer \
	2>&1 | tee logs/timer-mlpcolwise.cpp.log

print_header "Train CIFAR10 using mlprowwise.cpp"
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
	--preprocessed=./cifar$seed \
	--threads=4 \
	--no-shuffle \
	--verbose \
	--timer \
	2>&1 | tee logs/timer-mlprowwise.cpp.log

print_header "Train CIFAR10 using mlpcolwise.py"
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
	--preprocessed=./cifar$seed \
	--timer \
	2>&1 | tee logs/timer-mlpcolwise.py.log

print_header "Train CIFAR10 using mlprowwise.py"
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
        --preprocessed=./cifar$seed \
        --timer \
        2>&1 | tee logs/timer-mlprowwise.py.log
