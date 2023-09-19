#!/bin/bash
source utilities.sh

seed=1
init_weights=Xavier
density=1
sizes="3072,1024,512,10"
layers="ReLU;ReLU;Linear"
dropouts="0.3,0,0"
optimizers="Momentum(0.9)"
learning_rate="Constant(0.01)"
loss=SoftmaxCrossEntropy
batch_size=100
epochs=5

print_header "Train CIFAR10 using mlprowwise.cpp"
../tools/dist/mlprowwise \
	--seed=$seed \
	--overall-density=$density \
	--batch-size=$batch_size \
	--epochs=$epochs \
	--sizes=$sizes \
	--layers=$layers \
	--dropouts="$dropouts" \
	--optimizers=$optimizers \
	--init-weights=$init_weights \
	--learning-rate=$learning_rate \
	--loss=$loss \
	--dataset=cifar10 \
	--threads=4 \
	--no-shuffle \
	--verbose \
	2>&1 | tee logs/dropout-mlprowwise.cpp.log

print_header "Train CIFAR10 using mlpcolwise.cpp"
../tools/dist/mlpcolwise \
        --seed=$seed \
        --overall-density=$density \
        --batch-size=$batch_size \
        --epochs=$epochs \
        --sizes=$sizes \
        --layers=$layers \
        --dropouts="$dropouts" \
        --optimizers=$optimizers \
        --init-weights=$init_weights \
        --learning-rate=$learning_rate \
        --loss=$loss \
        --dataset=cifar10 \
        --threads=4 \
        --no-shuffle \
        --verbose \
        2>&1 | tee logs/dropout-mlpcolwise.cpp.log

print_header "Train CIFAR10 using mlprowwise.py"
python3 -u mlprowwise.py \
	--seed=$seed \
	--overall-density=$density \
	--batch-size=$batch_size \
	--epochs=$epochs \
	--sizes=$sizes \
	--layers=$layers \
	--dropouts="$dropouts" \
	--optimizers=$optimizers \
	--init-weights=$init_weights \
	--learning-rate=$learning_rate \
	--loss=$loss \
	--datadir=./data \
	2>&1 | tee logs/dropout-mlprowwise.py.log

print_header "Train CIFAR10 using mlpcolwise.py"
python3 -u mlpcolwise.py \
	--seed=$seed \
	--overall-density=$density \
	--batch-size=$batch_size \
	--epochs=$epochs \
	--sizes=$sizes \
	--layers=$layers \
	--dropouts="$dropouts" \
	--optimizers=$optimizers \
	--init-weights=$init_weights \
	--learning-rate=$learning_rate \
	--loss=$loss \
	--datadir=./data \
	2>&1 | tee logs/dropout-mlpcolwise.py.log

