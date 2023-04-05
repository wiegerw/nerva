#!/bin/bash
source utilities.sh

print_header "Generate weights"
python3 mlp.py \
        --nerva \
        --seed=1 \
	--init-weights=Xavier \
        --overall-density=0.01 \
        --lr=0.1 \
        --sizes=3072,1024,512,10 \
        --epochs=0 \
        --datadir=./data \
	--save-weights="mlp-regrow.npz"

print_header "Nerva-python with regrow"
python3 mlp.py \
	--nerva \
	--seed=1 \
	--overall-density=0.01 \
	--lr=0.1 \
	--sizes=3072,1024,512,10 \
	--batch-size=100 \
	--epochs=10 \
	--momentum=0.9 \
	--nesterov \
	--datadir=./data \
	--prune='Magnitude(0.2)' \
	--grow='Random' \
	--grow-weights=XavierNormalized \
	--load-weights="mlp-regrow.npz"

print_header "Nerva-python preprocessed with regrow"
python3 mlp.py \
	--nerva \
	--seed=1 \
	--overall-density=0.01 \
	--lr=0.1 \
	--sizes=3072,1024,512,10 \
	--batch-size=100 \
	--epochs=10 \
	--momentum=0.9 \
	--nesterov \
	--preprocessed=./cifar1 \
	--prune='Magnitude(0.2)' \
	--grow='Random' \
	--grow-weights=XavierNormalized \
	--load-weights="mlp-regrow.npz"

print_header "Nerva-c++ with regrow"
../tools/dist/mlp \
	--seed=1 \
	--overall-density=0.01 \
	--sizes='3072,1024,512,10' \
	--batch-size=100 \
	--epochs=10 \
	--learning-rate='constant(0.1)' \
	--optimizer='nesterov(0.9)' \
	--layers="ReLU;ReLU;Linear" \
	--dataset=cifar10 \
	--size=50000 \
	--loss=softmax-cross-entropy \
	--threads=4 \
	--no-shuffle \
	--verbose \
	--prune='Magnitude(0.2)' \
	--grow='Random' \
	--grow-weights=XavierNormalized \
	--load-weights="mlp-regrow.npz"

print_header "Nerva-c++ preprocessed with regrow"
../tools/dist/mlp \
	--seed=1 \
	--overall-density=0.01 \
	--sizes='3072,1024,512,10' \
	--batch-size=100 \
	--epochs=10 \
	--learning-rate='constant(0.1)' \
	--optimizer='nesterov(0.9)' \
	--layers="ReLU;ReLU;Linear" \
	--preprocessed=./cifar1 \
	--size=50000 \
	--loss=softmax-cross-entropy \
	--threads=4 \
	--no-shuffle \
	--verbose \
	--prune='Magnitude(0.2)' \
	--grow='Random' \
	--grow-weights=XavierNormalized \
	--load-weights="mlp-regrow.npz"

