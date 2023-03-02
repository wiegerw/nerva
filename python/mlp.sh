#!/bin/bash

function print_header()
{
  local title="$1"
  local line="====================================================================================="
  local padding="$(printf '%0.1s' ' '{1..80})" # create 80 spaces
  local title_padding_length=$(( (80 - ${#title}) / 2 ))
  local title_padding="${padding:0:title_padding_length}"
  echo "$line"
  echo "===$title_padding$title$title_padding==="
  echo "$line"
}

print_header "PyTorch --datadir"
python3 snn.py --torch --seed=1 --overall-density=0.05 --lr=0.1 --sizes=3072,1024,512,10 --batch-size=100 --epochs=1 --momentum=0.9 --nesterov --datadir=./data

print_header "PyTorch --preprocessed"
python3 snn.py --torch --seed=1 --overall-density=0.05 --lr=0.1 --sizes=3072,1024,512,10 --batch-size=100 --epochs=1 --momentum=0.9 --nesterov --preprocessed=./cifar1

print_header "Nerva-python --datadir"
python3 mlp.py --nerva --seed=1 --overall-density=0.05 --lr=0.1 --sizes=3072,1024,512,10 --batch-size=100 --epochs=1 --momentum=0.9 --nesterov --datadir=./data

print_header "Nerva-python --preprocessed"
python3 mlp.py --nerva --seed=1 --overall-density=0.05 --lr=0.1 --sizes=3072,1024,512,10 --batch-size=100 --epochs=1 --momentum=0.9 --nesterov --preprocessed=./cifar1

print_header "Nerva-c++ --dataset"
../tools/dist/mlpf --seed=1 --overall-density=0.05 --sizes='3072,1024,512,10' --batch-size=100 --epochs=1 --learning-rate='constant(0.1)' --optimizer='nesterov(0.9)' --architecture=RRL --weights=xxx --dataset=cifar10 --size=50000 --loss=softmax-cross-entropy --algorithm=sgd --threads=4 --no-shuffle -v

print_header "Nerva-c++ --preprocessed"
../tools/dist/mlpf --seed=1 --overall-density=0.05 --sizes='3072,1024,512,10' --batch-size=100 --epochs=1 --learning-rate='constant(0.1)' --optimizer='nesterov(0.9)' --architecture=RRL --weights=xxx --dataset=cifar10 --preprocessed=./cifar1 --size=50000 --loss=softmax-cross-entropy --algorithm=sgd --threads=4 --no-shuffle -v

print_header "Nerva-python with regrow"
python3 mlp.py --nerva --seed=1 --overall-density=0.01 --lr=0.1 --sizes=3072,1024,512,10 --batch-size=100 --epochs=10 --momentum=0.9 --nesterov --datadir=./data --zeta=0.2

print_header "Nerva-python preprocessed with regrow"
python3 mlp.py --nerva --seed=1 --overall-density=0.05 --lr=0.1 --sizes=3072,1024,512,10 --batch-size=100 --epochs=10 --momentum=0.9 --nesterov --preprocessed=./cifar1 --zeta=0.2
