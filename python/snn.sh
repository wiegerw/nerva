#!/bin/bash

#python3 snn.py --sparse --fix --seed 17 --sparse_init ER --lr 0.1 --density 0.05 --model mlp_cifar10 --data cifar10 --epoch 100 --valid_split 0.0 --batch-size 100 --no-cuda --nerva
python3 snn.py --sparse --fix --seed 17 --sparse_init ER --lr 0.1 --density 0.05 --model mlp_cifar10 --data cifar10 --epoch 100 --valid_split 0.0 --batch-size 100 --no-cuda
