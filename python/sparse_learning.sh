#!/bin/bash

python3 sparse_learning.py --sparse --fix --seed 17 --sparse_init ER --lr 0.1 --density 0.05 --model mlp_cifar10 --data cifar10 --epoch 10
