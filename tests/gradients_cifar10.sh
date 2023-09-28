#!/bin/bash
source gradients.sh

dataset=cifar10
sizes="3072,4,4,10"
epochs=1

toolname=mlp_colwise_double
tool="../tools/dist/$toolname"
run_all 2>&1 | tee "logs/gradients_cifar10_$toolname.log"

toolname=mlp_rowwise_double
tool="../tools/dist/$toolname"
run_all 2>&1 | tee "logs/gradients_cifar10_$toolname.log"

toolname=mlp_colwise_mkl_double
tool="../tools/dist/$toolname"
run_all 2>&1 | tee "logs/gradients_cifar10_$toolname.log"

toolname=mlp_rowwise_mkl_double
tool="../tools/dist/$toolname"
run_all 2>&1 | tee "logs/gradients_cifar10_$toolname.log"
