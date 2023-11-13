#!/bin/bash

for framework in "nerva-jax" "nerva-torch" "nerva-tensorflow" "nerva-numpy"
do
    echo "Testing $framework"
    cd $framework/examples
    ./cifar10.sh
    ./mnist.sh
done
