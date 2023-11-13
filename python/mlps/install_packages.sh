#!/bin/bash

for framework in "nerva-jax" "nerva-sympy" "nerva-torch" "nerva-tensorflow" "nerva-numpy"
do
    echo "Installing $framework"
    cd $framework
    pip3 install . --user
    cd ..
done
