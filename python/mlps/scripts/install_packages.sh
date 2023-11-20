#!/bin/bash

for framework in "nerva-jax" "nerva-torch" "nerva-tensorflow" "nerva-numpy" "nerva-sympy"
do
    echo "Installing $framework"
    cd $framework
    pip3 install . --user
    cd ..
done
