#!/bin/bash

for framework in "nerva-jax" "nerva-sympy" "nerva-torch" "nerva-tensorflow" "nerva-numpy"
do
    echo "Uninstalling $framework"
    cd $framework
    pip3 uninstall -y $framework
    cd ..
done
