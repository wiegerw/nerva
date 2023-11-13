#!/bin/bash

# download datasets using PyTorch
cd nerva-torch/examples
./prepare_datasets.sh
cd ..

# add symbolic links for the other frameworks
cd ../nerva-jax
ln -s ../nerva-torch/data

cd ../nerva-numpy
ln -s ../nerva-numpy/data

cd ../nerva-tensorflow
ln -s ../nerva-tensorflow/data
