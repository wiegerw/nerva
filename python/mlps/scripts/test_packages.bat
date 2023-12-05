@echo off

cd nerva-jax
cd examples
call cifar10.bat
call mnist.bat
cd ..
cd ..

cd nerva-numpy
cd examples
call cifar10.bat
call mnist.bat
cd ..
cd ..

cd nerva-tensorflow
cd examples
call cifar10.bat
call mnist.bat
cd ..
cd ..

cd nerva-torch
cd examples
call cifar10.bat
call mnist.bat
cd ..
cd ..
