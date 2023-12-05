@echo off

cd nerva-jax
call pip install . --user
cd ..

cd nerva-numpy
call pip install . --user
cd ..

cd nerva-tensorflow
call pip install . --user
cd ..

cd nerva-torch
call pip install . --user
cd ..

cd nerva-sympy
call pip install . --user
cd ..
