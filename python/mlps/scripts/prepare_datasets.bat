@echo off

rem download datasets using PyTorch
cd nerva-torch\examples
call prepare_datasets.bat
cd ..

rem add symbolic links for the other frameworks
cd ..\nerva-jax
mklink /d data ..\nerva-torch\data

cd ..\nerva-numpy
mklink /d data ..\nerva-numpy\data

cd ..\nerva-tensorflow
mklink /d data ..\nerva-tensorflow\data

