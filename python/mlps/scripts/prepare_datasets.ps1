# download datasets using PyTorch

Set-Location nerva-torch\examples
.\prepare_datasets.bat

Set-Location ..

# add symbolic links for the other frameworks
Set-Location ..\nerva-jax
New-Item -ItemType SymbolicLink -Target "..\nerva-torch\data" -Path "data"

Set-Location ..\nerva-numpy
New-Item -ItemType SymbolicLink -Target "..\nerva-numpy\data" -Path "data"

Set-Location ..\nerva-tensorflow
New-Item -ItemType SymbolicLink -Target "..\nerva-tensorflow\data" -Path "data"
