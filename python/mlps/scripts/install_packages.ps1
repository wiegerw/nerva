# Install packages using pip for each directory

Set-Location nerva-jax
pip install . --user
Set-Location ..

Set-Location nerva-numpy
pip install . --user
Set-Location ..

Set-Location nerva-tensorflow
pip install . --user
Set-Location ..

Set-Location nerva-torch
pip install . --user
Set-Location ..

Set-Location nerva-sympy
pip install . --user
Set-Location ..
