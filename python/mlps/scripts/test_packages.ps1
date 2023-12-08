# Run test scripts

Set-Location nerva-jax\examples
Write-Host Testing nerva-jax
.\cifar10.bat
.\mnist.bat
Set-Location ..\..

Set-Location nerva-numpy\examples
Write-Host Testing nerva-numpy
.\cifar10.bat
.\mnist.bat
Set-Location ..\..

Set-Location nerva-tensorflow\examples
Write-Host Testing nerva-tensorflow
.\cifar10.bat
.\mnist.bat
Set-Location ..\..

Set-Location nerva-torch\examples
Write-Host Testing nerva-torch
.\cifar10.bat
.\mnist.bat
Set-Location ..\..
