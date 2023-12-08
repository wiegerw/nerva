# Run test scripts

Set-Location nerva-jax\examples
.\cifar10.bat
.\mnist.bat
Set-Location ..\..

Set-Location nerva-numpy\examples
.\cifar10.bat
.\mnist.bat
Set-Location ..\..

Set-Location nerva-tensorflow\examples
.\cifar10.bat
.\mnist.bat
Set-Location ..\..

Set-Location nerva-torch\examples
.\cifar10.bat
.\mnist.bat
Set-Location ..\..
