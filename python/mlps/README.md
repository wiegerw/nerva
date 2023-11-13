# The PACKAGE library

**_NOTE:_**  This page is a stub. The library has not yet been released!

This repository contains an implementation of multilayer perceptrons in FRAMEWORK.
It is part of a group of five Python packages that can be installed via pip:

* [nerva-jax](https://pypi.org/project/nerva_jax/) An implementation in [JAX](https://jax.readthedocs.io).
* [nerva-numpy](https://pypi.org/project/nerva_numpy/) An implementation in [NumPy](https://numpy.org).
* [nerva-sympy](https://pypi.org/project/nerva_sympy/) An implementation in [SymPy](https://www.sympy.org), used for validation and testing.
* [nerva-tensorflow](https://pypi.org/project/nerva_tensorflow/) An implementation in [TensorFlow](https://www.tensorflow.org/).
* [nerva-torch](https://pypi.org/project/nerva_torch/) An implementation in [PyTorch](https://pytorch.org/). 

The packages can be installed standalone, except for `nerva-sympy` which requires
installation of the other four. Each package has its own GitHub repository:

* [https://github.com/wiegerw/nerva-jax](https://github.com/wiegerw/nerva-jax/)
* [https://github.com/wiegerw/nerva-numpy](https://github.com/wiegerw/nerva-numpy/)
* [https://github.com/wiegerw/nerva-sympy](https://github.com/wiegerw/nerva-sympy/)
* [https://github.com/wiegerw/nerva-tensorflow](https://github.com/wiegerw/nerva-tensorflow/)
* [https://github.com/wiegerw/nerva-torch](https://github.com/wiegerw/nerva-torch/)

## Purpose

The main purpose of these repositories is on the practical implementation of
neural networks. We aim to achieve the following goals: 
* To provide precise mathematical specifications of the execution of multilayer perceptrons.
* To provide an overview of the equations for common layers, activation functions and loss functions.
* To provide easily understandable implementations that match the specifications closely.
 
An important advantage of our implementations is that they 
are fully transparent: even the implementation of
backpropagation is provided in an accessible manner. Currently, the scope is
limited to multilayer perceptrons. However, the approach can easily be
generalized to more complex neural network architectures.

## Documentation

TODO

## Installation

The code is available as the PyPI package [CONEstrip](https://pypi.org/project/CONEstrip/).
It can be installed using

```
pip install PACKAGE
```

## Licensing

The code is available under the [Boost Software License 1.0](http://www.boost.org/LICENSE_1_0.txt).
A [local copy](https://github.com/wiegerw/PACKAGE/blob/main/LICENSE) is included in the repository.

## Using the library

### Multilayer perceptrons
Our multilayer perceptron class has a straightforward implementation:
```python
class MultilayerPerceptron(object):

    def feedforward(self, X: Matrix) -> Matrix:
        for layer in self.layers:
            X = layer.feedforward(X)
        return X

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        for layer in reversed(self.layers):
            layer.backpropagate(Y, DY)
            Y, DY = layer.X, layer.DX

    def optimize(self, lr: float):
        for layer in self.layers:
            layer.optimize(lr)
```
A multilayer perceptron can be constructed like this:
```python
    M = MultilayerPerceptron()

    input_size = 3072
    output_size = 1024
    act = AllReLUActivation(0.3)
    layer = ActivationLayer(input_size, output_size, act)
    layer.set_optimizer('Momentum(0.9)')
    layer.set_weights('Xavier')   
    M.layers.append(layer)
    ...
```
An example of a layer is the softmax layer. Note that the feedforward and
backpropagation implementations exactly match with the equations given in the documentation.
```python
class SoftmaxLayer(LinearLayer):

    def feedforward(self, X: Matrix) -> Matrix:
        ...
        Z = W @ X + column_repeat(b, N)
        Y = softmax_colwise(Z)
        ...

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        ...
        DZ = hadamard(Y, DY - row_repeat(diag(Y.T @ DY).T, K))
        DW = DZ @ X.T
        Db = rows_sum(DZ)
        DX = W.T @ DZ
        ...
```
The gradients computed by backpropagation are used to update the parameters of
the neural network using an *optimizer*. The user has full control over how to
update them. The composite design pattern is used to achieve this:
```python
    optimizer_W = MomentumOptimizer(layer.W, layer.DW, 0.9)
    optimizer_b = NesterovOptimizer(layer.b, layer.Db, 0.9)
    layer.optimizer = CompositeOptimizer(optimizer_W, optimizer_b)
```
Here `W` and `b` are the weights and bias of a linear layer, and `DW` and
`Db` are the gradients of these parameters with respect to the loss function.
Other parameters can be learned as well, see for example the SReLU layer.

### Training

For training of an MLP we provide an implementation of stochastic gradient descent
in the file `training.py` that looks like this:

```python
def stochastic_gradient_descent(M: MultilayerPerceptron,
                                epochs: int,
                                loss: LossFunction,
                                learning_rate: LearningRateScheduler,
                                train_loader: DataLoader):

    for epoch in range(epochs):
        lr = learning_rate(epoch)
        for (X, T) in train_loader:
            Y = M.feedforward(X)
            DY = loss.gradient(Y, T) / Y.shape[0]  # divide by the number of examples
            M.backpropagate(Y, DY)
            M.optimize(lr)
```

Here `train_loader` is an object similar to a `DataLoader` in PyTorch that splits
a dataset into batches of inputs `X` and expected outputs `T`.
Training of an MLP consists of three steps:
1. **feedforward** Given an input batch `X` and expected outputs `T`, compute the output `Y`.
2. **backpropagation** Given outputs `Y` and expected outputs `T`, compute the
 gradient of the MLP parameters with respect to a given loss function `L`.
3. **optimization** Update the MLP parameters using the gradient computed in step 2.

These steps are performed for each input batch of a dataset, and this process is
repeated `epoch` times.

#### Command line script
For convenience, a command line script `tools/mlp.py` is included that can be
used to do a training experiment. An example invocation of this script is
provided in `examples/cifar10.sh`:

```bash
mlp.py --layers="ReLU;ReLU;Linear" \
       --sizes="3072,1024,512,10" \
       --optimizers="Momentum(0.9);Momentum(0.9);Momentum(0.9)" \
       --init-weights="Xavier,Xavier,Xavier" \
       --batch-size=100 \
       --epochs=10 \
       --loss=SoftmaxCrossEntropy \
       --learning-rate="Constant(0.01)" \
       --dataset="cifar10.npz"
```
The output of this script may look like this:
``` text
$ ./cifar10.sh 
Loading dataset from file ../data/cifar10.npz
epoch   0  lr: 0.01000000  loss: 2.49815798  train accuracy: 0.10048000  test accuracy: 0.10110000  time: 0.00000000s
epoch   1  lr: 0.01000000  loss: 1.64330590  train accuracy: 0.41250000  test accuracy: 0.40890000  time: 11.27521224s
epoch   2  lr: 0.01000000  loss: 1.54620886  train accuracy: 0.44674000  test accuracy: 0.43910000  time: 11.33507117s
epoch   3  lr: 0.01000000  loss: 1.46849191  train accuracy: 0.47462000  test accuracy: 0.46280000  time: 11.30941587s
epoch   4  lr: 0.01000000  loss: 1.40283990  train accuracy: 0.49964000  test accuracy: 0.48370000  time: 11.30728618s
epoch   5  lr: 0.01000000  loss: 1.36808932  train accuracy: 0.51214000  test accuracy: 0.49030000  time: 11.32811917s
epoch   6  lr: 0.01000000  loss: 1.33309329  train accuracy: 0.52786000  test accuracy: 0.49490000  time: 11.27888232s
epoch   7  lr: 0.01000000  loss: 1.31322646  train accuracy: 0.53440000  test accuracy: 0.49800000  time: 11.29106200s
epoch   8  lr: 0.01000000  loss: 1.29327416  train accuracy: 0.53940000  test accuracy: 0.49850000  time: 11.42966828s
epoch   9  lr: 0.01000000  loss: 1.27069771  train accuracy: 0.55052000  test accuracy: 0.50120000  time: 11.30879241s
epoch  10  lr: 0.01000000  loss: 1.24060512  train accuracy: 0.56160000  test accuracy: 0.50690000  time: 11.42121017s
Total training time for the 10 epochs: 113.28471981s
```

### Validation
We do not rely on auto differentiation for the backpropagation. Instead, we provide
explicit implementations for it. Since the derivation of these equations is
error-prone, the results have to be validated. A common way to do this is to
rely on gradient checking using numerical approximations. Instead, we are using
symbolic differentation of SymPy. This is provided in the `nerva-sympy` package.
Gradient checking of the softmax layer looks like this: 

```python
    # backpropagation
    DZ = hadamard(Y, DY - row_repeat(diag(Y.T * DY).T, K))
    DW = DZ * X.T
    Db = rows_sum(DZ)
    DX = W.T * DZ

    # check gradients using symbolic differentiation
    DW1 = diff(loss(Y), w)
    Db1 = diff(loss(Y), b)
    DX1 = diff(loss(Y), x)
    DZ1 = diff(loss(Y), z)
    self.assertTrue(equal_matrices(DW, DW1))
    self.assertTrue(equal_matrices(Db, Db1))
    self.assertTrue(equal_matrices(DX, DX1))
    self.assertTrue(equal_matrices(DZ, DZ1))
```
An advantage of this approach is that errors can be found in an early stage.
Moreover, the source of such errors can be detected by checking intermediate
results.
