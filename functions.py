"""
Sample code automatically generated on 2020-07-24 14:55:45

by www.matrixcalculus.org

from input

d/dx x .* tanh(log(vector(1) + exp(x))) = diag(tanh(log(vector(1)+exp(x))))+diag(x.*(vector(1)-tanh(log(vector(1)+exp(x))).^2).*exp(x)./(vector(1)+exp(x)))

where

x is a vector

The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

def fAndG(x):
    assert isinstance(x, np.ndarray)
    dim = x.shape
    assert len(dim) == 1
    x_rows = dim[0]

    t_0 = np.exp(x)
    t_1 = (np.ones(x_rows) + t_0)
    t_2 = np.tanh(np.log(t_1))
    functionValue = (x * t_2)
    gradient = (np.diag(t_2) + np.diag((((x * (np.ones(x_rows) - (t_2 ** 2))) * t_0) / t_1)))

    return functionValue, gradient

def checkGradient(x):
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    t = 1E-6
    delta = np.random.randn(3)
    f1, _ = fAndG(x + t * delta)
    f2, _ = fAndG(x - t * delta)
    f, g = fAndG(x)
    print('approximation error',
          np.linalg.norm((f1 - f2) / (2*t) - np.tensordot(g, delta, axes=1)))

def generateRandomData():
    x = np.random.randn(3)

    return x

if __name__ == '__main__':
    x = generateRandomData()
    functionValue, gradient = fAndG(x)
    print('functionValue = ', functionValue)
    print('gradient = ', gradient)

    print('numerical gradient checking ...')
    checkGradient(x)
