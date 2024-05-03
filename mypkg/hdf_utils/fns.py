# this file contains some fns for beta(s)
import numpy as np
from numbers import Number

def fn1(x):
    """10*sin(6pi*x)
    """
    return 10*np.sin(2*np.pi*3*x)

def fn2(x):
    """10 * (exp(x)-x+sin(4pi*x))
    """
    p1 = np.exp(x) -x
    p2 = 1*np.sin(2*np.pi*2*x)
    return 10*(p1 + p2)

def fn3(x):
    """ x^3 - 3x^2 + 2x - 10
    """
    return x**3 - 3*x**2 + 2*x -10

def fn4(x):
    return 20*x 

def fn5(x):
    return -20*np.log(x**4+1)-6

def zero_fn(x):
    if isinstance(x, Number):
        return 0
    else:
        return np.zeros(len(x))