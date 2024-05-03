# some fns
import numpy as np
import torch

def logit_fn(x):
    """
    Applies the sigmoid function element-wise to the input array x.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Output array with the same shape as x.
    """
    rv = np.exp(x)/(1+np.exp(x))
    return rv


