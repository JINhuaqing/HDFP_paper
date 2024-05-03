import numpy as np
import matplotlib.pyplot as plt

def seq_cmap(n, type="Reds"):
    """
    Returns a sequential color map with n colors of the specified type.

    Parameters:
    n (int): The number of colors in the color map.
    type (str): The type of color map to use. Default is "Reds".
        Some good choice: YlOrRd, Greys,  
        More refers to https://matplotlib.org/stable/users/explain/colors/colormaps.html

    Returns:
    numpy.ndarray: An array of n colors in the specified color map.
    """
    cmap = plt.cm.get_cmap(type)
    cols = cmap(np.linspace(0, 1, n))
    return cols

def div_cmap(n, type="coolwarm"):
    """
    Returns a diverging color map with n colors of the specified type.

    Parameters:
    n (int): The number of colors in the color map.
    type (str): The type of color map to use. Default is "coolwarm".
        Some good choice: bwr, Spectral
        More refers to https://matplotlib.org/stable/users/explain/colors/colormaps.html

    Returns:
    numpy.ndarray: An array of n colors in the specified color map.
    """
    cmap = plt.cm.get_cmap(type)
    cols = cmap(np.linspace(0, 1, n))
    return cols

def qual_cmap(n):
    """
    Returns a qualtative color map with n colors of the specified type.

    Parameters:
    n (int): The number of colors in the color map.
    More refers to https://matplotlib.org/stable/users/explain/colors/colormaps.html

    Returns:
    numpy.ndarray: An array of n colors in the specified color map.
    """
    if n <= 8:
        cmap = plt.cm.get_cmap("Set2")
    elif n <= 20:
        cmap = plt.cm.get_cmap("tab20")
    else:
        raise ValueError(f'Not support n >20!')
    cols = cmap(np.linspace(0, 1, n))
    return cols
