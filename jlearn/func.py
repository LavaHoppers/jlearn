"""
Vectorized functions to calculate activation and loss.
"""

import numpy as np

def _sigmoid(x):
    if x > 20:
        return 1
    elif x < -20:
        return 0
    return 1 / (1 + np.exp(-x))

sigmoid = np.vectorize(_sigmoid)
sigmoid.__doc__ = """
TODO
"""

ReLU = np.vectorize(lambda x: max(0, x))


