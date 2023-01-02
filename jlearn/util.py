"""
Utility functions for manipulating data.

Often an extension of numpy.
"""

import numpy as np

def shape_size(x: tuple[int]) -> int:
    """ Get the product of a tuple's elements
    
    Parameters
    ----------
    x : tuple of ints

        
    Returns
    -------
    The product
    """
    t = 1
    for s in x:
        t *= s
    return t

def multi_reshape(x: np.ndarray, shapes: list[tuple[int]]) -> list[np.ndarray]:
    """ reshapes and splits a vector into multiple matracies 
    
    Parameters
    ---------
    TODO
    """
    
    shape_sizes = [shape_size(t) for t in shapes]
    
    # check for a valid partition
    if sum(shape_sizes) != len(x):
        raise Exception(f"Invalid partition from len(x)={len(x)} to"\
            f" {sum(shape_sizes)}")
        
    out = list()
    s = 0 
    for i in range(len(shapes)):
        t = x[s : s+shape_sizes[i]]
        s += shape_sizes[i]
        out.append(t.reshape(shapes[i]))
        
    return out
