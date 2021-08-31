import numpy as np


def _change_void_val(x):
    x = np.array(x)
    x[x == 255] = 1
    return x
