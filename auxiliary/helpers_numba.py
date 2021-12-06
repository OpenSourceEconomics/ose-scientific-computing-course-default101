import numpy as np
from numba import jit, int32

@jit(nopython=True)
def np_max_axis1(x):
    """Numba compatible version of np.amax(x, axis=1) and np.argmax(x,axis=1)."""
    temp = np.empty(x.shape[0])
    temp[:] = np.NaN
    ix = np.zeros(x.shape[0], dtype=int32)
    for i in range(x.shape[0]):
        temp[i] = np.max(x[i,:])
        ix[i] = np.argmax(x[i,:])
    return temp, ix

#@jit(nopython=True)
def np_cumsum_axis1(x):
    """Numba compatible version of np.cumsum(x, axis=1)."""
    temp = np.empty((x.shape[0],x.shape[1]))
    temp[:] = np.NaN
    for i in range(x.shape[0]):
        temp[i] = np.cumsum(x[i,:])
    return temp