from numba import jit
import numpy as np

@jit(nopython=True)
def gridlookup(n, grid, valplace): #Is there a np function for this? 
    """Check in which interval/ capital state valplace is"""
    
    ilow = 1
    ihigh = n-1

    distance = 2

    while (distance > 1):
        
        inow = np.floor((ilow+ihigh)/2)
        valnow = grid[int(inow)]

        # The strict inequality here ensures that grid[iloc] is less than or equal to valplace
        if (valnow > valplace):
            ihigh = inow
        else:
            ilow = inow

        distance = ihigh - ilow
    
    iloc = ilow
    
    return int(iloc)