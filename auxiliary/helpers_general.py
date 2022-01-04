from numba import jit, guvectorize, float64

@jit(nopython=True)
def gridlookup(n, grid, valplace):
    """Check in which interval/ capital state valplace is"""
    
    ilow = 1
    ihigh = n-1

    distance = 2

    while (distance > 1):
        
        inow = int((ilow+ihigh)/2)
        valnow = grid[inow]

        # The strict inequality here ensures that grid[iloc] is less than or equal to valplace
        if (valnow > valplace):
            ihigh = inow
        else:
            ilow = inow

        distance = ihigh - ilow
    
    iloc = ilow
    
    return iloc

# @guvectorize([(float64, float64[:], float64[:], float64[:])], '(),(n),(m)->(m)',nopython=True)
def gridlookup_nb(n, grid, valplace):
    """Check in which interval/ capital state valplace is"""
    
    ilow = 1
    ihigh = n-1

    distance = 2

    while (distance > 1):
        
        inow = int((ilow+ihigh)/2)
        valnow = grid[inow]

        # The strict inequality here ensures that grid[iloc] is less than or equal to valplace
        if (valnow > valplace):
            ihigh = inow
        else:
            ilow = inow

        distance = ihigh - ilow
    
    iloc = ilow

    return iloc
    