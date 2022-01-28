from numba import jit, guvectorize, float64
import numpy as np

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
# @jit(nopython=True)
def gridlookup_nb(n, grid, valplace):
    """Check in which interval/ capital state valplace is"""
    
    ilow = 0 # prev 1
    ihigh = n-1

    inow = int((ilow+ihigh)/2) * np.ones(len(valplace), dtype=int)
    distance = 2 * np.ones(len(valplace), dtype=int)

    while (distance > 1).any():

        valnow = grid[inow]

        # The strict inequality here ensures that grid[iloc] is less than or equal to valplace
        binary = valnow > valplace

        ihigh = binary * inow + (1 - binary) * ihigh
        ilow = binary * ilow + (1 - binary) * inow 

        distance = ihigh - ilow
        inow = ((ilow+ihigh)/2).astype(int)

    return ilow



if __name__=='__main__':

    print("Test session")

    from timeit import timeit

    sample_size = 20
    n = 101
    dummy_grid = np.linspace(0,1,n)
    dummy_vals = np.random.random(sample_size)
    iloc = np.empty(sample_size)

    @jit(nopython=True)
    def gridlookup_test(n, grid, vals, iloc):

        for i,x  in enumerate(vals):
            iloc[i] = gridlookup(n, grid, x)

        return iloc

    t = timeit('gridlookup_test(n,dummy_grid,dummy_vals, iloc)', 'from __main__ import gridlookup_test, n, dummy_grid, dummy_vals, iloc', number=100000)
    print(t)
    t = timeit('gridlookup_nb(n,dummy_grid,dummy_vals)', 'from __main__ import gridlookup_nb, n, dummy_grid, dummy_vals',number=100000)
    print(t)
    