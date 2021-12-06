import numpy as np
from numba import jit, int32

firm_investment_data = [
    ('alpha', float64),          
    ('delta', float64),                
    ('beta', float64),              
    ('_lambda', float64),            
    ('nz', int32),
    ('nk', int32),
    ('max_iter', int32),
    ('precision', float64),
    ('mgrid', float64[:]),
    ('pr_mat_m', float64[:,:]),
    ('statenum', int32),
    ('grid_val', float64[:,:]),
    ('grid_ind', int32[:,:]),
    ('Rmat', float64[:,:]),
    ('CFfirm_mat', float64[:,:]),
    ('kvec', float64[:]),
    ]

@jitclass(firm_investment_data)
class FirmInvestmentModel:
    """
    Class definition.

    Input
    -----
    * alpha:    # degree of returns to scale (parameter of interest)
    * delta:    # rate of capital depreciation (parameter of interest)
    * beta:     # real discount factor
    * _lambda:  # linear cost of equity issuance
    * nz:       # number of grid points for tfp
    * nk:       # number of capital grid points
    * max_iter: # max iteration steps (before the solution routine stops and spits out "Did not converge")
    * precision:# precision of value function iteration
    * mgrid:    # grid of productivity shocks (name reminiscent of the first model Terry presented)
    * pr_mat_m: # transition probabilities (generated from Tauchen method)
    """
    
    def __init__(self,
            alpha,      
            delta,                
            beta = 0.96,              
            _lambda = 0.05,            
            nz = 11,
            nk = 101,
            max_iter = 1000,
            precision = 0.0001,
            mgrid = mgrid,
            pr_mat_m = pr_mat_m):
    
        self.alpha = alpha
        self.delta = delta
        self.beta = beta              
        self._lambda = _lambda            
        
        self.nz = nz
        self.nk = nk
        self.max_iter = max_iter
        self.precision = precision
        self.mgrid = mgrid
        self.pr_mat_m = pr_mat_m
        self.statenum = self.nk * self.nz

        # Define the capital grid
        klow = 0.5*(self.alpha*self.mgrid[0]*self.beta/(1-(1-self.delta)*self.beta))**(1/(1-self.alpha))
        khigh = (self.alpha*self.mgrid[self.nz-1]*self.beta/(1-(1-self.delta)*self.beta))**(1/(1-self.alpha))
        
        self.kvec = np.linspace(klow, khigh, self.nk)

        # Create grid_val
        # alternative for the following non-jittable version 
        # self.grid_val[:,0] = np.kron(np.ones(1, self.nz)), self.kvec)
        # self.grid_val[:,1] = np.kron(self.mgrid, np.ones((1, self.nk)))
        self.grid_val = np.empty((self.statenum,2)) 
        for i in range(self.nz):
            self.grid_val[i*self.nk : (i+1)*self.nk,0] = self.kvec
        self.grid_val[:,1] = np.repeat(self.mgrid, self.nk)

        # Create grid_ind
        # alternative for the following non-jittable version 
        # self.grid_ind[:, 0] = np.kron(np.ones((1, self.nz)), np.arange(0, self.nk))
        # self.grid_ind[:, 1] = np.kron(np.arange(0, self.nz), np.ones((1, self.nk)))
        self.grid_ind = np.empty((self.statenum,2),dtype=int32)
        for i in range(self.nz):
            self.grid_ind[i*self.nk : (i+1)*self.nk,0] = np.arange(0, self.nk, dtype=int32)
        self.grid_ind[:,1] = np.repeat(np.arange(0, self.nz, dtype=int32), self.nk)

        self.CFfirm_mat = np.empty((self.statenum, self.nk))
        for i in range(self.nk):
            self.CFfirm_mat[:,i] = self.grid_val[:,1]*self.grid_val[:,0]**self.alpha + (1-self.delta) * self.grid_val[:,0] - self.kvec[i]

        CF_share_mat = self.CFfirm_mat.ravel()
        CF_share_mat[CF_share_mat<0] = (1+self._lambda)*CF_share_mat[CF_share_mat<0]

        # Static payoff is shareholder payoff
        self.Rmat = np.reshape(CF_share_mat, (self.statenum, self.nk))


@jit(nopython=True)
def solve_model_notreshaped(terry):
    """
    Solves the model.

    Parameters
    ----------
    * terry:    instance of the class FirmInvestmentModel

    Output (not reshaped yet!)
    ------
    * Vold (vector of dimension terry.statenum):        value function
    * polind (vector of dimension terry.statenum):      index of optimal policy on the k grid
    * pol (vector of dimension terry.statenum)          optimal policy (point on the k grid)
    * equity_iss (vector of dimension terry.statenum)   negative part of equity issuance
    """
    
    # Initialize the guess for the value function
    Vold = np.zeros(terry.statenum)
    polindold = np.zeros(terry.statenum)
    # polold = polindold # commented out calc of pol during the loop

    i = 0
    solerr = terry.precision + 1
    verbose=True
    print_skip=25

    while i < terry.max_iter and solerr > terry.precision:
        
        EVmat = np.kron((np.reshape(Vold, (terry.nz, terry.nk)).T@terry.pr_mat_m.T), np.ones((1, terry.nk))).T

        RHSmat = terry.Rmat + terry.beta*EVmat
        V, polind = np_max_axis1(RHSmat)
        # pol = terry.kvec[polind]
        
        solerr = np.max(np.abs(V-Vold))
        # polerr = np.max(np.abs(pol-polold))

        i += 1
        if verbose and i % print_skip == 0:
            print(i , "(",  solerr,' error)')
        
        minshift = (terry.beta/(1-terry.beta))*np.min(V-Vold)
        maxshift = (terry.beta/(1-terry.beta))*np.max(V-Vold)
        Vold = V+(maxshift + minshift)/2
        polinold = polind
        # polold = pol
        
    if i == terry.max_iter:
        print("Failed to converge!")

    if verbose and i < terry.max_iter:
        print("\nConverged in", i, " iterations.")

    # optimal policy
    pol = terry.kvec[polind]

    # Select equity issuance based on optimised policy index
    equity_iss = np.empty(len(polind))
    for i in np.arange(0, terry.statenum):
        equity_iss[i] = terry.CFfirm_mat[i, polind[i]]

    equity_iss[equity_iss >= 0] = 0
    equity_iss[equity_iss < 0] = -equity_iss[equity_iss<0]

    return Vold, polind, pol, equity_iss