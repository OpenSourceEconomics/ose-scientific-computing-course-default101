

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
    * mgrid:    # grid of productivity shocks (name remniscent of the first model Terry presented)
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