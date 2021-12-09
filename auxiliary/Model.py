import numpy as np
from scipy.stats import norm

class Model:
    """
    Class definition
    
    """

    def __init__(self, deep_param, discretization_param) -> None:

        self.beta = deep_param["beta"]
        self.gamma = deep_param["gamma"]
        self.r_inv = self.beta/(1-self.beta)
        # self.rho = deep_param["rho"]
        # self.sigma = deep_param["sigma"] 

        self.nz = discretization_param["size_shock_grid"]
        # self.range_z = discretization_param["range_shock_grid"]

        self.shock_grid, self.shock_transition = self._create_shock_grid(
            rho=deep_param["rho"],
            sigma=deep_param["sigma"],
            m=discretization_param["range_shock_grid"],
            n=discretization_param["size_shock_grid"])

        pass

    def _solve_model(self, alpha, delta, approx_param, verbose=True, print_skip=25): 
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
        
        max_iter = approx_param["max_iter"]
        precision = approx_param["precision"]
        nk = approx_param["size_capital_grid"]

        # Initialize the guess for the value function
        Vold = np.zeros(self.nz*nk)

        i = 0
        solerr = precision + 1
        

        while i < max_iter and solerr > precision:
            
            EVmat = np.kron((np.reshape(Vold, (self.nz, nk)).T@terry.pr_mat_m.T), np.ones((1, nk))).T

            RHSmat = terry.Rmat + self.beta*EVmat
            V, policy_func = np_max_axis1(RHSmat)
            # pol = terry.kvec[polind]
            
            solerr = np.max(np.abs(V-Vold))
            # polerr = np.max(np.abs(pol-polold))

            i += 1
            if verbose and i % print_skip == 0:
                print(i , "(",  solerr,' error)')
            
            minshift = self.r_inv*np.min(V-Vold)
            maxshift = self.r_inv*np.max(V-Vold)
            Vold = V+(maxshift + minshift)/2
            
        if i == max_iter:
            print("Failed to converge!")

        if verbose and i < max_iter:
            print("\nConverged in", i, " iterations.")


        return value_func, policy_func

    def _simulate_model(self):
        pass

    def _create_shock_grid(self, rho, sigma, m=3, n=7):
        """
            from filename: tauchen.py
            authors: Thomas Sargent, John Stachurski
            
            Computes the Markov matrix associated with a discretized version of
            the linear Gaussian AR(1) process

            y_{t+1} = rho * y_t + u_{t+1}

            according to Tauchen's method.  Here {u_t} is an iid Gaussian
            process with zero mean.

            Parameters
            ----------
            rho : scalar(float)
                The autocorrelation coefficient
            sigma_u : scalar(float)
                The standard deviation of the random process
            m : scalar(int), optional(default=3)
                The number of standard deviations to approximate out to
            n : scalar(int), optional(default=7)
                The number of states to use in the approximation

            Returns
            -------

            x : array_like(float, ndim=1)
                The state space of the discretized process
            P : array_like(float, ndim=2)
                The Markov transition matrix where P[i, j] is the probability
                of transitioning from x[i] to x[j]

        """
        F = norm(loc=0, scale=sigma).cdf

        # standard deviation of y_t
        std_y = np.sqrt(sigma**2 / (1-rho**2))

        # top of discrete state space
        x_max = m * std_y

        # bottom of discrete state space
        x_min = - x_max

        # discretized state space
        x = np.linspace(x_min, x_max, n)

        step = (x_max - x_min) / (n - 1)
        half_step = 0.5 * step
        P = np.empty((n, n))

        P[:,0] = F(x[0] - rho * x + half_step)
        P[:,n-1] = 1 - F(x[n-1] - rho * x - half_step)

        z = np.add.outer(-rho * x, x[1:n-1])
        P[:,1:n-1] = F(z + half_step) - F(z - half_step)

        return x, P

    def _create_capital_grid(self, alpha, delta, nk):
        # Define the capital grid

        k_low, k_high = (alpha * self.shock_grid[[0, self.nz-1]]
            * self.beta / (1 - (1 - delta) * self.beta))**(1 / (1 - alpha))

        capital_grid = np.linspace(0.5*k_low, k_high, nk)

        # Create grid_val
        grid_val = np.empty((self.nz * nk, 2))
        grid_val[:,0] = np.tile(capital_grid, self.nz)
        grid_val[:,1] = np.repeat(self.shock_grid, nk)

        # Create grid_ind
        # grid_ind = np.empty((self.nz*nk,2))
        # grid_ind[:,0] = np.tile(np.arange(nk), self.nz)
        # grid_ind[:,1] = np.repeat(np.arange(self.nz), nk)

        return capital_grid, grid_val#, grid_ind

    def _create_CF_grid(self, alpha, delta, nk):
        
        capital_grid, grid_val = self._create_capital_grid(alpha, delta, nk)

        CFfirm_mat = np.empty((self.nz*nk, nk))

        for i in range(nk):
            CFfirm_mat[:,i] = grid_val[:,1]*grid_val[:,0]**alpha + (1-delta) * grid_val[:,0] - capital_grid[i]

        CF_share_mat = CFfirm_mat.ravel()
        CF_share_mat[CF_share_mat<0] = (1+self.gamma)*CF_share_mat[CF_share_mat<0]

        # Static payoff is shareholder payoff
        Rmat = np.reshape(CF_share_mat, (self.nz*nk, nk))
        return Rmat

    def _get_full_model_sol(self):
        # optimal policy
        pol = terry.kvec[polind]

        # Select equity issuance based on optimised policy index
        equity_iss = np.empty(len(polind))
        for i in np.arange(0, terry.statenum):
            equity_iss[i] = terry.CFfirm_mat[i, polind[i]]

        equity_iss[equity_iss >= 0] = 0
        equity_iss[equity_iss < 0] = -equity_iss[equity_iss<0]
        return Vold, polind, pol, equity_iss

    def _get_mom_sensitivity(self):
        pass

    def visualize_model_sol(self):
        pass

    def visualize_mom_sensitivity(self):
        pass