import numpy as np
from scipy.stats import norm
from auxiliary.helpers_numba import np_max_axis1
from auxiliary.helpers_plotting import line_plot, threedim_plot

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
        value_func = np.zeros(self.nz*nk)
        Rmat, _, _ = self._create_CF_grid(alpha, delta, nk)

        i = 0
        solerr = precision + 1

        while i < max_iter and solerr > precision:
            
            EVmat = np.kron((np.reshape(value_func,\
                 (self.nz, nk)).T@self.shock_transition.T),\
                      np.ones((1, nk))).T

            RHSmat = Rmat + self.beta*EVmat
            V = np.amax(RHSmat, axis=1)

            value_diff = V - value_func
            
            solerr = np.max(np.abs(value_diff))
            minshift = np.min(value_diff)
            maxshift = np.max(value_diff)

            value_func = V + self.r_inv * (maxshift + minshift) / 2

            i += 1
            if verbose and i % print_skip == 0:
                print(i , "(",  solerr,' error)\n')
            
        policy_func = np.argmax(RHSmat, axis=1)

        if i == max_iter:
            print("Failed to converge!")

        if verbose and i < max_iter:
            print("\nConverged in", i, " iterations.")


        return value_func, policy_func

    def _get_shock_stat_distr(self):
        # At time 1, for each firm, draws an initial z process from the stationary distribution of z (stationary means 10000 draws)
        pi00 = np.linalg.matrix_power(self.shock_transition, 10000)
        pi0 = pi00[0,0:self.nz-1]

        # cumulative distribution function of the stationary distribution
        cumpi0 = np.cumsum(pi0) # here the original np.cumsum works, since it is a 1d array

        return cumpi0

    def _get_shock_series(self):

        
        # Total time for the loop
        NyearsPerFirmsburn = NYearsPerFirms + burnin

        # vector that stores the value for z
        zsim = np.zeros((NyearsPerFirmsburn, NFirms))
        izsim = zsim.copy()
        
        
        runif = np.random.uniform(0,1, size=(NyearsPerFirmsburn, NFirms))

        # Draw and store the initial values of z
        for i in range(0, NFirms):
            izsim[0,i] = np.sum(((runif[0, i] - cumpi0) >= 0)*1)
            izsim[0,i] = min([int(izsim[0,i]) + 1, nz])
            zsim[0,i]  = mgrid[int(izsim[0,i])]

        # Simulate the process for all the other z
        cumpiz = np_cumsum_axis1(terry.pr_mat_m)
        NyearsPerFirms1 = NyearsPerFirmsburn-1
        for i in range(0, NFirms):
            for t in range(0, NyearsPerFirms1):
                cumpizi = cumpiz[int(izsim[t, i])-1, :]
                izsim[t+1, i] = np.sum(((runif[t,i] - cumpizi) >= 0)*1)
                izsim[t+1, i] = min([izsim[t+1, i]+1, nz])
                zsim[t+1, i] = terry.mgrid[int(izsim[t+1, i])-1]
        pass

    def _setup_sim(self, sim_param):

        nfirms = sim_param["number_firms"]
        nyears = sim_param["number_years_per_firm"]
        nsim = sim_param["number_simulations_per_firm"]

        k0val = (terry.kvec[0] + terry.kvec[terry.nk-1])/2
        shock_series = self._get_shock_series()

        pass

    def _simulate_model(self):
        """
            Simulates a panel.

            Paramters
            ---------
            * terry:                    an instance of the class FirmInvestmentModel
            * NYearsPerFirm:            number of years per firm
            * NFirms:                   number of firms
            * seed:                     seed
            * burnin:                   burn-in period (i.e. number of first periods to be added first and then excluded in the final simulated panel)

            Output
            ------
            * Simdata (dimension (NYearsPerFirm x NFirms) x 8) with the following columns
            ** col1: id of the firm
            ** col2: year
            ** col3: profitability = z*(k^alpha)/K
            ** col4: investment rate = I/K
            ** col5: decision rule (issue equity or not)
            ** col6: y = zsim(t,i) * (k^alpha)
            ** col7: k
            ** col8: z
        """
        # Solve the model for the instance terry
        V, polind, pol, equity_iss = solve_model_notreshaped(terry)

        # Reshape
        pol = np.reshape(pol, (terry.nz, terry.nk))

        # Set seed
        np.random.seed(seed)


        # storage variables       
        ksim = np.zeros((NyearsPerFirmsburn+1, NFirms))
        ksim[0, 0:NFirms] = k0val #do all firms start with the same capital? check def of k0val and terrys kvec
        ysim = np.zeros((NyearsPerFirmsburn, NFirms))
        # Important to make deep copies not pointers (!)
        esim = ysim.copy() #?
        profitsim = esim.copy()
        profitabilitysim = esim.copy()
        investmentrate = esim.copy()
        
        # Simulate panel of firms
        # double loop on time and firm
        for i in range(0, NFirms):
            for t in range(0, NyearsPerFirmsburn):
                kval = ksim[t, i]
                iz = izsim[t, i]
                iloc = gridlookup(terry.nk, terry.kvec, kval)
                weight = (terry.kvec[iloc+1] - kval) / ((terry.kvec[iloc+1]) - terry.kvec[iloc])
                kfval = pol[int(iz)-1, iloc]*weight + pol[int(iz)-1, iloc+1]*(1-weight)

                ksim[t+1, i] = kfval

                ysim[t, i] = zsim[t, i-1]*(kval**terry.alpha)
                profitabilitysim[t, i] = ysim[t, i]/kval
                esim[t, i] = max(kfval - (ysim[t, i] + (1-terry.delta)*kval),  0)
                investmentrate[t, i] = (kfval - (1-terry.delta)*kval)/kval
                if (esim[t, i]>0):
                    esim[t, i] = esim[t, i]*(1+terry._lambda)
                    profitsim[t, i] = (1+terry._lambda)*(ysim[t, i] + (1-terry.delta)*kval - kfval)
                else:
                    profitsim[t, i] = ysim[t, i] + (1-terry.delta)*kval - kfval
                    
        
                    
        return SimData

    def _collect_sim_data(self):
        k = 8 # number of columns in Simdata
        SimData = np.zeros((NFirms*NYearsPerFirms, k)) # Allocate final panel

        # Fill final panel (see description of function)
        for id_n in range(0, NFirms):
            SimData[(id_n*NYearsPerFirms):(id_n+1)*NYearsPerFirms, ] = (
                np.concatenate((

                np.expand_dims(np.repeat(np.array([id_n]), NYearsPerFirms), axis = 1),
                np.expand_dims(np.arange(0, NYearsPerFirms), axis = 1),
                np.expand_dims(profitabilitysim[(burnin):(burnin+NYearsPerFirms), id_n], axis = 1),
                np.expand_dims(investmentrate[(burnin):(burnin+NYearsPerFirms), id_n], axis = 1),
                np.expand_dims(esim[(burnin):(burnin+NYearsPerFirms), id_n], axis = 1),
                np.expand_dims(ysim[(burnin):(burnin+NYearsPerFirms), id_n], axis = 1),
                np.expand_dims(ksim[(burnin):(burnin+NYearsPerFirms), id_n], axis = 1),
                np.expand_dims(zsim[(burnin):(burnin+NYearsPerFirms), id_n], axis = 1)

                ), axis = 1)
            )

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

        x = np.exp(x)

        return x, P

    def _create_capital_grid(self, alpha, delta, nk):
        # Define the capital grid

        k_low, k_high = (alpha * self.shock_grid[[0, self.nz-1]]
            * self.beta / (1 - (1 - delta) * self.beta))**(1 / (1 - alpha))

        capital_grid = np.linspace(0.5*k_low, k_high, nk)

        klow = 0.5*(alpha*self.shock_grid[0]*self.beta/(1-(1-delta)*self.beta))**(1/(1-alpha))
        khigh = (alpha*self.shock_grid[self.nz-1]*self.beta/(1-(1-delta)*self.beta))**(1/(1-alpha))
        
        kvec = np.linspace(klow, khigh, nk)

        np.testing.assert_array_equal(capital_grid, kvec)

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
        return Rmat, CFfirm_mat, capital_grid

    def _get_full_model_sol(self, alpha, delta, approx_param):

        nk = approx_param["size_capital_grid"]

        value_func, policy_func = self._solve_model(alpha, delta, approx_param)
        _, CFfirm_mat, capital_grid = self._create_CF_grid(alpha, delta, nk)
        # optimal policy
        opt_policies = capital_grid[policy_func]

        # Select equity issuance based on optimized policy index
        equity_iss = np.empty(len(policy_func))
        for i in np.arange(self.nz*nk):
            equity_iss[i] = CFfirm_mat[i, policy_func[i]]

        equity_iss[equity_iss >= 0] = 0
        equity_iss[equity_iss < 0] = -equity_iss[equity_iss<0]
        return value_func, policy_func, opt_policies, equity_iss, capital_grid

    def _get_mom_sensitivity(self):
        pass

    def visualize_model_sol(self, alpha, delta, approx_param):
        """
            Visualizes the model solution (in 2D and 3D) taking alpha and delta as input for instantiating the class FirmInvestmentModel.

            Input
            -----
            * alpha:    parameter of interest (capital's share of output)
            * delta:    parameter of interest (depreciation rate)
            
            Output
            ------
            plots
        """
        

        nk = approx_param["size_capital_grid"]
        # Solve the model
        value_func, _, opt_policies, equity_iss, capital_grid = self._get_full_model_sol(alpha, delta, approx_param)

        # Reshape
        value_func = np.reshape(value_func, (self.nz, nk))
        opt_policies = np.reshape(opt_policies, (self.nz, nk))
        equity_iss = np.reshape(equity_iss, (self.nz, nk))

        # Create contour lines
        line_plot(capital_grid, value_func, self.nz, 'Firm Capital k', 'V(z,k)', "Firm Value, (alpha, delta) = ({}, {})".format(alpha, delta))
        line_plot(capital_grid, opt_policies, self.nz, 'Firm Capital k', 'kprime(z,k)', "Firm Capital Choice, (alpha, delta) = ({}, {})".format(alpha, delta))
        line_plot(capital_grid, equity_iss, self.nz, 'Firm Capital k', 'Negative Part(Equity Issuance)', "Firm Equity Issuance, (alpha, delta) = ({}, {})".format(alpha, delta))

        # Create 3D plots
        threedim_plot(self.shock_grid, capital_grid, value_func, "z", "k", "V(z,k)", "Firm Value, (alpha, delta) = ({}, {})".format(alpha, delta))
        threedim_plot(self.shock_grid, capital_grid, opt_policies, "z", "k", "kprime(z,k)", "Firm Capital Choice, (alpha, delta) = ({}, {})".format(alpha, delta))
        threedim_plot(self.shock_grid, capital_grid, equity_iss, "z", "k", "Negative Part(Equity Issuance)", "Negative Part(Equity Issuance), (alpha, delta) = ({}, {})".format(alpha, delta))
        


    def visualize_mom_sensitivity(self):
        pass