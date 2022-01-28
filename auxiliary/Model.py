import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from auxiliary.helpers_plotting import line_plot, threedim_plot, plot_mom_sensitivity
from auxiliary.helpers_calcmoments import calc_moments, demean_by_index
from auxiliary.helpers_general import gridlookup, gridlookup_nb

class Model:
    """
    Class definition
    
    """

    def __init__(self, deep_param, discretization_param, approx_param) -> None:

        self.beta = deep_param["beta"]
        self.gamma = deep_param["gamma"]
        self.r_inv = self.beta/(1-self.beta) 
        
        self.nz = discretization_param["size_shock_grid"]
        self.shock_grid, self.shock_transition = self._create_shock_grid(
            rho=deep_param["rho"],
            sigma=deep_param["sigma"],
            m=discretization_param["range_shock_grid"],
            n=discretization_param["size_shock_grid"])
        self.shock_stat_distr, self.shock_cum_transition = self._get_shock_stat_distr()
        
        self.max_iter = approx_param["max_iter"]
        self.precision = approx_param["precision"]
        self.nk = approx_param["size_capital_grid"]

    def _solve_model(self, alpha, delta, renew_a_d_flag=True, verbose=False, print_skip=250): 
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
        
        if renew_a_d_flag:
            self._create_capital_grid(alpha, delta)
            self._create_CF_grid(alpha, delta)

        # Initialize the guess for the value function
        value_func = np.zeros(self.nz*self.nk)

        i = 0
        solerr = self.precision + 1

        while i < self.max_iter and solerr > self.precision:
            
            EVmat = np.kron((np.reshape(value_func,\
                 (self.nz, self.nk)).T@self.shock_transition.T),\
                      np.ones((1, self.nk))).T

            RHSmat = self.Rmat + self.beta*EVmat
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

        if i == self.max_iter:
            print("Failed to converge!")

        if verbose and i < self.max_iter:
            print("\nConverged in", i, " iterations with error ", solerr)


        return value_func, policy_func

    def _get_shock_stat_distr(self):
        # At time 1, for each firm, draws an initial z process from the stationary distribution of z (stationary means 10000 draws)
        pi00 = np.linalg.matrix_power(self.shock_transition, 10000)
        pi0 = pi00[0,:-1]

        # cumulative distribution function of the stationary distribution
        shock_stat_distr = np.cumsum(pi0)
        shock_cum_transition = np.cumsum(self.shock_transition, axis=1)

        return shock_stat_distr, shock_cum_transition

    def _get_shock_series(self, sim_param):

        np.random.seed(sim_param["seed"])

        nfirms = sim_param["number_firms"]
        nyears = sim_param["number_years_per_firm"]
        nsim = sim_param["number_simulations_per_firm"]
        burnin = sim_param["burnin"]

        M = nfirms*nsim

        # Total time for the loop
        NyearsPerFirmsburn = nyears + burnin

        # vector that stores the value for z
        zsim = np.zeros((NyearsPerFirmsburn, M))
        izsim = np.zeros((NyearsPerFirmsburn, M), dtype=np.int8)
        
        
        runif = np.random.uniform(0,1, size=(NyearsPerFirmsburn, M))

        # Draw and store the initial values of z
        izsim[0,:] = np.sum(np.add.outer(runif[0,:], -self.shock_stat_distr) >= 0, axis=1)


        # Simulate the process for all the other z
        for t in range(1,NyearsPerFirmsburn):
            shock_cum_transition_t = self.shock_cum_transition[izsim[t-1, :], :]
            izsim[t,:] = np.sum((runif[t,:] - shock_cum_transition_t.T) >= 0, axis=0)

        # print(zsim, izsim)

        zsim = self.shock_grid[izsim]

        return zsim, izsim

    def _simulate_model(self, alpha, delta, sim_param, verbose=True):

        shock_series, shock_series_indices = self._get_shock_series(sim_param)
        _, policy_func = self._solve_model(alpha, delta, verbose=verbose)
        start_capital = self._get_start_capital()

        ksim = self._run_sim(alpha, delta, policy_func, start_capital, shock_series, shock_series_indices, sim_param)
        
        SimData = self._collect_sim_data(ksim, sim_param, shock_series, alpha, delta)

        return SimData

    def _get_start_capital(self):

        start_capital = (self.capital_grid[0] + self.capital_grid[-1])/2

        return start_capital

    def _run_sim(self, alpha, delta, policy_func, start_capital, shock_series, shock_series_indices, sim_param):
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
                # Reshape
        policy_func = np.reshape(policy_func, (self.nz, self.nk))
        
        nfirms = sim_param["number_firms"]
        nyears = sim_param["number_years_per_firm"]
        nsim = sim_param["number_simulations_per_firm"]
        burnin = sim_param["burnin"]
        
        M = nfirms*nsim

        # Total time for the loop
        NyearsPerFirmsburn = nyears + burnin

        # storage variables       
        ksim = np.zeros((NyearsPerFirmsburn+1, M))
        ksim[0, 0:M] = start_capital #do all firms start with the same capital? check def of k0val and terrys kvec


        for t in range(NyearsPerFirmsburn):
            kval = ksim[t,:]
            iz = shock_series_indices[t,:]
            iloc = gridlookup_nb(self.nk, self.capital_grid, kval)
            weight = (self.capital_grid[iloc+1] - kval) / (self.capital_grid[iloc+1] - self.capital_grid[iloc])
            weight [weight > 1] = 1
            # tmp = (kval >= self.capital_grid[iloc])
            # if not tmp.all():
            #     print(t)
            #     df = pd.DataFrame(np.vstack((tmp, kval, self.capital_grid[iloc], iloc))).transpose()
            #     df.rename(columns={list(df)[0]: 'true'}, inplace=True)
            #     print(df[ df['true'] < 1 ])

            kfval = policy_func[iz-1, iloc]*weight + policy_func[iz-1, iloc+1]*(1-weight)

            ksim[t+1, :] = kfval

        return ksim


    def _collect_sim_data(self, ksim, sim_param, shock_series, alpha, delta):
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

        nfirms = sim_param["number_firms"]
        nyears = sim_param["number_years_per_firm"]
        nsim = sim_param["number_simulations_per_firm"]
        burnin = sim_param["burnin"]
        M = nfirms*nsim

        # Total time for the loop
        NyearsPerFirmsburn = nyears + burnin
        
        ysim = np.zeros((NyearsPerFirmsburn, M))
        esim = np.zeros((NyearsPerFirmsburn, M))
        profitsim = np.zeros((NyearsPerFirmsburn, M))
        profitabilitysim = np.zeros((NyearsPerFirmsburn, M))
        investmentrate = np.zeros((NyearsPerFirmsburn, M))
        
        ysim = shock_series*ksim[:-1,:]**alpha
        profitabilitysim = ysim/ksim[:-1,:]
        investmentrate = ksim[1:,:] / ksim[:-1,:] - (1 - delta)
        profitsim = ysim - ksim[1:,:] + (1 - delta) * ksim[:-1,:]
        profitsim[profitsim < 0] = (1 + self.gamma) * profitsim[profitsim < 0]
        esim[profitsim < 0] = -profitsim[profitsim < 0]

        profitabilitysim_demeaned = demean_by_index(\
            profitabilitysim[burnin:burnin+nyears,:].T.flatten(), np.repeat(np.arange(M), nyears))
                    
        # Allocate final panel
        SimData = np.vstack( (\
                np.repeat(np.arange(M), nyears),
                np.tile(np.arange(nyears), M),
                profitabilitysim[burnin:burnin+nyears,:].T.flatten(),
                investmentrate[burnin:burnin+nyears,:].T.flatten(),
                profitabilitysim_demeaned,
                esim[burnin:burnin+nyears,:].T.flatten(),
                ysim[burnin:burnin+nyears,:].T.flatten(),
                ksim[burnin:burnin+nyears,:].T.flatten(),
                shock_series[burnin:burnin+nyears,:].T.flatten()
                )).T
        return SimData

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

    def _create_capital_grid(self, alpha, delta):
        # Define the capital grid

        k_low, k_high = (alpha * self.shock_grid[[0, self.nz-1]]
            * self.beta / (1 - (1 - delta) * self.beta))**(1 / (1 - alpha))

        capital_grid = np.linspace(0.5*k_low, k_high, self.nk)

        # Create grid_val
        grid_val = np.empty((self.nz * self.nk, 2))
        grid_val[:,0] = np.tile(capital_grid, self.nz)
        grid_val[:,1] = np.repeat(self.shock_grid, self.nk)

        self.capital_grid = capital_grid
        self.grid_val = grid_val

        return 1

    def _create_CF_grid(self, alpha, delta):

        CFfirm_mat = np.empty((self.nz*self.nk, self.nk))

        for i in range(self.nk):
            CFfirm_mat[:,i] = self.grid_val[:,1]*self.grid_val[:,0]**alpha + (1-delta) * self.grid_val[:,0] - self.capital_grid[i]

        CF_share_mat = CFfirm_mat.ravel()
        CF_share_mat[CF_share_mat<0] = (1+self.gamma)*CF_share_mat[CF_share_mat<0]

        # Static payoff is shareholder payoff
        Rmat = np.reshape(CF_share_mat, (self.nz*self.nk, self.nk))

        self.Rmat = Rmat
        self.CFfirm_mat = CFfirm_mat
        return 1

    def _get_full_model_sol(self, alpha, delta, renew_a_d_flag=True):

        value_func, policy_func = self._solve_model(alpha, delta, renew_a_d_flag=renew_a_d_flag)
        # optimal policy
        opt_policies = self.capital_grid[policy_func]

        # Select equity issuance based on optimized policy index
        equity_iss = np.empty(len(policy_func))
        for i in np.arange(self.nz*self.nk):
            equity_iss[i] = self.CFfirm_mat[i, policy_func[i]]

        equity_iss[equity_iss >= 0] = 0
        equity_iss[equity_iss < 0] = -equity_iss[equity_iss<0]
        return value_func, policy_func, opt_policies, equity_iss

    def _get_sim_moments(self, alpha, delta, sim_param):

        sim_data = self._simulate_model(alpha, delta, sim_param)

        xx = sim_param["number_firms"]*sim_param["number_years_per_firm"]

        moments = calc_moments(sim_data[0:xx,:])/sim_param["number_simulations_per_firm"]
        for s in range(1,sim_param["number_simulations_per_firm"]):
            moments = moments + \
                calc_moments(sim_data[xx*s:xx*(s+1),:])/sim_param["number_simulations_per_firm"]

        return moments

    def _get_objective_func(self, approx_param):

        f = lambda a, d: self._get_sim_moments(a,d, approx_param)

        return f

    def _get_mom_sensitivity(self, grid_alpha, mid_alpha, grid_delta, mid_delta, sim_param, no_moments=3):
        """
        Sensitivity of the model solution to alpha and delta.

        Args
        ----
        grid_alpha ():
        mid_alpha (np.float):   fixed alpha for sensitivity analysis for delta
        grid_delta ():
        mid_delta (np.float):   fixed delta for sensitivity analysis for alpha
        no_moments (int):       number of moments (default: 3)

        Returns
        -------
        out_alpha (numpy.ndarray):  simulated moments for varying levels of alpha, of shape (grid_points x 3)
        out_delta (numpy.ndarray):  simulated moments for varying levels of delta, of shape (grid_points x 3)
        """
        n_alpha = len(grid_alpha)
        n_delta = len(grid_delta)

        moments_alpha = np.empty((n_alpha, no_moments))
        moments_delta = np.empty((n_delta, no_moments))

        for i, alpha in enumerate(grid_alpha):
            moments_alpha[i,:] = self._get_sim_moments(alpha, mid_delta, sim_param)

        for i, delta in enumerate(grid_delta):
            moments_delta[i,:] = self._get_sim_moments(mid_alpha, delta, sim_param)

        return moments_alpha, moments_delta

    def visualize_model_sol(self, alpha, delta, renew_a_d_flag=True):
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

        # Solve the model
        value_func, _, opt_policies, equity_iss = self._get_full_model_sol(alpha, delta, renew_a_d_flag=renew_a_d_flag)

        # Reshape
        value_func = np.reshape(value_func, (self.nz, self.nk))
        opt_policies = np.reshape(opt_policies, (self.nz, self.nk))
        equity_iss = np.reshape(equity_iss, (self.nz, self.nk))

        # Create contour lines
        line_plot(self.capital_grid, value_func, self.nz, 'Firm Capital k', 'V(z,k)', "Firm Value, (alpha, delta) = ({}, {})".format(alpha, delta))
        line_plot(self.capital_grid, opt_policies, self.nz, 'Firm Capital k', 'kprime(z,k)', "Firm Capital Choice, (alpha, delta) = ({}, {})".format(alpha, delta))
        line_plot(self.capital_grid, equity_iss, self.nz, 'Firm Capital k', 'Negative Part(Equity Issuance)', "Firm Equity Issuance, (alpha, delta) = ({}, {})".format(alpha, delta))

        # Create 3D plots
        threedim_plot(self.shock_grid, self.capital_grid, value_func, "z", "k", "V(z,k)", "Firm Value, (alpha, delta) = ({}, {})".format(alpha, delta))
        threedim_plot(self.shock_grid, self.capital_grid, opt_policies, "z", "k", "kprime(z,k)", "Firm Capital Choice, (alpha, delta) = ({}, {})".format(alpha, delta))
        threedim_plot(self.shock_grid, self.capital_grid, equity_iss, "z", "k", "Negative Part(Equity Issuance)", "Negative Part(Equity Issuance), (alpha, delta) = ({}, {})".format(alpha, delta))
        
    def visualize_mom_sensitivity(self, visualization_param, sim_param):

        a_bounds = visualization_param["alpha grid bounds"]
        d_bounds = visualization_param["delta grid bounds"]
        mid_alpha = visualization_param["fixed alpha"]
        mid_delta = visualization_param["fixed delta"]
        ngrid = visualization_param["parameter grid size"]

        grid_alpha, grid_delta = self._get_a_d_grids(a_bounds, d_bounds, ngrid)
        
        x, y = self._get_mom_sensitivity(grid_alpha, mid_alpha, grid_delta, mid_delta, sim_param)

        xlabel = {
            'alpha' : 'Parameter alpha for fixed delta={}'.format(mid_delta),
            'delta' : 'Parameter delta for fixed alpha={}'.format(mid_alpha),
        }

        plot_mom_sensitivity((grid_alpha, grid_delta), (x,y), xlabel)

    def _get_a_d_grids(self, a_bounds, d_bounds, ngrid):

        grid_alpha = np.linspace(a_bounds[0], a_bounds[1], ngrid)
        grid_delta = np.linspace(d_bounds[0], d_bounds[1], ngrid)

        return grid_alpha, grid_delta

    def test_model_solve(self, alpha, delta):
        
        self.capital_grid = self._create_capital_grid(alpha, delta)

    def test_model_sim(self, alpha, delta, sim_param):

        out = self._get_sim_moments(alpha, delta, sim_param)

        return out


if __name__=='__main__':

    
    seed = 10082021
    np.random.seed(seed)

    alpha = 0.5
    delta = 0.05
    
    deep_param = {
        "beta" : 0.96,
        "gamma": 0.05,
        "rho" : 0.7, 
        "sigma" : 0.15
    }

    discretization_param = {
        "size_shock_grid" : 11, 
        "range_shock_grid" : 2.575
    }

    approx_param = {
        "max_iter" : 30, 
        "precision" : 1e-4, 
        "size_capital_grid" : 101, 
    }

    sim_param = {
        "number_firms" : 20, 
        "number_simulations_per_firm" : 1, 
        "number_years_per_firm" : 10, 
        "burnin" : 30, 
        "seed" : 10082021
    }

    visualization_param = {
        "alpha grid bounds" : (0.45, 0.55), 
        "delta grid bounds" : (0.04,0.06),
        "fixed alpha" : alpha, 
        "fixed delta" : delta, 
        "parameter grid size" : 3
    }

    model = Model(deep_param, discretization_param, approx_param)
    model._solve_model(alpha, delta, approx_param)

    model.visualize_mom_sensitivity(visualization_param, sim_param)
