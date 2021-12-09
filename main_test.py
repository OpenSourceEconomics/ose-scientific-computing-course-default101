import numpy as np
from auxiliary.Model import Model
from auxiliary.FirmInvestmentModel import FirmInvestmentModel, solve_model_notreshaped
from auxiliary.tauchen import approx_markov



if __name__=='__main__':

    
    seed = 10082021
    np.random.seed(seed)
    
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
        "number_firms" : 10, 
        "number_simulations_per_firm" : 1, 
        "number_years_per_firm" : 30, 
        "burnin" : 200, 
        "seed" : 10082021
    }

    alpha = 0.5
    delta = 0.05

    model = Model(deep_param, discretization_param)
    model._solve_model(alpha, delta, approx_param)

    nz = 11         # Number of grid points for TFP
    emean = 0.0     # Mean of innovations to firm TFP
    sigma_z = 0.15  # Standard Deviations of innovations
    rho = 0.7       # TFP persistence
    multiple = 2.575 # Number of standard deviations the grid should span

    print("half")

    # Productivity shock grid and transition probability matrix
    # mgrid, pr_mat_m = approx_markov(rho, sigma_z, multiple, nz)
    # mgrid = np.exp(mgrid)
    # FIM = FirmInvestmentModel(alpha, delta, mgrid, pr_mat_m)

    # V, polind, pol, equity_iss = solve_model_notreshaped(FIM)