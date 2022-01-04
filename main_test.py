import numpy as np
from auxiliary.Model import Model
from auxiliary.FirmInvestmentModel import FirmInvestmentModel, solve_model_notreshaped
from auxiliary.tauchen import approx_markov



if __name__=='__main__':

    
    seed = 10082021
    np.random.seed(seed)

    alpha_default = 0.5
    delta_default = 0.05
    
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
        "max_iter" : 300, 
        "precision" : 1e-5, 
        "size_capital_grid" : 101, 
    }

    sim_param = {
        "number_firms" : 10, 
        "number_simulations_per_firm" : 1, 
        "number_years_per_firm" : 30, 
        "burnin" : 200, 
        "seed" : 10082021
    }

    visualization_param = {
        "alpha grid bounds" : (0.35, 0.65), 
        "delta grid bounds" : (0.01,0.1),
        "fixed alpha" : alpha_default, 
        "fixed delta" : delta_default, 
        "parameter grid size" : 15
    }

    model = Model(deep_param, discretization_param, approx_param)


    # model.visualize_model_sol(alpha, delta, approx_param)
    model.visualize_mom_sensitivity(visualization_param, sim_param)
