import numpy as np
import pandas as pd

from auxiliary.Model import Model
from auxiliary.Sample import Sample

from auxiliary.estimation_func import *
from auxiliary.helpers_calcmoments import *
from auxiliary.helpers_plotting import *
from auxiliary.helpers_general import gridlookup_nb

if __name__=='__main__':
    
    seed = 10082021
    np.random.seed(seed)

    alpha = 0.5190
    delta = 0.0437

    deep_param = {
        "beta" : 0.96,
        "gamma": 0.05,
        "rho" : 0.75, 
        "sigma" : 0.3 # prev 0.15
    }

    discretization_param = {
        "size_shock_grid" : 11, 
        "range_shock_grid" : 2.575
    }

    approx_param = {
        "max_iter" : 1000, # prev 30 
        "precision" : 1e-4, 
        "size_capital_grid" : 101, 
    }

    sim_param = {
        "number_firms" : 1000, 
        "number_simulations_per_firm" : 1, 
        "number_years_per_firm" : 10, 
        "burnin" : 200, 
        "seed" : 10082021
    }

    visualization_param = {
        "alpha grid bounds" : (alpha-0.15, alpha+0.15), 
        "delta grid bounds" : (delta-0.02, delta+0.02),
        "fixed alpha" : alpha, 
        "fixed delta" : delta, 
        "parameter grid size" : 20,
        "bounds_optimizer": [[0.001,1], [0.001, 0.3]],
    }

    mom_param = {
    "no_moments" : 3, 
    "no_param" : 2,
    }


    model = Model(deep_param, discretization_param, approx_param)
    sample = Sample(mom_param)

    model.visualize_model_sol(visualization_param)
    model.visualize_simulated_capital(visualization_param, sim_param)
    # model.visualize_mom_sensitivity(visualization_param, sim_param)

    visualize_model_fit(sample, model, alpha, delta, sim_param)