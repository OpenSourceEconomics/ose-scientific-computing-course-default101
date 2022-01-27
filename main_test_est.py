import numpy as np
import time
from auxiliary.Model import Model
from auxiliary.Sample import Sample
from auxiliary.tauchen import approx_markov
from auxiliary.estimation_func import *
from auxiliary.helpers_calcmoments import import_data

if __name__=='__main__':

    seed = 10082021
    np.random.seed(seed)

    # alpha = 0.5
    # delta = 0.05

    deep_param = {
        "beta" : 0.96,
        "gamma": 0.05,
        "rho" : 0.75, 
        "sigma" : 0.15
    }

    discretization_param = {
        "size_shock_grid" : 11, 
        "range_shock_grid" : 2.575
    }

    approx_param = {
        "max_iter" : 1000, 
        "precision" : 1e-5, 
        "size_capital_grid" : 101, 
    }

    sim_param = {
        "number_firms" : 3, 
        "number_simulations_per_firm" : 1, 
        "number_years_per_firm" : 10, 
        "burnin" : 200, 
        "seed" : 10082021,
        "bounds_optimizer": [[0.001,1], [0.001, 0.3]],
    }

    visualization_param = {
        "alpha grid bounds" : (0.35, 0.65), 
        "delta grid bounds" : (0.03,0.07),
        "fixed alpha" : 0.5, 
        "fixed delta" : 0.05, 
        "parameter grid size" : 20
    }

    mom_param = {
        "no_moments" : 3, 
        "no_param" : 2,
    }

    model = Model(deep_param, discretization_param, approx_param)
    sample = Sample(mom_param)

    # model._solve_model(alpha, delta, approx_param)

    # model.visualize_model_sol(alpha, delta, approx_param)
    # model.visualize_mom_sensitivity(visualization_param, sim_param)

    # print(sample.sample_mom)

    est, sim_mom = get_estimation_results(sample, model, sim_param)
    print(f'{est=}')
    print(f'{sim_mom=}')

    # test1, test2 = get_estimation_results(sample, model, sim_param)
    # print(f'{test1=}')
    # print(f'{test2=}')
