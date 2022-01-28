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

    alpha = 0.5
    delta = 0.05

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
        "number_simulations_per_firm" : 3, 
        "number_years_per_firm" : 10, 
        "burnin" : 200, 
        "seed" : 10082021,
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

    opt_param = {
        "solver" : "bobyqa",
        # "solver" : "dual_annealing",
        "start_value" : np.array([alpha, delta]),
        "bounds_optimizer": [[0.001,1], [0.001, 0.3]],
        "bounds_optimizer_bobyqa" : (np.array([0,0]), np.array([1,0.3])),
        "max_iter" : 10,
        "step_size" : np.finfo(float).eps**(1/3),
        "noise_range" : 10**(-np.arange(7)),
        "noisy_function_opt" : False
    }

    model = Model(deep_param, discretization_param, approx_param)
    sample = Sample(mom_param)

    # model._solve_model(alpha, delta, approx_param)

    # model.visualize_model_sol(alpha, delta, approx_param)
    # model.visualize_mom_sensitivity(visualization_param, sim_param)

    # print(sample.sample_mom)

    # print(model.test_model_sim(alpha, delta, sim_param))

    # est, sim_mom, res, se = get_estimation_results(sample, model, sim_param, opt_param)
    # print(f'{est=}')
    # print(f'{sim_mom=}')
    # print(res)
    # print(se)

    run_noisy_estimation(sample, model, sim_param, opt_param)

    # test1, test2 = get_estimation_results(sample, model, sim_param)
    # print(f'{test1=}')
    # print(f'{test2=}')
    
    print("Done!")
