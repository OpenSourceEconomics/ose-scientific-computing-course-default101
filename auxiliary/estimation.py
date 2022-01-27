from os import times_result
import numpy as np
from scipy.optimize import dual_annealing
from auxiliary.helpers_calcmoments import *
from auxiliary.Model import Model
from auxiliary.tauchen import approx_markov
from numpy.linalg import inv

def _objective_func(sample_moments, weight_matrix, terry, sim_param):
    """
    """
    # dual_annealing needs the function input as a 1D array
    # x[0]: alpha, x[1]: delta
    f = lambda x: ((terry._get_sim_moments(x[0],x[1],sim_param)- sample_moments).T @weight_matrix@(terry._get_sim_moments(x[0],x[1],sim_param)- sample_moments))
        
    return f

def _run_optimization(obj_func, parameter_space):
    """
    """
    ret = dual_annealing(obj_func, bounds=parameter_space, maxiter=100)

    return ret

def _get_weight_matrix(sample, sample_mom, no_moments):
    """
    Choice of weight matrix suggested on slide 19/76 in Part 2 Handout.
    No idea why.
    """
    covar_emp_mom = _get_covar_emp_moments(sample, sample_mom, no_moments)

    try:
        out = inv(covar_emp_mom)

    except np.linalg.LinAlgError as e:
        if 'Singular matrix' in str(e):
            print("Empirical covariance matrix not invertible. Use identity as weight matrix instead.")
            out = np.identity(no_moments)
        else:
            raise
        
    return out

def _get_covar_emp_moments(sample, sample_mom, no_moments):
    """
    REDO
    Calculates the weight matrix from influence functions with clustering at the firm level.

    Args
    ----
    sample (pandas.DataFrame):      DataFrame with 4 columns: firm, year, profitability, inv_rate
    no_moments (int):               number of moments

    Returns
    -------
    weight matrix (np.ndarray):     weight matrix of shape (no_moments x no_moments)

    """
    sample_nobs = sample.shape[0]  # number of year-firm obs
    
    yearspfirm = get_nyears_per_firm(sample)  # vector with years per firm
    no_firms = len(yearspfirm)
    pos_firms = np.cumsum(yearspfirm)  # to know positions of obs per firm 
    pos_firms = np.insert(pos_firms, 0, 0) # to ensure subsequent loop starts at beginning

    infl_mat = _get_influence_func(sample, sample_mom)

    out = np.zeros((no_moments, no_moments))  # initialization

    for i in range(0,no_firms):
        mat_clust = np.ones((yearspfirm[i], yearspfirm[i]))
        out = out + infl_mat[pos_firms[i]:pos_firms[i+1],:].transpose() @ mat_clust @ infl_mat[pos_firms[i]:pos_firms[i+1],:]
        # print(f'{out=}')
    
    print(f'{out=}')
    out = out / (np.square(sample_nobs))

    return out

def _get_influence_func(sample, sample_mom):
    """
    Calculate the stacked influence function for mean of profitability, mean of inv_rate and variance of profitability.
    
    Args
    ----------
    sample (pandas.DataFrame):  DataFrame with 4 columns: firm, year, profitability, inv_rate
    sample_mom (numpy.ndarray): array with sample moments

    Returns
    -------
    infl_fct (numpy.ndarray):    array (sample_obs x 3) where columns contain the influence functions for each of the moments
    """
    sample_nobs = sample.shape[0] # number of year-firm obs

    infl_fct = sample[["profitability", "inv_rate", "profitability"]].to_numpy(dtype=float, copy=True)

    # Influence function for means: x - E[X]
    infl_fct[:0] = infl_fct[:0] - sample_mom[0]
    infl_fct[:1] = infl_fct[:1] - sample_mom[1]

    # Influence function for variance: (x-E[X])^2 - Var[X]
    infl_fct[:2] = np.square((infl_fct[:2] - sample_mom[0])) - sample_mom[2]

    # # TODO in MATLAB code: why use demeaned data?

    return infl_fct

def _get_sample_moments(data):
    """
    """
    moments = calc_moments(data, DataFrame=True)

    return moments

def _get_sample():
    """
    """
    data = import_data("data/RealData.csv", ["%firm_id","year","profitability","inv_rate"])
    
    return data

def _optimization(model, sim_param):
    """
    # OPEN: take out parameters of dual_annealing!
    """
    # Set-up
    data = _get_sample()
    sample_moments = _get_sample_moments(data)
    weight_mat = _get_weight_matrix(data, sample_moments, no_moments=3)
    f = _objective_func(sample_moments, weight_mat, model, sim_param)
    bounds=sim_param["bounds_optimizer"]

    out = _run_optimization(f, bounds)
    # out is a dictionary with keys: ['success', 'status', 'x', 'fun', 'nit', 'nfev', 'njev', 'nhev', 'message']
    
    return out

def _get_sim_moments_est(model, alpha, delta, sim_param):
    """
    """
    sim_moments = model._get_sim_moments(alpha, delta, sim_param)

    return sim_moments

def _get_Jacobian():
    pass

def _get_covar_est():
    pass

def get_estimation_results(model, sim_param):
    """
    """
    result = _optimization(model, sim_param)

    final_est = result['x']
    alpha_est = final_est[0]
    delta_est = final_est[1]

    sim_moments_est = _get_sim_moments_est(model, alpha_est, delta_est, sim_param)

    return final_est, sim_moments_est
    # # return parameter_est, se_est, covar_est, sim_moments_est

    # data = _get_sample()
    # sample_moments = _get_sample_moments(data)
    # test = _get_weight_matrix(data, sample_moments, no_moments=3)

    # return test