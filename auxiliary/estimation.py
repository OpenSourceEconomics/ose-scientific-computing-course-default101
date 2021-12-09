import numpy as np
from scipy.optimize import dual_annealing
from auxiliary.helpers_calcmoments import *

def objective_function(sample_moments, weight_matrix, terry, alpha, delta):
    """
    
    """
    if alpha >= 1 or alpha <= 0:
        out = np.inf
        # TODO sensible choice?
    elif delta >= 0.3 or delta <= 0:
        out = np.inf
    else:
        sim_moments = terry.get_sim_mom(alpha,delta)
        deviations = sim_moments - sample_moments
        out = deviations.transpose() @ weight_matrix @ deviations 
        
        return out

def optimization(optimizer=dual_annealing, obj_func, parameter_space):
    """
    """
    ret = dual_annealing(obj_func, bounds=parameter_space)

    return

def weight_matrix(sample, no_moments):
    """
    Calculates the weight matrix from influence functions with clustering at the firm level.

    Args
    ----
    sample (pandas.DataFrame):      DataFrame with 4 columns: firm, year, profitability, inv_rate
    no_moments (int):               number of moments

    Returns
    -------
    weight matrix (np.ndarray):     weight matrix of shape (no_moments x no_moments)

    """
    sample_nobs = sample.shape()[0]  # number of year-firm obs
    
    yearspfirm = get_nyears_per_firm(sample)  # vector with years per firm
    no_firms = len(yearspfirm)
    pos_firms = np.cumsum(yearspfirm)  # to know positions of obs per firm 
    
    infl_mat = influence_fct(sample)

    out = np.zeros((no_moments, no_moments))  # initialization

    for i in range(0,no_firms):
        mat_clust = np.ones((yearspfirm[i], yearspfirm[i]))
        out = out + infl_mat[pos_firms[i]:pos_firms[i+1],:].transpose() @ mat_clust @ infl_mat[pos_firms[i]:pos_firms[i+1],:]
    
    out = out / (sample_nobs).square()

    return out

def influence_fct(sample):
    """
    Calculate the stacked influence function for mean of profitability, mean of inv_rate and variance of profitability.
    
    Args
    ----------
    sample (pandas.DataFrame):  DataFrame with 4 columns: firm, year, profitability, inv_rate

    Returns
    -------
    infl_fct (numpy.ndarray):    array (sample_obs x 3) where columns contain the influence functions for each of the moments
    """
    sample_nobs = sample.shape()[0] # number of year-firm obs
    sample_mom = calc_moments(sample)

    infl_fct = np.array([np.copy(sample.loc[:,2].values), np.copy(sample.loc[:,3].values), np.copy(sample.loc[:,2].values)])

    # Influence function for means: x - E[X]
    infl_fct[:0] = infl_fct[:0] - sample_mom[0]
    infl_fct[:1] = infl_fct[:1] - sample_mom[1]

    # Influence function for variance: (x-E[X])^2 - Var[X]
    infl_fct[:2] = (infl_fct[:2] - sample_mom[0]).square() - sample_mom[2]

    # TODO in MATLAB code: why use demeaned data?

    return infl_fct