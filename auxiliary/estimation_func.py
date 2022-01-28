from os import times_result
import numpy as np
from scipy.optimize import dual_annealing
from auxiliary.helpers_calcmoments import *
from auxiliary.Model import Model
from numpy.linalg import inv
import pybobyqa

def _objective_func(sample, model, sim_param):
    """
    """
    # dual_annealing needs the function input as a 1D array
    # x[0]: alpha, x[1]: delta
    f = lambda x: ((model._get_sim_moments(x[0],x[1],sim_param)- sample.sample_mom).T @sample.weight_mat@(model._get_sim_moments(x[0],x[1],sim_param)- sample.sample_mom))
        
    return f

def _run_optimization(obj_func, parameter_space, solver="dual_annealing"):
    """
    """

    if solver=="dual_annealing":
        ret = dual_annealing(obj_func, bounds=parameter_space, maxiter=10)
    elif solver=="":
        pass


    return ret

def _optimization(sample, model, sim_param):
    """
    # OPEN: take out parameters of dual_annealing!
    """
    f = _objective_func(sample, model, sim_param)
    bounds = sim_param["bounds_optimizer"]

    out = _run_optimization(f, bounds)
    # out is a dictionary with keys: ['success', 'status', 'x', 'fun', 'nit', 'nfev', 'njev', 'nhev', 'message']
    
    return out

def _get_sim_moments_est(model, alpha, delta, sim_param):
    """
    """
    sim_moments = model._get_sim_moments(alpha, delta, sim_param)

    return sim_moments

def _get_Jacobian(sample, alpha, delta):
    """
    """
    # out = np.zeros([sample.no_mom, sample.no_param])
    pass

def _create_covar_est(sample, alpha, delta, nsim):
    """
    """
    grad = _get_Jacobian(sample, alpha, delta)
    covm = (1 + 1/nsim) * sample.covar_emp_mom

    aux1 = grad.transpose() @ sample.weight_mat @ grad

    try:
        # Nicht invertierern, sondern GLS lösen.
        aux2 = inv(aux1) @ grad.transpose()
        out = aux2 @ sample.weight_mat @ covm @ sample.weight_mat.transpose() @ aux2.transpose()

        return out
    
    except np.linalg.LinAlgError as e:
        if 'Singular matrix' in str(e):
            print("Matrix at intermediate step not invertible.")
        else:
            raise

def _create_se_est(covar_est):
    """
    """
    out = np.sqrt(np.diag(covar_est))

    return out


def get_estimation_results(sample, model, sim_param):
    """
    """
    result = _optimization(sample, model, sim_param)

    final_est = result['x']
    alpha_est = final_est[0]
    delta_est = final_est[1]

    sim_moments_est = _get_sim_moments_est(model, alpha_est, delta_est, sim_param)

    # return parameter_est, se_est, covar_est, sim_moments_est
    return final_est, sim_moments_est, result