from os import times_result
import numpy as np
from scipy.optimize import dual_annealing
from auxiliary.helpers_calcmoments import *
from auxiliary.helpers_plotting import plot_noisy_optima
import pybobyqa

def _objective_func(sample, model, sim_param, noise):
    """
    """
    # dual_annealing needs the function input as a 1D array
    # x[0]: alpha, x[1]: delta
    if noise > 0:
        f = lambda x: (1 + noise*np.random.normal(size=(1,))[0])*\
            ((model._get_sim_moments(x[0],x[1],sim_param)- sample.sample_mom).T \
            @sample.weight_mat@(model._get_sim_moments(x[0],x[1],sim_param)- sample.sample_mom))
    else: 
        f = lambda x: ((model._get_sim_moments(x[0],x[1],sim_param)- sample.sample_mom).T \
            @sample.weight_mat@(model._get_sim_moments(x[0],x[1],sim_param)- sample.sample_mom))
        
    return f

def _run_optimization(obj_func, opt_param):
    """
    """

    if opt_param["solver"]=="dual_annealing":
        ret = dual_annealing(func=obj_func,
                            bounds=opt_param["bounds_optimizer"], 
                            maxiter=opt_param["max_iter"]
                            )

    elif opt_param["solver"]=="bobyqa":
        ret = pybobyqa.solve(objfun=obj_func,
                            x0=opt_param["start_value"],
                            bounds=opt_param["bounds_optimizer_bobyqa"], 
                            maxfun=opt_param["max_iter"],
                            objfun_has_noise=opt_param["noisy_function_opt"],
                            seek_global_minimum=True
                            )

    return ret

def _optimization(sample, model, sim_param, opt_param, noise=0):
    """
    """
    f = _objective_func(sample, model, sim_param, noise)
    
    out = _run_optimization(f, opt_param)
    
    return out

def _get_sim_moments_est(model, alpha, delta, sim_param):
    """
    """
    sim_moments = model._get_sim_moments(alpha, delta, sim_param)

    return sim_moments

def _get_Jacobian(model, alpha, delta, sim_param, h=np.finfo(float).eps**(1/3)):
    """
    """

    mom_alpha_upper = _get_sim_moments_est(model, alpha+h, delta, sim_param)
    mom_alpha_lower = _get_sim_moments_est(model, alpha-h, delta, sim_param)
    mom_delta_upper = _get_sim_moments_est(model, alpha, delta+h, sim_param)
    mom_delta_lower = _get_sim_moments_est(model, alpha, delta-h, sim_param)

    grad = np.array([mom_alpha_upper-mom_alpha_lower, 
                    mom_delta_upper-mom_delta_lower])/(2*h)

    return grad 

def _create_covar_est(grad, sample, sim_param):
    """
    """

    nsim = sim_param["number_simulations_per_firm"]

    covm = (1 + 1/nsim) * sample.covar_emp_mom

    aux1 = grad @ sample.weight_mat @ grad.T

    try:
        aux2 = np.linalg.solve(aux1, grad)
        out = aux2 @ sample.weight_mat @ covm @ sample.weight_mat.transpose() @ aux2.transpose()

        return out
    
    except np.linalg.LinAlgError as e:
        if 'Singular matrix' in str(e):
            print("Matrix at intermediate step not invertible.")
        else:
            raise

def _get_se_est(model, sample, alpha, delta, sim_param):
    """
    """

    grad = _get_Jacobian(model, alpha, delta, sim_param)

    covar_est = _create_covar_est(grad, sample, sim_param)

    out = np.sqrt(np.diag(covar_est))

    return out

def _test_overidentification(sample, grad):

    aux1 = np.eye(sample.no_mom)
    pass

def get_estimation_results(sample, model, sim_param, opt_param):
    """
    """
    result = _optimization(sample, model, sim_param, opt_param)


    if opt_param["solver"]=="dual_annealing":
        # result is a dictionary with keys: ['success', 'status', 'x', 'fun', 'nit', 'nfev', 'njev', 'nhev', 'message']
        final_est = result['x']
        alpha_est = final_est[0]
        delta_est = final_est[1]

    elif opt_param["solver"]=="bobyqa":
        # result is an object with : 
        # x: estimate of solution,
        # f: objective vaule at x,
        # gradient: estimate of gradient at x
        final_est = result.x
        alpha_est = final_est[0]
        delta_est = final_est[1]


    sim_moments_est = _get_sim_moments_est(model, alpha_est, delta_est, sim_param)
    standard_errors = _get_se_est(model, sample, alpha_est, delta_est, sim_param)

    # return parameter_est, se_est, covar_est, sim_moments_est
    return final_est, sim_moments_est, result, standard_errors

def run_noisy_estimation(sample, model, sim_param, opt_param):


    noise_range = opt_param["noise_range"]
    minima = np.zeros((len(noise_range),2))

    for i in range(len(noise_range)):

        res = _optimization(sample, model, sim_param, opt_param, noise=noise_range[i])
        if opt_param["solver"]=="dual_annealing":
            minima[i] = res['x']

        elif opt_param["solver"]=="bobyqa":
            minima[i] = res.x

    plot_noisy_optima(noise_range, minima)
