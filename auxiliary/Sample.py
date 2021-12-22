from os import times_result
import numpy as np
from auxiliary.helpers_calcmoments import *
from auxiliary.Model import Model
from auxiliary.tauchen import approx_markov
from numpy.linalg import inv

class Sample:
    """
    
    """
    def __init__(self, mom_param) -> None:
        
        self.no_mom = mom_param["no_moments"]
        self.no_param = mom_param["no_param"]
        
        self.sample = self._get_sample()
        self.sample_mom = self._get_sample_moments()
        self.covar_emp_mom = self._get_covar_emp_moments()
        self.weight_mat = self._get_weight_matrix()

    def _get_sample(self):
        """
        """
        data = import_data("data/RealData.csv", ["%firm_id","year","profitability","inv_rate"])
    
        return data

    def _get_sample_moments(self):
        """
        """
        moments = calc_moments(self.sample, DataFrame=True)

        return moments

    def _get_influence_func(self):
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
        sample_nobs = self.sample.shape[0] # number of year-firm obs

        infl_fct = self.sample[["profitability", "inv_rate", "profitability"]].to_numpy(dtype=float, copy=True)

        # Influence function for means: x - E[X]
        infl_fct[:0] = infl_fct[:0] - self.sample_mom[0]
        infl_fct[:1] = infl_fct[:1] - self.sample_mom[1]

        # Influence function for variance: (x-E[X])^2 - Var[X]
        infl_fct[:2] = np.square((infl_fct[:2] - self.sample_mom[0])) - self.sample_mom[2]

        # # TODO in MATLAB code: why use demeaned data?

        return infl_fct

    def _get_covar_emp_moments(self):
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
        sample_nobs = self.sample.shape[0]  # number of year-firm obs
        
        yearspfirm = get_nyears_per_firm(self.sample)  # vector with years per firm
        no_firms = len(yearspfirm)
        pos_firms = np.cumsum(yearspfirm)  # to know positions of obs per firm 
        pos_firms = np.insert(pos_firms, 0, 0) # to ensure subsequent loop starts at beginning

        infl_mat = self._get_influence_func()

        out = np.zeros((self.no_mom, self.no_mom))  # initialization

        for i in range(0,no_firms):
            mat_clust = np.ones((yearspfirm[i], yearspfirm[i]))
            out = out + infl_mat[pos_firms[i]:pos_firms[i+1],:].transpose() @ mat_clust @ infl_mat[pos_firms[i]:pos_firms[i+1],:]
            # print(f'{out=}')
        
        out = out / (np.square(sample_nobs))

        return out
    
    def _get_weight_matrix(self):
        """
        Choice of weight matrix suggested on slide 19/76 in Part 2 Handout.
        No idea why.
        """
        try:
            out = inv(self.covar_emp_mom)

        except np.linalg.LinAlgError as e:
            if 'Singular matrix' in str(e):
                print("Empirical covariance matrix not invertible. Use identity as weight matrix instead.")
                out = np.identity(self.no_mom)
            else:
                raise
            
        return out