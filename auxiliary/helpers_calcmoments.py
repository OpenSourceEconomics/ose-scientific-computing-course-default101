import pandas as pd
import sys, os
import numpy as np

def import_data(relative_path, to_keep):
    """ 
    Import csv and keep specified columns.
    
    Parameters
    ----------
    relative_path (str):        relative path to csv
    to_keep (list of str):      column names to keep

    Returns
    -------
    dataframe

    """
    cwd = os.getcwd()
    sourcepath = os.path.join(cwd, relative_path)

    data = pd.read_csv(sourcepath, sep=',', header=0, encoding = "ISO-8859-1")
    data.drop(data.columns.difference(to_keep), 1, inplace=True)
    
    return data

def calc_moments(data):
    """
    Calculate the mean of profitability and inv_rate.
    """
    moments = np.zeros(2)

    moments[0] = data["profitability"].mean()
    moments[1] = data["inv_rate"].mean()

    return moments

def add_deviations_from_sample_mean(data):
    """
    Adds columns in which [profitability, inv_rate] are demeaned at the firm-level and then the full-sample mean is added again.

    Parameters
    ----------
    data (dataframe):           dataframe with firm, year, profitability, inv_rate

    Returns
    -------
    data_merges (dataframe):    dataframe with additional columns profitability_adj, inv_adj
    """
    mean_by_firm = data.groupby(by=["%firm_id"]).mean()

    data_merged = data.merge(mean_by_firm, how="left", on="%firm_id", suffixes=("", "_firmmean"))

    variables = ["profitability", "inv_rate"]
    for var in variables:
        data_merged[var + "_adj"] = data_merged[var] - data_merged[var + "_firmmean"] + data_merged[var].mean()

    data_merged.drop(labels=["year_firmmean", "profitability_firmmean", "inv_rate_firmmean"], axis=1, inplace=True)

    return data_merged