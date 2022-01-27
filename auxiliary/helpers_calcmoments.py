import pandas as pd
import sys, os
import numpy as np

def import_data(relative_path, to_keep):
    """ 
    Import csv, keep specified columns, sort based on firm_id and check for missings.
    (Sorting to ensure that future groupby commands do not alter the ordering of the firms.)
    
    Args
    -----
    relative_path (str):        relative path to csv
    to_keep (list of str):      column names to keep

    Returns
    -------
    data (pandas.DataFrame)

    """
    cwd = os.getcwd()
    sourcepath = os.path.join(cwd, relative_path)

    data = pd.read_csv(sourcepath, sep=',', header=0, encoding = "ISO-8859-1")
    data.drop(labels=data.columns.difference(to_keep), axis=1, inplace=True)

    data.sort_values(by=["%firm_id"], inplace=True)

    if np.isnan(data.values).any():
        raise ValueError("Sample contains missing values.")
    
    else:
        return data

def export_data(data, relative_path="data/SimData", datatype=".csv"):
    """
    Export simulated data to file of given type.
    """
    cwd = os.getcwd()
    sourcepath = os.path.join(cwd, relative_path, datatype)

    data.to_csv(sourcepath)

def label_data(data, labels=["%firm_id", "year", "profitability", "inv_rate", "equity_iss", "cashflow", "capital", "prod_shock"]):
    """
    Label raw data from simulation.
    """

    labeled_data = pd.DataFrame(data=data, columns=labels)

    labeled_data = labeled_data.astype({"%firm_id":"int32", "year":"int32"})

    return labeled_data


def calc_moments(data, DataFrame=False):
    """
    Calculate (full-sample) mean of profitability, mean of inv_rate, and variance of profitability.

    Args
    ----
    data (pandas.DataFrame or np.ndarray):    sample, for np.ndarrays the columns must be ordered as follows: firm, year, profitability, inv_rate

    Returns
    -------
    moments (numpy.ndarray):                  sample moments: mean profitability, mean inv_rate, variance profitability    
    """
    moments = np.zeros(3)
    
    if DataFrame == True: # If input is a DataFrame
        moments[0] = data["profitability"].mean()
        moments[1] = data["inv_rate"].mean()
        moments[2] = data["profitability_adj"].var()

    else:
        moments[0] = data[:,2].mean()
        moments[1] = data[:,3].mean()
        moments[2] = data[:,2].var()

    return moments

def add_deviations_from_sample_mean(data):
    """
    Adds columns in which [profitability, inv_rate] are demeaned at the firm-level and then the full-sample mean is added again.

    Args
    ----
    data (pandas.DataFrame):           dataframe with 4 columns: firm, year, profitability, inv_rate

    Returns
    -------
    data_merges (pandas.DataFrame):    dataframe with additional columns profitability_adj, inv_adj
    """
    mean_by_firm = data.groupby(by=["%firm_id"]).mean()

    data_merged = data.merge(mean_by_firm, how="left", on="%firm_id", suffixes=("", "_firmmean"))

    variables = ["profitability", "inv_rate"]
    for var in variables:
        data_merged[var + "_adj"] = data_merged[var] - data_merged[var + "_firmmean"] # + data_merged[var].mean()

    data_merged.drop(list(data_merged.filter(regex = '_firmmean')), axis=1, inplace=True)

    return data_merged

def get_nyears_per_firm(data):
    """
    Stores the number of years each firm appears in the sample.

    Args
    ----
    data (pandas.DataFrame):        dataframe with 4 columns: firm, year, profitability, inv_rate

    Returns
    -------
    nyears (numpy.ndarray):         vector with the number of years per firms, firms ordered in the same way as in data
    """
    # Get collapsed dataframe with counts per firm
    years_per_firm = data.groupby(by=["%firm_id"]).count()

    out = np.array(np.copy(years_per_firm.iloc[:,0].values))
    
    return out

