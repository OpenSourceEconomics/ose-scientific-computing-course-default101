import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D # for 3d plots
from matplotlib import cm # for 3d plots
import seaborn as sns
import numpy as np
import pandas as pd

def line_plot(x, df_y, nz, xlabel, ylabel, title):
    """
    Auxiliary function for visualizing the model solution.
    """
    fig, ax = plt.subplots()
    ax.plot(x, df_y[1,:], color='red', label='low productivity shock')
    ax.plot(x, df_y[int(np.floor(nz/2)),:], color='green', label='medium productivity shock')
    ax.plot(x, df_y[nz-1,:], color='blue', label='high productivity shock')
    ax.set_xlabel('{}'.format(xlabel))
    ax.set_ylabel('{}'.format(ylabel))
    ax.set_title(title)
    ax.legend()
    plt.show()


def plot_mom_sensitivity(grids, moments, xlabel, \
                    labels=["mean profitability", "mean inv rate", "var profitability"],\
                    colors=["red", "green", "blue"],\
                    title="Moment sensitivities to alpha and delta", nmoments=3):
    """
    Auxiliary function for visualizing the sensitivity of the moments to the model parameters.
    """
    fig, axs = plt.subplots(nrows=nmoments,ncols=2, sharex='col', figsize=(10,6))

    fig.suptitle(title)

    for i in range(nmoments):
        axs[i,0].plot(grids[0], moments[0][:,i], color=colors[i], label=labels[i])
        axs[i,1].plot(grids[1], moments[1][:,i], color=colors[i])
        axs[i,0].legend()

    axs[i,0].set_xlabel('{}'.format(xlabel['alpha']))
    axs[i,1].set_xlabel('{}'.format(xlabel['delta']))

    plt.show()


def threedim_plot(x_grid,y_grid,z,xlabel,ylabel,zlabel,title):
    """
    Auxiliary function for visualizing the model solution.
    """
    x, y = np.meshgrid(x_grid, y_grid)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z.T,
                    rstride=2, cstride=2,
                    cmap=cm.jet,
                    alpha=0.7,
                    linewidth=0.25)
    ax.set_xlim3d(np.max(x_grid), np.min(x_grid)) # otherwise the x axis is displayed "flipped" (relative to the Matlab plot)
    ax.set_ylabel('{}'.format(ylabel))
    ax.set_xlabel('{}'.format(xlabel))
    ax.set_zlabel('{}'.format(zlabel))
    ax.set_title(title)
    plt.show()


def visualize_model_fit(sample, model, alpha, delta, sim_param):
    """
    Boxplots of the targeted moments in the real data and in the simulated data.
    """
    size = {}
    final_sim = model._simulate_model(alpha, delta, sim_param)
    final_sim_data = pd.DataFrame(data=final_sim[:,2:4],columns=["profitability", "inv_rate"])
    size['model'] = final_sim_data.shape[0]

    data_orig = sample._get_sample()[ ['profitability', 'inv_rate']].copy()
    size['data'] = data_orig.shape[0]

    # Add type for using seaborns boxplot method "hue"
    final_sim_data['type'] = f'model, N: {size["model"]}'
    data_orig['type'] = f'data, N: {size["data"]}'

    # data needs to be in long form for grouped seaborn boxplots
    final_sim_data = pd.melt(final_sim_data, id_vars=['type'], value_vars=['profitability', 'inv_rate'])
    data_orig = pd.melt(data_orig, id_vars=['type'], value_vars=['profitability', 'inv_rate'])
    to_plot = data_orig.append(final_sim_data, ignore_index=True)
    to_plot.replace({'inv_rate':'investment rate'}, inplace=True)

    # Plotting
    fig,ax = plt.subplots(figsize=(5, 6))
    ax.grid(axis='y')
    ax.legend([f'data, N: {size["data"]}', f'model, N: {size["model"]}'])
    ax = sns.boxplot(x="variable", y="value", hue="type", width=0.3, data=to_plot, whis=0.4, showfliers=False, showmeans=True)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Comparing the distribution of targeted moments")

    plt.show()