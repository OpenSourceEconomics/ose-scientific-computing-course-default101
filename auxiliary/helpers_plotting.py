import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D # for 3d plots
from matplotlib import cm # for 3d plots
import numpy as np

def line_plot(x, df_y, nz, xlabel, ylabel, title):
    
    fig, ax = plt.subplots()
    ax.plot(x, df_y[1,:], color='red', label='low productivity shock')
    ax.plot(x, df_y[int(np.floor(nz/2)),:], color='green', label='medium productivity shock')
    ax.plot(x, df_y[nz-1,:], color='blue', label='high productivity shock')
    ax.set_xlabel('{}'.format(xlabel))
    ax.set_ylabel('{}'.format(ylabel))
    ax.set_title(title)
    ax.legend()
    plt.show()

def threedim_plot(x_grid,y_grid,z,xlabel,ylabel,zlabel,title):
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
