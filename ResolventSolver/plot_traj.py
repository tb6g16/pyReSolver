# This file will contian the definitions for an easy to use plotting wrapper
# for the trajectory object.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ResolventSolver.my_fft import my_irfft
from ResolventSolver.TrajPlotObject import TrajPlotObject

def plot_single_traj(plot_object, ax = None, proj = None, show = False):
    """
        Plot a single trajectory on a provided matplotlib axis.

        Parameters
        ----------
        plot_object : TrajPlotObject
        ax : matplotlib.axes
            The axis on which to plot this trajectory.
        proj : {'xy', 'yx', 'xz', 'zx', 'yz', 'zy', None}, default=None
            The projection of the trajectory on the axis, not necessary if the
            plot is 3D.
        show : {True, False}, default=False
            Boolean for whether the resulting plot should be shown.
    """
    # pad with zeros to increase resolution
    temp = plot_object.traj.modes
    if plot_object.disc != None:
        tot_modes = int(plot_object.disc/2) + 1
        pad_len = tot_modes - plot_object.traj.shape[0]
        if pad_len >= 0:
            modes_padded = np.pad(temp, ((0, pad_len), (0, 0)), 'constant')
        else:
            print("WARNING: Cannot reduce resolution, setting to default!")
            modes_padded = temp
    else:
        modes_padded = temp

    # adding mean
    if type(plot_object.mean) == np.ndarray:
        modes_padded[0] = plot_object.mean
    elif plot_object.mean != None:
        modes_padded[0] = plot_object.mean

    # convert to time domain
    traj_time = my_irfft(modes_padded)

    # plot curve in time domain
    if plot_object.traj.shape[1] == 3:
        if proj == None:
            plt.plot(np.append(traj_time[:, 0], traj_time[0, 0]), \
                     np.append(traj_time[:, 1], traj_time[0, 1]), \
                     np.append(traj_time[:, 2], traj_time[0, 2]))

        elif proj == 'xy' or proj == 'yx':
            plt.plot(np.append(traj_time[:, 0], traj_time[0, 0]), \
                     np.append(traj_time[:, 1], traj_time[0, 1]))

        elif proj == 'xz' or proj == 'zx':
            plt.plot(np.append(traj_time[:, 0], traj_time[0, 0]), \
                     np.append(traj_time[:, 2], traj_time[0, 2]))

        elif proj == 'yz' or proj == 'zy':
            plt.plot(np.append(traj_time[:, 1], traj_time[0, 1]), \
                     np.append(traj_time[:, 2], traj_time[0, 2]))

    elif plot_object.traj.shape[1] == 2:
        plt.plot(np.append(traj_time[:, 0], traj_time[0, 0]), \
                 np.append(traj_time[:, 1], traj_time[0, 1]))

    else:
        raise ValueError("Can't plot dimensions higher then 3!")
    
    # show plot if given true as argument
    if show == True:
        plt.show()

def plot_traj(*args, **kwargs):
    """
        Plot and show one or more trajectories on a single axis.

        Parameters
        ----------
        *args : Trajectory
            An arbitrary number of trajectories to be plotted.
        title : str, defult=None
            Title of the plot.
        aspect : positive float, default=None
            Aspect ratio of the plot.
        proj : {'xy', 'yx', 'xz', 'zx', 'yz', 'zy', None}, default=None
            Projection of the trajectories in the plot.
        discs : list of int, default=list of None
            The discretisation resolution for all the trajectories.
        means : list of ndarray, default=list of None
            The mean values for all the trajectories.
    """
    # unpack keyword arguments
    title = kwargs.get('title', None)
    aspect = kwargs.get('aspect', None)
    proj = kwargs.get('proj', None)
    discs = kwargs.get('discs', [None]*len(args))
    means = kwargs.get('means', [None]*len(args))

    # initialise figure and axis
    fig = plt.figure()
    if args[0].shape[1] == 3 and proj == None:
        ax = fig.gca(projection = '3d')
    else:
        ax = fig.gca()

    # set title
    fig.suptitle(title)

    # set aspect ratio
    if aspect != None:
        ax.set_aspect(aspect)

    # loop through all given trajectories and plot
    for index, arg in enumerate(args):
        plot_object = TrajPlotObject(arg, disc = discs[index], mean = means[index])
        plot_single_traj(plot_object, ax = ax, proj = proj)

    # show the plot (OR SAVE IT????)
    plt.show()

def plot_along_s(*args, **kwargs):
    """
        Plot the values of all the components of a trajectory along their own
        length (from 0 to 2*pi).

        Parameters
        ----------
        *args : Trajectory
            An arbitrary number of trajectories to be plotted.
        labels : list of str, default=None
            List of labels for each component of the trajectories to construct
            a legend.
        ylim : list of two floats
            The limits of the y-axis of the resulting plot.
    """
    # unpack keyword arguments
    labels = kwargs.get('labels', None)
    ylim = kwargs.get('ylim', None)

    # plt.figure(fig_no)
    plt.figure()
    ax = plt.gca()

    # loop over given trajectories to plot
    for arg in args:
        traj_time = my_irfft(arg.modes)
        s = np.linspace(0, 2*np.pi, np.shape(traj_time)[0] + 1)
        for i in range(np.shape(traj_time)[1]):
            if labels != None:
                ax.plot(s, np.append(traj_time[:, i], traj_time[0, i]), label = labels[i])
            else:
                ax.plot(s, np.append(traj_time[:, i], traj_time[0, i]))

    # add labels
    if labels != None:
        ax.legend()

    # add y-axis limits
    if ylim != None:
        plt.ylim(ylim)

    # add grid and show plot
    ax.grid()
    plt.show()
