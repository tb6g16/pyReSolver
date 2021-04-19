# This file will contian the definitions for an easy to use plotting wrapper
# for the trajectory object.

import warnings
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from traj_util import list2array
from my_fft import my_irfft
from TrajPlotObject import TrajPlotObject

def plot_single_traj(plot_object, ax = None, proj = None, show = False):
    # pad with zeros to increase resolution
    temp = list2array(plot_object.traj.mode_list)
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
    if plot_object.mean != None:
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
    pass

if __name__ == "__main__":
    from trajectory_definitions import unit_circle as uc
    from trajectory_definitions import ellipse as elps
    from trajectory_definitions import unit_circle_3d as uc3

    from Trajectory import Trajectory

    traj1 = Trajectory(uc.x, modes = 5)
    traj2 = Trajectory(elps.x)
    traj3 = Trajectory(uc3.x)

    # traj1_plot = TrajPlotObject(traj1, disc = 256)
    # traj2_plot = TrajPlotObject(traj2, mean = [1, 1])
    # traj3_plot = TrajPlotObject(traj3)

    plot_traj(traj3, discs = [1024], means = [[np.e, np.pi, 1]], title = "Test Plot", proj = 'xy', aspect = 0.5)
