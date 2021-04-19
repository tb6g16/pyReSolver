# This file contains the class definition to allow simple implementation of
# trajectory plotting in the plot_traj.py file.

import numpy as np

class TrajPlotObject():
    """
        This class definition has all the information required to plot a single
        trajectory such that any number can easily be looped over in a plotting
        function.
    """
    
    __slots__ = ['traj', 'disc', 'mean']

    def __init__(self, traj, disc = None, mean = None):
        self.traj = traj
        self.disc = disc
        self.mean = mean
