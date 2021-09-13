# This file contains the definitions to generate a random trajectory in a given
# state space defined by a dynamical system.

import numpy as np
import random as rand

from Trajectory import Trajectory

def gen_rand_traj(dim, no_modes):
    """
        Generate a random trajectory with a Gaussian distribution.

        Parameters
        ----------
        dim: positive int
            dimension of the trajectory
        no_mondes: positive int
            number of modes making up the trajectory
        
        Returns
        -------
        traj: Trajectory
    """
    # initialise empty trajectory
    traj = Trajectory(np.zeros([int(no_modes), dim], dtype = complex))

    # loop over elements of trajectory and assign values
    for i in range(1, traj.shape[0]):
        for j in range(dim):
            traj[i, j] = 0.1*(rand.gauss(0, 1) + 1j*rand.gauss(0, 1))

    return traj
