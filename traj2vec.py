# This file contains the functions that convert a trajectory frequency pair to
# a vector for optimisation purposes, and also the inverse conversion.

import numpy as np
from Trajectory import Trajectory

def traj2vec(traj, freq):
    """
        This function takes in a trajectory and frequency and returns a vector
        that will be used for the optimisation.

        Parameters
        ----------
        traj: Trajectory
            the trajectory that makes up most of the optimisation vector
        freq: float
            the fundamental frequency of the associated trajectory, the last
            element of the optimisation vector
        
        Returns
        -------
        opt_vector: numpy array
            the optimisation vector defined by the trajectory frequency pair
    """
    dofs = (traj.shape[0]*traj.shape[1]) + 1
    vector = np.zeros([dofs])
    for j in range(traj.shape[1]):
        for i in range(traj.shape[0]):
            vector[i + j*traj.shape[0]] = traj[i, j]
    vector[-1] = freq
    return vector

def vec2traj(opt_vector, dim):
    """
        This function converts an optimisation variable back into its
        corresponding trajectory frequency pair.

        Parameters
        ----------
        opt_vector: numpy array
            the optimisation vector
        dim: positive integer
            the dimension of the state-space through which the trajectory
            travels
        
        Returns
        -------
        traj: Trajectory
            the corresponding trajectory
        freq: float
            the corresponding frequency
    """
    vec_size = np.shape(opt_vector)[0]
    if (vec_size - 1)/dim % 1 != 0:
        raise ValueError("Vector length not compatible with dim!")
    traj_array = np.zeros([dim, int((vec_size - 1)/dim)])
    for i in range(vec_size - 1):
        traj_array[i - dim*int(i/dim), int(i/dim)] = opt_vector[i]
    return Trajectory(traj_array), opt_vector[-1]
