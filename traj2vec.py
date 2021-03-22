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
    dofs = (2*traj.shape[1]*(traj.shape[0] - 2)) + 1
    vector = np.zeros(dofs)
    a = 0
    for i in range(traj.shape[0] - 2):
        for j in range(traj.shape[1]):
            for k in range(2):
                if k == 0:
                    vector[i*traj.shape[1] + j + a] = np.real(traj[i + 1, j])
                else:
                    a += 1
                    vector[i*traj.shape[1] + j + a] = np.imag(traj[i + 1, j])
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
    dofs = np.shape(opt_vector)[0]
    if (dofs - 1)/dim % 1 != 0:
        raise ValueError("Vector length not compatible with dimensions!")
    traj_list = [None]*(int((dofs - 1)/(2*dim)) + 2)
    mode_vector = np.zeros(dim, dtype = complex)
    traj_list[0] = np.zeros(dim, dtype = complex)
    traj_list[-1] = np.zeros(dim, dtype = complex)
    a = 0
    for i in range(int((dofs - 1)/2)):
        mode_vector[i - dim*int(i/dim)] = opt_vector[a] + 1j*opt_vector[a + 1]
        if (i + 1)/dim % 1 == 0:
            traj_list[int(i/dim) + 1] = np.copy(mode_vector)
        a += 2
    return Trajectory(traj_list), opt_vector[-1]
