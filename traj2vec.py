# This file contains the functions that convert a trajectory frequency pair to
# a vector for optimisation purposes, and also the inverse conversion.

import numpy as np
from Trajectory import Trajectory

def traj2vec(traj, freq):
    """
        Return the vectorised form of the given trajectory frequency pair.

        Take all the elements (modes) making up the given trajectory and the
        frequency float and store it all as a 1D array.

        Parameters
        ----------
        traj : Trajectory
        freq : float

        Returns
        -------
        vector : ndarray
            1D array containing data of float type.
    """
    # defining the degrees of freedom of the system
    dofs = (2*traj.shape[1]*(traj.shape[0] - 2)) + 1

    # initialise the vector with the degrees of freedome of the system.
    vector = np.zeros(dofs)

    # loop over all trajectory values and assign to the vector
    a = 0
    for i in range(traj.shape[0] - 2):
        for j in range(traj.shape[1]):
            for k in range(2):
                if k == 0:
                    vector[i*traj.shape[1] + j + a] = np.real(traj[i + 1, j])
                else:
                    a += 1
                    vector[i*traj.shape[1] + j + a] = np.imag(traj[i + 1, j])

    # assign the frequency value do the vector
    vector[-1] = freq

    return vector

def vec2traj(opt_vector, dim):
    """
        Return the equivalent trajectory frequency pair for a given vector.

        Convert a given vector into a state trajectory and frequency pair.

        Parameters
        ----------
        opt_vector : ndarray
            1D array containing data of float type.
        dim : int
            Positive integer for the space the trajectory resides in.
        
        Returns
        -------
        Trajectory
            The trajectory from the given vector.
        float
            The frequency from the given vector.
    """
    # define the degrees of freedom of the system
    dofs = np.shape(opt_vector)[0]

    # initialise lists and arrays
    traj_array = np.zeros([int((dofs - 1)/(2*dim)) + 2, dim], dtype = complex)
    mode_vector = np.zeros(dim, dtype = complex)

    # loop over degrees of freedom and assign the elements of the trajectory list
    a = 0
    for i in range(int((dofs - 1)/2)):
        mode_vector[i - dim*int(i/dim)] = opt_vector[a] + 1j*opt_vector[a + 1]
        if (i + 1)/dim % 1 == 0:
            traj_array[int(i/dim) + 1] = np.copy(mode_vector)
        a += 2

    return Trajectory(traj_array), opt_vector[-1]
