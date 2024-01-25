# This file contains the functions that convert a trajectory frequency pair to
# a vector for optimisation purposes, and also the inverse conversion.

import numpy as np

from ResolventSolver.Trajectory import Trajectory

def init_comp_vec(traj):
    return np.zeros([2*traj.shape[1]*(traj.shape[0] - 1)])

def traj2vec(traj, vec):
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
    np.concatenate((np.reshape(traj[1:].real, -1), np.reshape(traj[1:].imag, -1)), out = vec)

def vec2traj(traj, vec):
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
    # split vector into real and imaginary parts
    real_comps = vec[:int(np.shape(vec)[0]/2)]
    imag_comps = vec[int(np.shape(vec)[0]/2):]

    # convert vectors into arrays
    opt_modes = int(np.shape(vec)[0]/(2*traj.shape[1]))
    real_comps = np.reshape(real_comps, (opt_modes, traj.shape[1]))
    imag_comps = np.reshape(imag_comps, (opt_modes, traj.shape[1]))

    # combine and pad zero and end modes
    np.copyto(traj[1:], real_comps + 1j*imag_comps)

    return traj
