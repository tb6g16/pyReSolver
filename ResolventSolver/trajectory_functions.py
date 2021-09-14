# This file contains the function definitions for some trajectory-trajectory
# and trajectory-system interactions.

import numpy as np

from ResolventSolver.Trajectory import Trajectory
from ResolventSolver.my_fft import my_rfft, my_irfft
from ResolventSolver.conv import conv_vec_vec_fast, conv_mat_vec_fast, conv_mat_mat_fast, conv_array

def transpose(traj):
    """
        Return the transpose of a trajectory.

        Parameters
        ----------
        traj : Trajectory

        Returns
        -------
        Trajectory
    """
    return Trajectory(np.transpose(traj.modes, axes = [0, *range(1, traj.modes.ndim)[::-1][0:]]))

def conj(traj):
    """
        Return the complex conjugate of a trajectory.

        Parameters
        ----------
        traj : Trajectory

        Returns
        -------
        Trajectory
    """
    return Trajectory(np.conj(traj.modes))

def traj_rfft(array):
    """
        Return the real FFT of an array as a trajectory instance.

        Parameters
        ----------
        array : ndarray
            N-D array containing data of float type.
        
        Returns
        -------
        Trajectory
    """
    return Trajectory(my_rfft(array))

def traj_irfft(traj):
    """
        Return the inverse real FFT of a trajectory as an array.

        Parameters
        ----------
        traj : Trajectory

        Returns
        -------
        ndarray
    """
    return my_irfft(traj.modes)

def traj_conv(traj1, traj2, method = 'fft'):
    """
        Return the convolution of two trajectories.

        Perform a discrete convolution of two trajectory instances using either
        a direct sum or indirect FFT approach.

        Parameters
        ----------
        traj1, traj2 : Trajectory
            Trajectories of the same size to be convolved
        method : {'fft', 'sum'}, default='fft'

        Returns
        -------
        Trajectory
    """
    return Trajectory(conv_array(traj1.modes, traj2.modes, method = method))

def traj_conv_vec_vec(traj1, traj2):
    """
        Return the convolution of two vector trajectories.

        Perform a discrete convolution of two trajectories with vector states
        using in-built numpy functions.

        Parameters
        ----------
        traj1, traj2 : Trajectory
            Trajectories of the same size to be convolved

        Returns
        -------
        Trajectory
    """
    return Trajectory(conv_vec_vec_fast(traj1.modes, traj2.modes))

def traj_conv_mat_vec(traj1, traj2):
    """
        Return the convolution of a matrix trajectory and a vector trajectory.

        Perform a discrete convolution of two trajectories, one with matrix
        states and the other with vector states, using in-built numpy#
        functions.

        Parameters
        ----------
        traj1, traj2 : Trajectory
            Trajectories of the same size to be convolved

        Returns
        -------
        Trajectory
    """
    return Trajectory(conv_mat_vec_fast(traj1.modes, traj2.modes))

def traj_conv_mat_mat(traj1, traj2):
    """
        Return the convolution of two matrix trajectories.

        Perform a discrete convolution of two trajectories with matrix states
        using in-built numpy functions.

        Parameters
        ----------
        traj1, traj2 : Trajectory
            Trajectories of the same size to be convolved

        Returns
        -------
        Trajectory
    """
    return Trajectory(conv_mat_mat_fast(traj1.modes, traj2.modes))

def traj_grad(traj):
    """
        Return the gradient of a trajectory.

        Parameters
        ----------
        traj : Trajectory

        Returns
        -------
        Trajectory
    """
    # initialise array of mode modifiers to be multiplied
    modifiers = np.transpose(np.tile(1j*np.arange(traj.shape[0]), (traj.shape[1], 1)))

    # multiply element-wise
    new_modes = modifiers*traj.modes

    # force end mode to be zero to preserve symmetry
    new_modes[-1][:] = 0

    return Trajectory(new_modes)

def traj_response(traj, func):
    """
        Return the response of a trajectory over its length due to a function.

        Parameters
        ----------
        traj : Trajectory
        func : function

        Returns
        -------
        Trajectory
    """
    # convert trajectory to time domain
    curve = traj_irfft(traj)

    # evaluate response in time domain
    new_curve = func(curve)

    # convert back to frequency domain and return
    return traj_rfft(new_curve)
