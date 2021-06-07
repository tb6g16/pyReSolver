# This file contains the function definitions for some trajectory-trajectory
# and trajectory-system interactions.

import numpy as np
from Trajectory import Trajectory
from my_fft import my_rfft, my_irfft
from traj_util import array2list, list2array
from conv import conv_array

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
    # initialise new trajectory list
    new_traj = array2list(np.zeros(traj.shape, dtype = complex))

    # loop over given traj and transpose it
    for i in range(len(traj.mode_list)):
        new_traj[i] = np.transpose(traj.mode_list[i])

    return Trajectory(new_traj)

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
    # initialise new trajectory list
    new_traj = array2list(np.zeros(traj.shape, dtype = complex))

    # loop over given trajectory and take its complex conjugate
    for i in range(len(traj.mode_list)):
        new_traj[i] = np.conj(traj.mode_list[i])

    return Trajectory(new_traj)

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
    return Trajectory(array2list(my_rfft(array)))

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
    return my_irfft(list2array(traj.mode_list))

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
    return Trajectory(array2list(conv_array(list2array(traj1.mode_list), list2array(traj2.mode_list), method = method)))

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
    # initialise array for new modes
    # new_modes = np.zeros(traj.shape, dtype = np.complex)
    new_mode_list = [None]*traj.shape[0]

    # loop over time and multiply modes by modifiers
    for k in range(traj.shape[0]):
        new_mode_list[k] = 1j*k*traj.mode_list[k]
    
    # force zero mode at end to preserve symmetry
    new_mode_list[-1][:] = 0

    return Trajectory(new_mode_list)

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
    curve = array2list(traj_irfft(traj))
    mode_no = len(curve)

    # evaluate response in time domain
    for i in range(mode_no):
        curve[i] = func(curve[i])

    # convert back to frequency domain and return
    return traj_rfft(list2array(curve))
