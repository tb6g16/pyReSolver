# This file contains the function definitions for some trajectory-trajectory
# and trajectory-system interactions.

import numpy as np
from Trajectory import Trajectory
from my_fft import my_rfft, my_irfft
from traj_util import array2list, list2array
from conv import conv_array

def transpose(traj):
    new_traj = array2list(np.zeros(traj.shape, dtype = complex))
    for i in range(len(traj.mode_list)):
        new_traj[i] = np.transpose(traj.mode_list[i])
    return Trajectory(new_traj)

def conj(traj):
    new_traj = array2list(np.zeros(traj.shape, dtype = complex))
    for i in range(len(traj.mode_list)):
        new_traj[i] = np.conj(traj.mode_list[i])
    return Trajectory(new_traj)

def traj_rfft(array):
    return Trajectory(array2list(my_rfft(array)))

def traj_irfft(traj):
    return my_irfft(list2array(traj.mode_list))

def traj_conv(traj1, traj2):
    return Trajectory(array2list(conv_array(list2array(traj1.mode_list), list2array(traj2.mode_list))))

def traj_grad(traj):
    """
        This function calculates the gradient vectors of a given trajectory and
        returns it as an instance of the Trajectory class. The algorithm used
        is based off of the RFFT

        Parameters
        ----------
        traj: Trajectory object
            the trajectory which we will calculate the gradient for
        
        Returns
        -------
        grad: Trajectory object
            the gradient of the input trajectory
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
        This function evaluates the response over the domain of a given
        trajectory due to a given dyanmical system.

        Parameters
        ----------
        traj: Trajectory object
            the trajectory over which the response will be evaluated
        func: function
            the function that defines the response at each location along the
            trajectory, must be a function that inputs a vector and outputs a
            vector of the same size
        
        Returns
        -------
        response_traj: Trajectory object
            the response at each location of the trajectory, given as an
            instance of the Trajectory class
    """
    # convert trajectory to time domain
    curve = array2list(traj_irfft(traj))
    mode_no = len(curve)

    # evaluate response in time domain
    for i in range(mode_no):
        curve[i] = func(curve[i])

    # convert back to frequency domain and return
    return traj_rfft(list2array(curve))
