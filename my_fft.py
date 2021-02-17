# This file contains the function definition for my FFT function that acts as 
# a wrapper to the numpy FFT function, mainly to properly take into account
# the effects of normalisation.

import numpy as np
from traj_util import list2array, array2list

def my_fft(array):
    """
        This function takes in a an array of the time domain representation of
        a state-space trajectory and returns an instance of the associated
        trajectory object instance.

        Parameters
        ----------
        traj_array: np.ndarray
            the array representing the time domain trajectory
        
        Returns
        -------
        traj_ob: list
            the associated trajectory given as an instance of the trajectory
            object class (and so with Fourier modes)
    """
    return np.fft.rfft(array, axis = 0)/np.shape(array)[0]

def my_ifft(array):
    """
        This function takes in an instance of the trajectory object class and
        returns an array of the associated time domain representation.

        Parameters
        ----------
        traj: list
            an instance of the trajectory object class to be transformed into
            the time domain
        
        Returns
        -------
        traj_time: np.ndarray
            the time domain representation of the input trajectory instance
    """
    return np.fft.irfft(array*(2*(np.shape(array)[0] - 1)), axis = 0)
