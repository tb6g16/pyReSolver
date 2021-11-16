# This file contains the function definition for several utility functions
# required by the Trajectory operations performed.

import numpy as np

def func2curve(traj_func, modes, if_freq = True):
    """
        Return the array of a function evaluated at a number of locations.

        Parameters
        ----------
        traj_func : function
        modes : int
            The number of modes for the FFT of the output array.
        
        Returns
        -------
        traj_array : ndarray
            N-D array containing data of float type.
    """
    # convert the number of modes the discretisation in time domain
    if if_freq:
        disc = 2*(modes - 1)
    else:
        disc = modes

    # initialise the output array and the parameter of the curve
    traj_array = np.zeros([disc, *np.shape(traj_func(0))])
    s = np.linspace(0, 2*np.pi*(1 - 1/disc), disc)

    # loop over the parameter of the curve evaluating the function
    for i in range(disc):
        traj_array[i, :] = traj_func(s[i])

    return traj_array
