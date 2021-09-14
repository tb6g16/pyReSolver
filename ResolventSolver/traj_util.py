# This file contains the function definition for several utility functions
# required by the Trajectory operations performed.

import numpy as np

def func2curve(traj_func, modes):
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
    disc = 2*(modes - 1)

    # initialise the output array and the parameter of the curve
    traj_array = np.zeros([disc, *np.shape(traj_func(0))])
    s = np.linspace(0, 2*np.pi*(1 - 1/disc), disc)

    # loop over the parameter of the curve evaluating the function
    for i in range(disc):
        traj_array[i, :] = traj_func(s[i])

    return traj_array

def list2array(my_list):
    """
        Return the array representation of a list.

        Convert a list of arrays with N dimensions into a single array with N+1
        dimensions.

        Parameters
        ----------
        my_list : list of ndarray
            List where each element is an array of the same shape and data
            type.

        Returns
        -------
        array : ndarray
            N-D array containing data of float type.
    """
    # initialise output array
    array = np.zeros([len(my_list), *np.shape(my_list[0])], dtype = my_list[0].dtype)

    # loop over elements of list and assign to the output array
    for i in range(len(my_list)):
        array[i] = my_list[i]

    return array

def array2list(array):
    """
        Return the list representation of an array.

        Convert an array of N+1 dimensions to an equivalent list where each
        element is an array of N dimensions.

        Parameters
        ----------
        array : ndarray
            N-D array containing data of float type.
        
        Returns
        -------
        my_list : list of ndarray
            Equivelent list containing data of ndarray types.
    """
    # initialising output list
    my_list = [None]*np.shape(array)[0]

    # loop over list assigning the arrays
    for i in range(len(my_list)):
        if type(array[i]) != np.ndarray:
            my_list[i] = np.array([array[i]])
        else:
            my_list[i] = array[i]

    return my_list
