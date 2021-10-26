# This file contains the function definition for my FFT function that acts as 
# a wrapper to the numpy FFT function, mainly to properly take into account
# the effects of normalisation.

import numpy as np

def my_fft(array):
    """
        Return the properly normalised FFT of an array.

        Parameters
        ----------
        array : ndarray
            N-D array containing data of float type.

        Returns
        -------
        ndarray
            N-D array containing data of float type.
    """
    return np.fft.fft(array, axis = 0)/np.shape(array)[0]

def my_ifft(array):
    """
        Return the properly normalised inverse FFT of an array.

        Parameters
        ----------
        array : ndarray
            N-D array containing data of float type.
        
        Returns
        -------
        ndarray
            N-D array containing data of float type.
    """
    return np.fft.ifft(array*np.shape(array)[0], axis = 0)

def my_rfft(array):
    """
        Return the properly normalised real FFT of an array.

        Parameters
        ----------
        array : ndarray
            N-D array containing data of float type.
        
        Returns
        -------
        ndarray
            N-D array containing data of float type.
    """
    return np.fft.rfft(array, axis = 0)/np.shape(array)[0]

def my_irfft(array):
    """
        Return the properly normalise inverse real FFT of an array.

        Parameters
        ----------
        array : ndarray
            N-D array containing data of float type.

        Returns
        -------
        ndarray
            N-D array containing data of float type.
    """
    # return np.fft.irfft(array*(2*np.shape(array)[0] - 1), 2*np.shape(array)[0] - 1, axis = 0) # works if original function had odd length
    return np.fft.irfft(array*2*(np.shape(array)[0] - 1), axis = 0) # works if original function had even length
