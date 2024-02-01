# This file provides a few simple functions to perform FFT and IFFT operations
# without having to initialise plans.

import numpy as np

def rfft(array):
    return np.fft.rfft(array, axis = 0)/np.shape(array)[0]

def irfft_even(array):
    return np.fft.irfft(array*2*(np.shape(array)[0] - 1), axis = 0) # works if original function had even length

def irfft_odd(array):
    return np.fft.irfft(array*(2*np.shape(array)[0] - 1), 2*np.shape(array)[0] - 1, axis = 0) # works if original function had odd length
