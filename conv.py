# This file contains the function definition for the convolution sum operation
# between two real valued Fourier mode pairs.

import numpy as np
from my_fft import my_fft, my_ifft, my_rfft, my_irfft

def conv_scalar_fast(scalar1, scalar2):
    # convert to full domain
    scalar1_full = np.fft.fftshift(my_fft(my_irfft(scalar1)), axes = 0)
    scalar2_full = np.fft.fftshift(my_fft(my_irfft(scalar2)), axes = 0)

    # perform convolution
    conv_array_full = np.convolve(scalar1_full, scalar2_full, mode = 'same')

    # truncate to include on rfft outputs
    conv_array = conv_array_full[np.shape(scalar1)[0]:]
    conv_array = np.append(conv_array, [conv_array_full[1], conv_array_full[0]])

    return conv_array

def conv_scalar(scalar1, scalar2, method = "sum"):
    # initialise arrays
    conv_array = np.zeros_like(scalar1)

    # nested loop to manually perform convolution
    for n in range(np.shape(scalar1)[0]):
        for m in range(1 - np.shape(scalar1)[0], np.shape(scalar1)[0]):
            if m < 0:
                x_m = np.conj(scalar1[-m])
            else:
                x_m = scalar1[m]
            if n - m < 0:
                y_nm = np.conj(scalar2[m - n])
            elif n - m > np.shape(scalar1)[0] - 1:
                y_nm = 0
            else:
                y_nm = scalar2[n - m]
            conv_array[n] += x_m*y_nm
    
    return conv_array

def conv_array(array1, array2, method = "sum"):
    # initialise arrays
    matmul_temp = np.matmul(array1[0], array2[0])
    conv_array = np.zeros([np.shape(array1)[0], *np.shape(matmul_temp)], dtype = complex)

    # nested loop to manually perform convolution
    for n in range(np.shape(array1)[0]):
        for m in range(1 - np.shape(array1)[0], np.shape(array1)[0]):
            if m < 0:
                x_m = np.conj(array1[-m])
            else:
                x_m = array1[m]
            if n - m < 0:
                y_nm = np.conj(array2[m - n])
            elif n - m > np.shape(array1)[0] - 1:
                y_nm = np.zeros_like(array2[0])
            else:
                y_nm = array2[n - m]
            conv_array[n] += np.matmul(x_m, y_nm)

    return conv_array
