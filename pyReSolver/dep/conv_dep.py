# This file contains the function definition for the convolution sum operation
# between two real valued Fourier mode pairs.

import numpy as np

from pyReSolver.my_fft import my_fft, my_ifft, my_rfft, my_irfft

def conv_scalar_fast(scalar1, scalar2):
    """
        Return convolution of two scalar arrays using in-built numpy function.

        Perform a discrete convolution over two sets of scalars using the numpy
        in-built convolve function.

        Paramaters
        ----------
        scalar1, scalar2 : ndarray
            1D arrays containing data of float type.
        
        Returns
        -------
        conv_array: ndarray
            1D array containing data of float type.
    """
    # convert to full domain
    scalar1_full = np.fft.fftshift(my_fft(my_irfft(scalar1)), axes = 0)
    scalar2_full = np.fft.fftshift(my_fft(my_irfft(scalar2)), axes = 0)

    # perform convolution
    conv_array_full = np.convolve(scalar1_full, scalar2_full, mode = 'same')

    # truncate to include on rfft outputs
    conv_array = conv_array_full[np.shape(scalar1)[0]:]
    conv_array = np.append(conv_array, [np.conj(conv_array_full[2]), np.conj(conv_array_full[1])])

    return conv_array

def conv_vec_vec_fast(array1, array2):
    """
        Return convolution of two vector arrays using in-built numpy functions.

        Perform a discrete convolution over two sets of vectors using the
        numpy in-built convolve function.

        Parameters
        ----------
        array1, array2 : ndarray
            2D arrays containing data of float type.

        Returns
        -------
        conv : ndarray
            1D array containing data of float type.
    """
    # initialise convolution array
    conv = np.zeros([np.shape(array1)[0], *np.shape(np.matmul(array1[0], array2[0]))], dtype = complex)

    # loop over the dimensions
    for i in range(np.shape(array1)[1]):
        # pick out the mode vector for each dimension
        array1_at_i = array1[:, i]
        array2_at_i = array2[:, i]

        # convolve the two vectors and add to the sum over this dimension
        conv[:] += conv_scalar_fast(array1_at_i, array2_at_i)

    return conv

def conv_mat_vec_fast(array1, array2):
    """
        Return convolution of a matrix array and a vector array using in-built
        numpy functions.

        Perform a discrete convolution over a set of matrices and a set of
        vectors using the numpy in-built convolve function.

        Parameters
        ----------
        array1, array2 : ndarray
            2D arrays containing data of float type.

        Returns
        -------
        conv : ndarray
            1D array containing data of float type.
    """
    # initialise convolution array
    conv = np.zeros([np.shape(array1)[0], *np.shape(np.matmul(array1[0], array2[0]))], dtype = complex)

    # loop over the dimensions
    for i in range(np.shape(array1)[1]):
        conv_at_i = 0
        for j in range(np.shape(array1)[2]):
            # pick out the mode matrix and vector for each dimension
            array1_at_ij = array1[:, i, j]
            array2_at_j = array2[:, j]

            # convolve the matrix and vector and add to sum over this dimension
            conv_at_i += conv_scalar_fast(array1_at_ij, array2_at_j)

        # assign convolution sum for current dimension
        conv[:, i] = conv_at_i

    return conv

def conv_mat_mat_fast(array1, array2):
    """
        Return convolution of two matrix arrays using in-built numpy functions.

        Perform a discrete convolution over two sets of matrices using the
        numpy in-built convolve function.

        Parameters
        ----------
        array1, array2 : ndarray
            2D arrays containing data of float type.

        Returns
        -------
        conv : ndarray
            1D array containing data of float type.
    """
    # initialise convolution array
    conv = np.zeros([np.shape(array1)[0], *np.shape(np.matmul(array1[0], array2[0]))], dtype = complex)

    # loop over the dimensions
    for i in range(np.shape(array1)[1]):
        for j in range(np.shape(array2)[2]):
            conv_at_ij = 0
            for k in range(np.shape(array1)[2]):
                # pick out the mode matrix and vector for each dimension
                array1_at_ik = array1[:, i, k]
                array2_at_kj = array2[:, k, j]

                # convolve the matrices and add to sum over this dimension
                conv_at_ij += conv_scalar_fast(array1_at_ik, array2_at_kj)

            # assign convolution sum for current dimension
            conv[:, i, j] = conv_at_ij

    return conv

def conv_scalar(scalar1, scalar2, method = 'fft'):
    """
        Return convolution of two scalar arrays.

        Perform a discrete convolution over two sets of scalars using either a
        direct sum or an indirect FFT approach.

        Parameters
        ----------
        scalar1, scalar2 : ndarray
            1D array containing data of float type.
        method : {'fft', 'sum'}, default='fft'
            The method used to perform the convolution.
        
        Returns
        -------
        ndarray
            1D array containing data of float type.
    """
    if method == 'direct':
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
    elif method == 'fft':
        # convert to time domain
        scalar1_time = my_irfft(scalar1)
        scalar2_time = my_irfft(scalar2)

        # initialise array and perform point-wise multiplication
        prod = scalar1_time*scalar2_time
        
        return my_rfft(prod)
    else:
        raise ValueError("Not a valid method!")

def conv_array(array1, array2, method = 'fft'):
    """
        Return convolution of two compatible arrays.

        Perform a discrete convolution over two sets of arrays (which can be
        multiplied together) using either a direct sum or indirect FFT
        approach.

        Parameters
        ----------
        array1, array2 : ndarray
            Arrays containing data of float type that can be multiplied
            together (using 'matmul').
        method : {'fft', 'sum'}, default='fft'
            The method used to perform the convolution.
        
        Returns
        -------
        ndarray
            Array containing data of float type.
    """
    if method == 'direct':
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
    elif method == 'fft':
        # convert to time domain
        array1_time = my_irfft(array1)
        array2_time = my_irfft(array2)

        # dummy array to find shape of result
        matmul_temp = np.matmul(array1_time[0], array2_time[1])

        # intialise array and perform mode-wise multiplication
        if len(array1_time.shape) == 2 and len(array2_time.shape) == 2:
            prod = np.diag(np.inner(array1_time, array2_time))
        elif len(array1_time.shape) == 3 and len(array2_time.shape) == 3:
            prod = np.matmul(array1_time, array2_time)
        else:
            prod = np.squeeze(np.matmul(array1_time, np.reshape(array2_time, (*np.shape(array2_time), 1))))

        return my_rfft(prod)
    else:
        raise ValueError("Not a valid method!")