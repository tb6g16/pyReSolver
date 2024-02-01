# This file contains the definitions required to compute the singular value
# decomposition of the resolvent operator.

import numpy as np

from .Trajectory import Trajectory
from .trajectory_functions import transpose, conj

def resolvent_inv(no_modes, freq, jac_at_mean):
    """
        Return the inverse resolvent array at a given number of modes.

        Parameters
        ----------
        no_modes : positive integer
            The number of modes at which to evaluate the resolvent.
        freq : float
        jac_at_mean : ndarray
            2D array containing data of float type.
        
        Returns
        -------
        Trajectory
    """
    # evaluate the number of dimensions using the size of the jacobian
    dim = np.shape(jac_at_mean)[0]

    # evaluate resolvent arrays (including zero)
    resolvent_inv = Trajectory((1j*freq*np.tile(np.arange(no_modes), (dim, dim, 1)).transpose())*np.identity(dim) - jac_at_mean)

    # set zero mode to zero
    resolvent_inv[0] = 0

    return resolvent_inv

def resolvent(freq, n, jac_at_mean, B):
    """
        Return the resolvent matrix for a given modenumber n.

        This resolvent array can be modified by left multiplication of an
        optional array B.

        Parameters
        ----------
        freq : float
        n : positive int
        jac_at_mean : ndarray
            2D array containing data of float type.
        B : ndarray, optional
            2D array containing data of flaot type.

        Returns
        -------
        H_n : ndarray, default=None
            2D containing data of float type.
    """
    # evaluate the number of dimensions using the size of the jacobian
    dim = np.shape(jac_at_mean)[0]

    # calculate single resolvent matrix if n is an integer
    shape = np.shape(np.zeros([dim, dim]) @ B)
    H_n = Trajectory(np.zeros([n[-1] + 1, *shape], dtype = complex))
    for i in n:
        H_n[i] = np.linalg.inv(1j*i*freq*np.eye(dim) - jac_at_mean) @ B

    return H_n

def resolvent_modes(resolvent, cut = 0):
    """
        Return the SVD of a resolvent array at every mode number.

        Parameters
        ----------
        resolvent : Trajectory
            2D array containing data of float type.
        cut : positive int, default=0
            The number of singular modes to exclude.
        
        Returns
        -------
        psi, sig, phi : Trajectory
    """
    # perform full svd
    psi, sig_vec, phi = np.linalg.svd(resolvent, full_matrices = False)

    # diagonalize singular value matrix and convert all to lists
    sig = np.zeros([resolvent.shape[0], sig_vec.shape[1], sig_vec.shape[1]], dtype = float)
    for i in range(resolvent.shape[0]):
        sig[i] = np.diag(sig_vec[i])

    # cut off the desired number of singular values
    if cut != 0:
        sig = sig[:, :-cut, :-cut]
        psi = psi[:, :, :-cut]
        phi = phi[:, :-cut, :]

    return psi, Trajectory(sig), conj(transpose(phi))
