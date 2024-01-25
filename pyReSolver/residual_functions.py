# This file contains the function definitions that calculate the residuals and
# their associated gradients.

import numpy as np

from pyReSolver.Trajectory import Trajectory
import pyReSolver.trajectory_functions as traj_funcs
from pyReSolver.Cache import Cache

def local_residual(cache, sys, H_n_inv, fftplans):
    """
        Return the local residual of a trajectory in a state-space.

        Parameters
        ----------
        traj : Trajectory
        sys : file
            File containing the necessary function definitions to define the
            state-space.
        freq : flaot
        mean : ndarray
            1D array containing data of float type.
        
        Returns
        -------
        local_res : Trajectory
    """
    # evaluate response and multiply by resolvent at every mode
    traj_funcs.traj_response(cache.traj, fftplans, sys.nl_factor, cache.f, cache.tmp_t1)

    # evaluate local residual trajectory for all modes
    np.copyto(cache.lr, cache.traj.matmul_left_traj(H_n_inv) - cache.f)

    # reassign the mean mode to the second constraint
    cache.lr[0] = -cache.resp_mean - cache.f[0]

    return cache.lr

def global_residual(cache):
    """
        Return the global residual of a trajectory in a state-space.

        Parameters
        ----------
        local_res: Trajectory
            local residual for the trajectory in a system
        
        Returns
        -------
        float
    """
    # evaluate inner product of local residuals
    np.copyto(cache.tmp_inner, traj_funcs.conj(cache.lr).traj_inner(cache.lr))

    # scale zero modes
    cache.tmp_inner[0] = 0.5*cache.tmp_inner[0]

    # sum and return real part
    return np.real(np.sum(cache.tmp_inner))

def gr_traj_grad(cache, sys, freq, mean, fftplans):
    """
        Return the gradient of the global residual with respect to a trajectory
        in state-space.

        Parameters
        ----------
        traj : Trajectory
        sys : file
            File containing the necessary function definitions to define the
            state-space.
        freq : float
        mean : ndarray
            1D array containing data of float type.
        conv_method : {'fft', 'sum'}, default='fft'
            Method to use for the convolution
        
        Returns
        -------
        Trajectory
    """
    # calculate trajectory gradients
    traj_funcs.traj_grad(cache.lr, cache.lr_grad)

    # calculate jacobian residual convolution
    cache.traj[0] = mean
    traj_funcs.traj_response2(cache.traj, cache.lr, fftplans, sys.jac_conv_adj, cache.tmp_conv, cache.tmp_t1, cache.tmp_t2)
    cache.traj[0] = 0

    # calculate and return gradients w.r.t trajectory and frequency respectively
    return -freq*cache.lr_grad - cache.tmp_conv

def gr_freq_grad(traj, local_res):
    """
        Return the gradient of the global residual with respect to the
        frequency of a trajectory in state-space.

        Parameters
        ----------
        traj : Trajectory
        sys : file
            File containing the necessary function definitions to define the
            state-space.
        freq : float
        mean : ndarray
            1D array containing data of float type.

        Returns
        -------
        float
    """
    return np.sum(2*np.arange(traj.shape[0])*np.imag(traj_funcs.conj(traj).traj_inner(local_res)))
    # return 0.001*np.sum(2*np.arange(traj.shape[0])*np.imag(traj_funcs.conj(traj).traj_inner(local_res)))
    # return 0.0
