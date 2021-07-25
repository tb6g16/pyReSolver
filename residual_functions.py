# This file contains the function definitions that calculate the residuals and
# their associated gradients.

import numpy as np

from Trajectory import Trajectory
import trajectory_functions as traj_funcs

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

    # initialise the resolvent array
    resolvent_inv = Trajectory(np.zeros([no_modes, dim, dim], dtype = complex))

    # loop over calculating the value at each wavenumber
    # GET RID OF LOOP IF POSSIBLE
    for n in range(1, no_modes):
        # resolvent[n] = np.linalg.inv((1j*n*freq*np.identity(dim)) - jac_at_mean)
        resolvent_inv[n] = (1j*n*freq*np.identity(dim)) - jac_at_mean

    return resolvent_inv

def local_residual(traj, sys, freq, mean):
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
    # evaluate jacobian at the mean
    jac_at_mean = sys.jacobian(mean)

    # evaluate the inverse resolvents
    H_n_inv = resolvent_inv(traj.shape[0], freq, jac_at_mean)

    # evaluate response and multiply by resolvent at every mode
    resp = traj_funcs.traj_response(traj, sys.nl_factor)

    # evaluate local residual trajectory for all modes
    local_res = traj.matmul_left_traj(H_n_inv) - resp

    # reassign the mean mode to the second constraint
    local_res[0] = -sys.response(mean) - resp[0]

    return local_res

def global_residual(local_res):
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
    # initialise and sum the norms of the complex residual vectors
    sum = 0
    for n in range(1, local_res.shape[0]):
        sum += np.dot(np.conj(local_res[n]), local_res[n])

    # add the zero mode for the mean constraint
    sum += 0.5*np.dot(np.conj(local_res[0]), local_res[0])

    return np.real(sum)

def gr_traj_grad(traj, sys, freq, mean, conv_method = 'fft'):
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
    # calculate local residual trajectory
    local_res = local_residual(traj, sys, freq, mean)

    # calculate trajectory gradients
    res_grad = traj_funcs.traj_grad(local_res)

    # initialise jacobian function and take transpose
    traj[0] = np.array(mean, dtype = complex)
    jac = traj_funcs.traj_response(traj, sys.jacobian)
    traj[0] = np.zeros_like(traj[1])
    jac = traj_funcs.transpose(jac)

    # perform convolution
    jac_res_conv = traj_funcs.traj_conv(jac, local_res, method = conv_method)

    # calculate and return gradients w.r.t trajectory and frequency respectively
    # return 2*((-freq*res_grad) - jac_res_conv)
    return (-freq*res_grad) - jac_res_conv

def gr_freq_grad(traj, sys, freq, mean):
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
    # calculate local residual
    local_res = local_residual(traj, sys, freq, mean)

    # initialise sum and loop over modes
    sum = 0
    for n in range(1, traj.shape[0]):
        sum += n*np.imag(np.dot(np.conj(traj[n]), local_res[n]))
    sum = 2*sum

    # return sum
    # return 0.001*sum
    return 0
