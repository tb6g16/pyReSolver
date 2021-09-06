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

    # evaluate resolvent arrays (including zero)
    resolvent_inv = Trajectory((1j*freq*np.tile(np.arange(no_modes), (dim, dim, 1)).transpose())*np.identity(dim) - jac_at_mean)

    # set zero mode to zero
    resolvent_inv[0] = 0

    return resolvent_inv

def init_H_n_inv(traj, sys, freq, mean):
    jac_at_mean = sys.jacobian(mean)
    return resolvent_inv(traj.shape[0], freq, jac_at_mean)

def local_residual(traj, sys, freq, mean, H_n_inv):
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
    # evaluate inner product of local residuals
    gr_sum = np.copy((traj_funcs.conj(local_res).matmul_left_traj(local_res)).modes)

    # scale zero modes
    gr_sum[0] = 0.5*gr_sum[0]

    # sum and return real part
    return np.real(np.sum(gr_sum))

def gr_traj_grad(traj, sys, freq, mean, local_res, conv_method = 'fft'):
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
    res_grad = traj_funcs.traj_grad(local_res)

    # initialise jacobian function and take transpose
    traj[0] = mean
    jac = traj_funcs.traj_response(traj, sys.jacobian)
    traj[0] = 0
    jac = traj_funcs.transpose(jac)

    # perform convolution
    jac_res_conv = traj_funcs.traj_conv(jac, local_res, method = conv_method)

    # calculate and return gradients w.r.t trajectory and frequency respectively
    return (-freq*res_grad) - jac_res_conv

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
    # initialise sum and loop over modes
    sum = 0
    for n in range(1, traj.shape[0]):
        sum += n*np.imag(np.dot(np.conj(traj[n]), local_res[n]))
    sum = 2*sum

    # return sum
    # return 0.001*sum
    return 0.0
