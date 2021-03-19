# This file contains the function definitions that calculate the residuals and
# their associated gradients.

import numpy as np
from Trajectory import Trajectory
from System import System
import trajectory_functions as traj_funcs

# CHANGE DOWN THE LINE FOR DIFFERENT RESIDUAL FUNCTION
def resolvent_inv(modes, freq, jac_at_mean):
    """
        This function calculates the resolvent operator for a given mode number
        n, system, and fundamental frequency.

        Parameters
        ----------
        n: positive integer
            mode number for the resolvent operator
        sys: System object
            the dynamical system required to calculate the Jacobian matrix
        freq: float
            the frequency of the trajectory
        dim: positive integer
            dimension of the dynamical system
        mean: vector
            the mean of the state-space trajectory
        
        Returns
        -------
        resolvent: numpy array
            the resolvent operator for the given mode number, system, and
            frequency
    """
    # evaluate the number of dimensions using the size of the jacobian
    dim = np.shape(jac_at_mean)[0]

    # initialise the resolvent list
    resolvent_inv = [None]*modes

    # loop over calculating the value at each wavenumber
    for n in range(modes):
        # resolvent[n] = np.linalg.inv((1j*n*freq*np.identity(dim)) - jac_at_mean)
        resolvent_inv[n] = (1j*n*freq*np.identity(dim)) - jac_at_mean

    # set mean mode resolvent to array of zeros
    resolvent_inv[0] = np.zeros_like(resolvent_inv[1])

    return Trajectory(resolvent_inv)

def local_residual(traj, sys, freq, mean):
    """
        This function calculates the local residual of a trajectory through a
        state-space defined by a given dynamical system.

        Parameters
        ----------
        traj: Trajectory object
            the trajectory through state-space
        sys: System object
            the dynamical system defining the state-space
        freq: float
            the fundamental frequency of the trajectory
        mean: vector
            the mean of the state-space trajectory
        
        Returns
        -------
        residual_traj: Trajectory object
            the local residual of the trajectory with respect to the dynamical
            system, given as an instance of the Trajectory class
    """
    # local_res = np.zeros_like(traj.modes)

    # L = sys.jacobian(mean)
    # L_prod = L @ traj
    # resp = traj_funcs.traj_response(traj, sys.nl_factor)

    # local_res[:, 0] = -sys.response(mean) - resp[:, 0]
    # for i in range(1, np.shape(local_res)[1]):
    #     local_res[:, i] = (freq*grad[:, i]) - L_prod[:, i] - resp[:, i]

    # return Trajectory(local_res)

    # evaluate jacobian at the mean
    jac_at_mean = sys.jacobian(mean)

    # evaluate the resolvents (and invert!!!)
    H_n_inv = resolvent_inv(traj.shape[0], freq, jac_at_mean)
    # for i in range(1, H_n.shape[0]):
    #     H_n[i] = np.linalg.inv(H_n[i])

    # evaluate response and multiply by resolvent at every mode
    resp = traj_funcs.traj_response(traj, sys.nl_factor)
    # H_resp_mult = H_n @ resp
    # print(resp[1])

    # evaluate local residual trajectory for all modes
    H_inv_traj_mult = H_n_inv @ traj
    local_res = H_inv_traj_mult - resp
    # local_res = traj - H_resp_mult

    # reassign the mean mode to the second constraint
    local_res[0] = -sys.response(mean) - resp[0]

    return local_res

def global_residual(traj, sys, freq, mean):
    """
        This function calculates the global residual of a trajectory through a
        state-space defined by a given dynamical system.

        Parameters
        ----------
        traj: Trajectory object
            the trajectory through state-space
        sys: System object
            the dynamical system defining the state-space
        freq: float
            the fundamental frequency of the trajectory
        
        Returns
        -------
        global_res: float
            the global residual of the trajectory-system pair
    """
    # obtain set of local residual vectors
    local_res = local_residual(traj, sys, freq)

    # take norm of the local residual vectors
    local_res_norm_sq = traj_funcs.traj_inner_prod(local_res, local_res)

    # integrate over the discretised time
    return 0.5*traj_funcs.average_over_s(local_res_norm_sq)[0]

def global_residual_grad(traj, sys, freq, mean):
    """
        This function calculates the gradient of the global residual with
        respect to the trajectory and the associated fundamental frequency for
        a trajectory through a state-space defined by a given dynamical system.

        Parameters
        ----------
        traj: Trajectory object
            the trajectory through state-space
        sys: System object
            the dynamical system defining the state-space
        freq: float
            the fundamental frequency of the trajectory
        
        Returns
        -------
        d_gr_wrt_traj: Trajectory object
            the gradient of the global residual with respect to the trajectory,
            given as an instance of the Trajectory class
        d_gr_wrt_freq: float
            the gradient of the global residual with respect to the trajectory
    """
    # calculate local residual trajectory
    local_res = local_residual(traj, sys, freq)

    # calculate trajectory gradients
    traj_grad = traj_funcs.traj_grad(traj)
    res_grad = traj_funcs.traj_grad(local_res)

    # initialise jacobian function
    jacob_func = traj_funcs.jacob_init(traj, sys, if_transp = True)

    # take norm of trajectory
    traj_grad_norm_sq = traj_funcs.traj_inner_prod(traj_grad, traj_grad)

    # take response of trajectory to dynamical system
    traj_resp = traj_funcs.traj_response(traj, sys.response)

    # define integrand trajectory to be integrated
    int_traj = (freq*traj_grad_norm_sq) - \
        (traj_funcs.traj_inner_prod(traj_grad, traj_resp))

    # calculate and return gradients w.r.t trajectory and frequency respectively
    return ((-freq*res_grad) - (jacob_func @ local_res))*(1/(40*np.pi)), \
        traj_funcs.average_over_s(int_traj)[0]
