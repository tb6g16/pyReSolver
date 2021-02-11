# This file contains the function definitions that calculate the residuals and
# their associated gradients.

import numpy as np
import scipy.integrate as integ
from Trajectory import Trajectory
from System import System
import trajectory_functions as traj_funcs

def resolvent(n, sys, freq, dim, mean):
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
    return np.linalg.inv((1j*n*freq*np.identity(dim)) - sys.jacobian(mean))

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
    # initialise arrays
    residual_traj = np.zeros(traj.shape, dtype = complex)

    # evaluate response
    resp = traj_funcs.traj_response(traj, sys.nl_factor)

    # loop through mode numbers
    for i in range(1, traj.shape[1]):
        H_n = resolvent(i, sys, freq, traj.shape[0], mean)
        residual_traj[:, i] = traj[:, i] - (H_n @ resp[:, i])

    return Trajectory(residual_traj)

def global_residual(traj, sys, freq):
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

def global_residual_grad(traj, sys, freq):
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
