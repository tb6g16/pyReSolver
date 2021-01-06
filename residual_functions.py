# This file contains the function definitions that calculate the residuals and
# their associated gradients

import numpy as np
import scipy.integrate as integ
from Trajectory import Trajectory
from System import System
import trajectory_functions as traj_funcs

def local_residual(traj, sys, freq):
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
        
        Returns
        -------
        residual_traj: Trajectory object
            the local residual of the trajectory with respect to the dynamical
            system, given as an instance of the Trajectory class
    """
    # compute gradient of trajectory
    traj_grad = traj_funcs.traj_grad(traj)

    # evaluate system response at the states of the trajectory
    response = traj_funcs.traj_response(traj, sys.response)

    # compute and return local residual trajectory
    return (freq*traj_grad) - response

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
