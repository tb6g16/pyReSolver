# This file contains the function definitions that will optimise a given
# trajectory and fundamental frequency to find the lowest global residual for
# a given dynamical system.

import numpy as np
from scipy.optimize import minimize

from ResolventSolver.Trajectory import Trajectory
from ResolventSolver.traj2vec import traj2vec, vec2traj
import ResolventSolver.residual_functions as res_funcs
from ResolventSolver.trajectory_functions import transpose, conj

def init_opt_funcs(freq, sys, dim, mean, psi = None, conv_method = 'fft'):
    """
        Return the functions to allow the calculation of the global residual
        and its associated gradients with a vector derived from a trajectory
        frequency pair.

        Parameters
        ----------
        sys : file
            File containing the necessary function definitions to define the
            state-space.
        dim : positive int
            Dimension of the state-space the trajectory is in so it can be
            unpacked by the vec2traj function.
        mean : ndarray
            1D array containing data of float type.
        psi : ndarray, default=None
            2D array containing data of float type, should be multiplicatively
            compatible with the trajectory.
        conv_method : {'fft', 'sum'}, default='fft'
            The convolution method used.
        
        Returns
        -------
        traj_global_res, traj_global_res_jac : function
            The global residual and global residual gradient functions
            respectively.
    """
    def traj_global_res(opt_vector):
        """
            Return the global residual of a trajectory frequency pair given as
            a vector.

            Parameters
            ----------
            opt_vector : ndarray
                1D array containing data of float type.

            Returns
            -------
            float
        """
        # unpack trajectory
        traj = vec2traj(opt_vector, dim)

        # convert to full space if singular matrix is provided
        if psi is not None:
            traj = traj.matmul_left_traj(psi)

        # calculate global residual and return
        H_n_inv = res_funcs.init_H_n_inv(traj, sys, freq, mean)
        return res_funcs.global_residual(res_funcs.local_residual(traj, sys, mean, H_n_inv))

    def traj_global_res_jac(opt_vector):
        """
            Return the gradient of the global residual with respect to the
            trajectory and frequency from a trajectory frequency pair given as
            a vector.

            Parameters
            ----------
            opt_vector : ndarray
                1D array containing data of float type.

            Returns
            -------
            traj_global_res : Trajectory
                Gradient of the global residual with respect to the trajectory.
            traj_global_res_jac : float
                Gradient of the global residual with respect to the frequency.
        """
        # unpack trajectory
        traj = vec2traj(opt_vector, dim)

        # convert to full space if singular matrix is provided
        if psi is not None:
            traj = traj.matmul_left_traj(psi)

        # calculate global residual gradients
        H_n_inv = res_funcs.init_H_n_inv(traj, sys, freq, mean)
        local_res = res_funcs.local_residual(traj, sys, mean, H_n_inv)
        gr_traj_grad = res_funcs.gr_traj_grad(traj, sys, freq, mean, local_res, conv_method = conv_method)
        gr_freq_grad = res_funcs.gr_freq_grad(traj, local_res)

        # convert gradient w.r.t modes to reduced space
        if psi is not None:
            gr_traj_grad = gr_traj_grad.matmul_left_traj(transpose(conj(psi)))

        # convert back to vector and return
        return traj2vec(gr_traj_grad)

    return traj_global_res, traj_global_res_jac

def my_min(traj, freq, sys, mean, **kwargs):
    """
        Return the trajectory that minimises the global residual given the
        system defining the state-space and the mean of the trajectory.

        Parameters
        ----------
        traj : Trajectory
        freq : float
        sys : file
            File containing the necessary function definitions to define the
            state-space.
        mean : ndarray
            1D array containing data of float type.
        use_jac : bool, default=True
            Whether or not to use the gradient in the optimisation algorithm.
        res_func : function, default=None
            An alternative residual function to use.
        jac_func : function, default=None
            An alternative gradient function to use.
        method : str, default='L-BFGS-B'
            The optimisation algorithm to use.
        quiet : bool, default=False
            Whether or not to operate the optimise in quiet mode.
        iter : positive int, default None
            The maximum number of iterations before terminating.
        traces : dictionary, default=None
            The dictionary that keeps track of all the important information
            during the optimisation.
        conv_method : {'fft', 'sum'}, default='fft'
            The convolution method used.
        psi : ndarray, default=None
            2D array containing data of type float.
        
        Returns
        -------
        op_traj : Trajectory
        op_freq : float
        traces : dictionary
        sol : OptimizeResult
            The result of the optimisation, default output for scipy minimize
            function.
    """
    # unpack keyword arguments
    use_jac = kwargs.get('use_jac', True)
    res_func = kwargs.get('res_func', None)
    jac_func = kwargs.get('jac_func', None)
    my_method = kwargs.get('method', 'L-BFGS-B')
    if_quiet = kwargs.get('quiet', False)
    maxiter = kwargs.get('iter', None)
    traces = kwargs.get('traces', None)
    conv_method = kwargs.get('conv_method', 'fft')
    psi = kwargs.get('psi', None)

    # convert to reduced space if singular matrix is provided
    if psi is not None:
        traj = traj.matmul_left_traj(transpose(conj(psi)))

    # setup the problem
    dim = traj.shape[1]
    if not hasattr(res_func, '__call__') and not hasattr(jac_func, '__call__'):
        res_func, jac_func = init_opt_funcs(freq, sys, dim, mean, psi = psi, conv_method = conv_method)
    elif not hasattr(res_func, '__call__'):
        res_func, _ = init_opt_funcs(freq, sys, dim, mean, psi = psi, conv_method = conv_method)
    elif not hasattr(jac_func, '__call__'):
        _, jac_func = init_opt_funcs(freq, sys, dim, mean, psi = psi, conv_method = conv_method)

    # define varaibles to be tracked using callback
    if traces == None:
        traces = {'traj': []}

    # define callback function
    def callback(x):
        cur_traj = vec2traj(x, dim)

        # convert to full space if singular matrix is provided
        if psi is not None:
            cur_traj = cur_traj.matmul_left_traj(psi)

        traces['traj'].append(cur_traj)

    # convert trajectory to vector of optimisation variables
    traj_vec = traj2vec(traj)

    # initialise options
    options = {}

    # if quiet
    if if_quiet == True:
        options['disp'] = False
    else:
        options['disp'] = True

    # maximum number of iterations
    if maxiter != None:
        options['maxiter'] = maxiter

    # perform optimisation
    if use_jac == True:
        sol = minimize(res_func, traj_vec, jac = jac_func, method = my_method, callback = callback, options = options)
    elif use_jac == False:
        sol = minimize(res_func, traj_vec, method = my_method, callback = callback, options = options)

    # unpack trajectory from solution
    op_vec = sol.x
    op_traj = vec2traj(op_vec, dim)

    # convert to full space if singular matrix is provided
    if psi is not None:
        op_traj = op_traj.matmul_left_traj(psi)

    return op_traj, traces, sol
