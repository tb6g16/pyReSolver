# This file contains the function definitions that will optimise a given
# trajectory and fundamental frequency to find the lowest global residual for
# a given dynamical system.

import numpy as np
from scipy.optimize import minimize

from pyReSolver.Cache import Cache
from pyReSolver.FFTPlans import FFTPlans
from pyReSolver.traj2vec import traj2vec, vec2traj, init_comp_vec
from pyReSolver.init_opt_funcs import init_opt_funcs
from pyReSolver.resolvent_modes import resolvent_inv
from pyReSolver.trajectory_functions import transpose, conj

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
    time_shape = [(traj.shape[0] - 1) << 1, traj.shape[1]]
    flag = kwargs.get('flag', 'FFTW_EXHAUSTIVE')
    plans = kwargs.get('plans', FFTPlans(time_shape, flag = flag))
    use_jac = kwargs.get('use_jac', True)
    res_func = kwargs.get('res_func', None)
    jac_func = kwargs.get('jac_func', None)
    my_method = kwargs.get('method', 'L-BFGS-B')
    if_quiet = kwargs.get('quiet', False)
    maxiter = kwargs.get('iter', None)
    traces = kwargs.get('traces', None)
    psi = kwargs.get('psi', None)

    # initialise cache
    cache = Cache(traj, mean, sys, plans, psi)

    # convert to reduced space if singular matrix is provided
    if psi is not None:
        traj = traj.matmul_left_traj(transpose(conj(psi)))

    # setup the problem
    dim = traj.shape[1]
    if not hasattr(res_func, '__call__') and not hasattr(jac_func, '__call__'):
        res_func, jac_func = init_opt_funcs(cache, freq, plans, sys, mean, psi = psi)
    elif not hasattr(res_func, '__call__'):
        res_func, _ = init_opt_funcs(cache, freq, plans, sys, mean, psi = psi)
    elif not hasattr(jac_func, '__call__'):
        _, jac_func = init_opt_funcs(cache, freq, plans, sys, mean, psi = psi)

    # define varaibles to be tracked using callback
    if traces == None:
        traces = {'traj': []}

    # define callback function
    cur_traj = np.zeros_like(traj)
    def callback(x, cur_traj=cur_traj):
        vec2traj(cur_traj, x)

        # convert to full space if singular matrix is provided
        if psi is not None:
            cur_traj2 = cur_traj.matmul_left_traj(psi)
        else:
            cur_traj2 = cur_traj

        traces['traj'].append(cur_traj2)

    # convert trajectory to vector of optimisation variables
    traj_vec = init_comp_vec(traj)
    traj2vec(traj, traj_vec)

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
    op_traj = np.zeros_like(traj)
    op_vec = sol.x
    vec2traj(op_traj, op_vec)

    # convert to full space if singular matrix is provided
    if psi is not None:
        op_traj = op_traj.matmul_left_traj(psi)

    return op_traj, traces, sol
