# This file contains the function definitions that will optimise a given
# trajectory and fundamental frequency to find the lowest global residual for
# a given dynamical system.

import numpy as np
from scipy.optimize import minimize

from .Cache import Cache
from .FFTPlans import FFTPlans
from .traj2vec import traj2vec, vec2traj, init_comp_vec
from .init_opt_funcs import init_opt_funcs
from .trajectory_functions import transpose, conj

def minimiseResidual(traj, freq, sys, mean, **kwargs):
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
        traces : dictionary, default=None
            The dictionary that keeps track of all the important information
            during the optimisation.
        psi : ndarray, default=None
            2D array containing data of type float.
        plans : FFTPlans, default=from trajectory shape
            FFTW plans to perform the spectral to physical transformations.
        flag : str, default="FFTW_EXHAUSTIVE"
            FFTW flag to setup the default transform plans.
        store_grad : bool, default=False
            Whether or not to store the gradient norm in the trace
        options : dict, default={}
            Minimisation options exposed from the SciPy interface.
        callback : callable, default=x->None
            User-defined callback function

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
    flag = kwargs.get('flag', 'FFTW_EXHAUSTIVE')
    plans = kwargs.get('plans', FFTPlans([(traj.shape[0] - 1) << 1, traj.shape[1]], flag = flag))
    use_jac = kwargs.get('use_jac', True)
    res_func = kwargs.get('res_func', None)
    jac_func = kwargs.get('jac_func', None)
    my_method = kwargs.get('method', 'L-BFGS-B')
    traces = kwargs.get('traces', None)
    psi = kwargs.get('psi', None)
    options = kwargs.get("options", {})
    store_grad = kwargs.get("store_grad", False)
    user_callback = kwargs.get("callback", lambda *args : None)

    # initialise cache
    cache = Cache(traj, mean, sys, plans, psi)

    # convert to reduced space if singular matrix is provided
    if psi is not None:
        traj = traj.matmul_left_traj(transpose(conj(psi)))

    # setup the problem
    if not hasattr(res_func, '__call__') and not hasattr(jac_func, '__call__'):
        res_func, jac_func = init_opt_funcs(cache, freq, plans, sys, mean, psi=psi)
    elif not hasattr(res_func, '__call__'):
        res_func, _ = init_opt_funcs(cache, freq, plans, sys, mean, psi=psi)
    elif not hasattr(jac_func, '__call__'):
        _, jac_func = init_opt_funcs(cache, freq, plans, sys, mean, psi=psi)

    # define varaibles to be tracked using callback
    if traces is None:
        traces = {"residual": [], "gradient": [], "iteration": []}
        startIteration = 0
    else:
        startIteration = traces["iteration"][-1]
        del traces["residual"][-1]
        del traces["gradient"][-1]
        del traces["iteration"][-1]

    # define callback function
    if store_grad:
        def initCallback(currentIteration):
            gradient = np.zeros_like(traj)
            def callback(x):
                nonlocal currentIteration
                vec2traj(gradient, jac_func(x))
                traces["residual"].append(res_func(x))
                traces["gradient"].append(np.real(np.sum(conj(gradient).traj_inner(gradient))))
                traces["iteration"].append(currentIteration)
                user_callback(x, currentIteration, psi, traces["residual"][-1], traces["gradient"][-1])
                currentIteration += 1
            return callback
    else:
        def initCallback(currentIteration):
            def callback(x):
                nonlocal currentIteration
                traces["residual"].append(res_func(x))
                traces["iteration"].append(currentIteration)
                user_callback(x, currentIteration, psi, traces["residual"][-1])
                currentIteration += 1
            return callback

    # convert trajectory to vector of optimisation variables
    traj_vec = init_comp_vec(traj)
    traj2vec(traj, traj_vec)

    # perform optimisation
    if use_jac:
        sol = minimize(res_func, traj_vec, jac=jac_func, method=my_method, callback=initCallback(startIteration), options=options)
    else:
        sol = minimize(res_func, traj_vec, method=my_method, callback=initCallback(startIteration), options=options)

    # unpack trajectory from solution
    op_traj = np.zeros_like(traj)
    op_vec = sol.x
    vec2traj(op_traj, op_vec)

    # convert to full space if singular matrix is provided
    if psi is not None:
        op_traj = op_traj.matmul_left_traj(psi)

    return op_traj, traces, sol
