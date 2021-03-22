# This file contains the function definitions that will optimise a given
# trajectory and fundamental frequency to find the lowest global residual for
# a given dynamical system.

import numpy as np
from scipy.optimize import minimize
import scipy.integrate as integ
from Trajectory import Trajectory
from System import System
from traj2vec import traj2vec, vec2traj
import trajectory_functions as traj_funcs
import residual_functions as res_funcs

def init_opt_funcs(sys, dim, mean):
    """
        This functions initialises the optimisation vectors for a specific
        system.
    """
    def traj_global_res(opt_vector):
        """
            This function calculates the global residual for a given vector
            that defines a trajectory frequency pair, for the purpose of
            optimisation.
        """
        # unpack trajectory
        traj, freq = vec2traj(opt_vector, dim)

        # calculate global residual and return
        return res_funcs.global_residual(traj, sys, freq, mean)

    def traj_global_res_jac(opt_vector):
        """
            This function calculates the gradient of the global residual for a
            given vector that defines a trajectory frequency pair, for the
            purpose of optimisation.
        """
        # unpack trajectory
        traj, freq = vec2traj(opt_vector, dim)

        # calculate global residual gradients
        gr_traj_grad = res_funcs.gr_traj_grad(traj, sys, freq, mean)
        gr_freq_grad = res_funcs.gr_freq_grad(traj, sys, freq, mean)

        # convert back to vector and return
        return traj2vec(gr_traj_grad, gr_freq_grad)
    
    return traj_global_res, traj_global_res_jac

def my_min(traj, freq, sys, mean, **kwargs):
    # unpack keyword arguments
    my_method = kwargs.get('method', 'L-BFGS-B')
    if_quiet = kwargs.get('quiet', False)
    maxiter = kwargs.get('iter', None)
    traces = kwargs.get('traces', None)

    # setup the problem
    dim = traj.shape[1]
    res_func, jac_func = init_opt_funcs(sys, dim, mean)

    # define varaibles to be tracked using callback
    if traces == None:
        traces = {'traj': [], 'freq': [], 'lr': [], 'gr': [], 'gr_grad': []}

    # define callback function
    def callback(x):
        cur_traj, cur_freq = vec2traj(x, dim)

        traces['traj'].append(cur_traj)
        traces['freq'].append(cur_freq)
        traces['lr'].append(res_funcs.local_residual(cur_traj, sys, cur_freq, mean))
        traces['gr'].append(res_func(x))
        traces['gr_grad'].append(jac_func(x))

    # convert trajectory to vector of optimisation variables
    traj_vec = traj2vec(traj, freq)

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
    sol = minimize(res_func, traj_vec, jac = jac_func, method = my_method, callback = callback, options = options)

    # unpack trajectory from solution
    op_vec = sol.x
    op_traj, op_freq = vec2traj(op_vec, dim)

    return op_traj, op_freq, traces, sol

if __name__ == "__main__":
    from trajectory_definitions import unit_circle as uc
    from systems import van_der_pol as vpd

    sys = System(vpd)
    sys.parameters['mu'] = 2
    init_traj = 2*Trajectory(uc.x, modes = 33)
    init_freq = 1
    mean = [0, 0]

    op_traj, op_freq, traces, sol = my_min(init_traj, init_freq, sys, mean)

    print(sol.message)
    print("Number of iterations: " + str(sol.nit))

    print("Period of orbit: " + str((2*np.pi)/op_freq))
    print("Global residual before: " + str(res_funcs.global_residual(init_traj, sys, init_freq, mean)))
    print("Global residual after: " + str(res_funcs.global_residual(op_traj, sys, op_freq, mean)))
    op_traj.plot(aspect = 1)
