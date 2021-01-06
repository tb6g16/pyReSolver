# This file contains the function definitions that will optimise a given
# trajectory and fundamental frequency to find the lowest global residual for
# a given dynamical system.

import numpy as np
import scipy.optimize as opt
import scipy.integrate as integ
from Trajectory import Trajectory
from System import System
from traj2vec import traj2vec, vec2traj
import trajectory_functions as traj_funcs
import residual_functions as res_funcs

def init_opt_funcs(sys, dim):
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
        return res_funcs.global_residual(traj, sys, freq)

    def traj_global_res_jac(opt_vector):
        """
            This function calculates the gradient of the global residual for a
            given vector that defines a trajectory frequency pair, for the
            purpose of optimisation.
        """
        # unpack trajectory
        traj, freq = vec2traj(opt_vector, dim)

        # calculate global residual gradients
        gr_traj, gr_freq = res_funcs.global_residual_grad(traj, sys, freq)

        # convert back to vector and return
        return traj2vec(gr_traj, gr_freq)
    
    return traj_global_res, traj_global_res_jac

def init_constraints(sys, dim, mean):
    """
        This function intialises the nonlinear constraints imposed on the
        optimisation, and returns the instance of the NonlinearConstraint class
        that is passed as an argument to the optimisation.
    """
    def constraints(opt_vector):
        """
            This function is the input to the NonlinearConstraint class to
            intialise it.
        """
        # unpack trajectory
        traj, _ = vec2traj(opt_vector, dim)

        # calculate fluctuation trajectory
        fluc = Trajectory(traj.curve_array - mean)

        # evaluate mean constraint
        con1 = traj_funcs.average_over_s(fluc)

        # evaluate nonlinear constraint (RANS)
        nl_fluc = traj_funcs.traj_response(fluc, sys.nl_factor)
        nl_fluc_av = traj_funcs.average_over_s(nl_fluc)
        con2 = np.squeeze(sys.response(mean)) + nl_fluc_av

        # combine and return
        return np.concatenate((con1, con2), axis = 0)

    def constraints_grad(opt_vector):
        """
            This function calculate the gradients of the constraint functionals
            at the given trajectory and frequency (vector).
        """
        # unpack trajectory
        traj, _ = vec2traj(opt_vector, dim)

        # define jacobian matrix
        jac = np.zeros([2*dim, (dim*traj.shape[1]) + 1])

        # first constraint gradients
        for i in range(dim):
            con1i_traj_grad = np.zeros([dim, traj.shape[1]])
            con1i_traj_grad[i, :] = 1/(2*np.pi)
            con1i_grad_vec = traj2vec(con1i_traj_grad, 0)
            jac[i, :] = con1i_grad_vec

        # second constraint gradients
        for i in range(dim):
            con2i_traj_grad_func = sys.nl_con_grads[i]
            con2i_traj_grad = traj_funcs.traj_response(traj, con2i_traj_grad_func)
            con2i_grad_vec = traj2vec(con2i_traj_grad, 0)
            jac[i + dim, :] = con2i_grad_vec
        
        return jac

    return constraints, constraints_grad

if __name__ == "__main__":
    from test_cases import unit_circle as uc
    from test_cases import van_der_pol as vpd
    from test_cases import viswanath as vis

    sys = System(vpd)
    sys.parameters['mu'] = 2
    # sys = System(vis)
    # sys.parameters['mu'] = 1
    circle = 2*Trajectory(uc.x, disc = 128)
    freq = 1
    dim = 2

    res_func, jac_func = init_opt_funcs(sys, dim)
    cons, cons_grad = init_constraints(sys, dim, np.zeros([2, 1]))
    constraint = opt.NonlinearConstraint(cons, np.zeros(2*dim), np.zeros(2*dim), jac = cons_grad)

    op_vec = opt.minimize(res_func, traj2vec(circle, freq), jac = jac_func, method = 'L-BFGS-B')
    # op_vec = opt.minimize(res_func, traj2vec(circle, freq), jac = jac_func, constraints = constraint)

    print(op_vec.message)
    print("Number of iterations: " + str(op_vec.nit))
    op_traj, op_freq = vec2traj(op_vec.x, dim)

    print("Period of orbit: " + str((2*np.pi)/op_freq))
    print("Global residual before: " + str(res_func(traj2vec(circle, freq))))
    print("Global residual after: " + str(res_func(traj2vec(op_traj, op_freq))))
    op_traj.plot(gradient = 16/64)

    # test jacbian is zero also
    # print(jac_func(traj2vec(circle, freq)))
    # print(jac_func(traj2vec(op_traj, op_freq)))
