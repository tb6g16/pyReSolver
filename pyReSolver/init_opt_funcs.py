# This file contains the function definitions required to initialise and
# calculate the residual and associated gradient for optimisation of a
# trajectory for a given system.

import numpy as np

from pyReSolver.Cache import Cache
from pyReSolver.resolvent_modes import resolvent_inv
import pyReSolverresidual_functions as res_funcs
from pyReSolver.trajectory_functions import transpose, conj
from pyReSolver.traj2vec import traj2vec, vec2traj

def init_opt_funcs(cache, freq, fftplans, sys, mean, psi = None):
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
    # initialise stuff
    H_n_inv = resolvent_inv(cache.traj.shape[0], freq, sys.jacobian(mean))

    if psi is not None:
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
            vec2traj(cache.red_traj, opt_vector)

            # convert to full space if singular matrix is provided
            np.copyto(cache.traj, cache.red_traj.matmul_left_traj(psi))

            # calculate global residual and return
            res_funcs.local_residual(cache, sys, H_n_inv, fftplans)
            return res_funcs.global_residual(cache)

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
            vec2traj(cache.red_traj, opt_vector)

            # convert to full space if singular matrix is provided
            np.copyto(cache.traj, cache.red_traj.matmul_left_traj(psi))

            # calculate global residual gradients
            gr_traj_grad = res_funcs.gr_traj_grad(cache, sys, freq, mean, fftplans)

            # convert gradient w.r.t modes to reduced space
            gr_traj_grad = gr_traj_grad.matmul_left_traj(transpose(conj(psi)))

            # convert back to vector and return
            traj2vec(gr_traj_grad, opt_vector)

            return opt_vector
    
    else:
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
            vec2traj(cache.traj, opt_vector)

            # calculate global residual and return
            res_funcs.local_residual(cache, sys, H_n_inv, fftplans)
            return res_funcs.global_residual(cache)

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
            vec2traj(cache.traj, opt_vector)

            # calculate global residual gradients
            gr_traj_grad = res_funcs.gr_traj_grad(cache, sys, freq, mean, fftplans)

            # convert back to vector and return
            traj2vec(gr_traj_grad, opt_vector)

            return opt_vector

    return traj_global_res, traj_global_res_jac