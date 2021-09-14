# This file contains the definitions required to estimate the Hessian matrix
# of the residual function near the minimum.

import numpy as np

from ResolventSolver.traj2vec import traj2vec
from ResolventSolver.my_min import init_opt_funcs

def hess(traj, freq, sys, mean, eps = 1e-6, conv_method = 'fft'):
    # initialise gradient function
    _, grad_func = init_opt_funcs(sys, traj.shape[1], mean, conv_method = conv_method)

    # convert trajectory to state vector
    state = traj2vec(traj, freq)

    # intialise hessian matrix
    hessian = np.zeros([np.shape(state)[0], np.shape(state)[0]])

    # precompute the gradient for the given trajectory
    grad_at_min = grad_func(state)

    # loop over columns of hessian matrix
    for j in range(np.shape(state)[0]):
        # define unit basis in j-th direction
        unit_basis_j = np.zeros(np.shape(state)[0])
        unit_basis_j[j] = 1.0

        # calculate gradient at two close states at minimum
        grad_off_min = grad_func(state + eps*unit_basis_j)

        # evaluate hessian column
        hessian[:, j] = (grad_at_min - grad_off_min)/eps

    return hessian[:-1, :-1]
