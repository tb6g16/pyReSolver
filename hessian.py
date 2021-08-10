# This file contains the definitions required to estimate the Hessian matrix
# of the residual function near the minimum.

import numpy as np

from traj2vec import traj2vec
from my_min import init_opt_funcs

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
        if j == 0:
            print(grad_off_min)

        # evaluate hessian column
        hessian[:, j] = (grad_at_min - grad_off_min)/eps

    return hessian

if __name__ == "__main__":
    import h5py
    from unpack_hdf5 import unpack_hdf5
    from change_traj_res import change_traj_res
    from System import System
    from systems import lorenz
    import random as rand
    import time
    import scipy

    traj, freq, mean, _ = unpack_hdf5(r'C:\Users\user\Desktop\PhD\Bruno Paper\Analysis\Example UPOs\upo01.orb')
    traj = change_traj_res(traj, 25)
    traj_vec = traj2vec(traj, freq)
    sys = System(lorenz)

    eps = 1e-6

    upo_hess = hess(traj, freq, sys, mean, eps = eps)

    hess_eigvals = scipy.linalg.eigvals(upo_hess)

    # for i in range(np.shape(hess_eigvals)[0]):
    #     if np.real(hess_eigvals[i]) > 0:
    #         print()
    #         print("Positive:   " + str(np.real(hess_eigvals[i]) > 0))
    #         print("Eigenvalue: " + str(np.real(hess_eigvals[i])))
    #         time.sleep(0.01)

    # ONLY SYMMETRIC TO LOW A TOLERANCE, GETS WORSE AS NUMBER OF MODES IS INCREASED
    # print()
    # print("Hessian symmetric: " + str(np.allclose(upo_hess, np.transpose(upo_hess), atol = 1e-2)))
