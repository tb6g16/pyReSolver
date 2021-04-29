# This file contains the definitions required to compute the singular value
# decomposition of the resolvent operator.

import numpy as np
from traj_util import list2array, array2list
from Trajectory import Trajectory
from trajectory_functions import transpose, conj

def resolvent_modes(res_inv, cut = 0):
    # perform full svd
    phi, sig, psi = np.linalg.svd(list2array(res_inv.mode_list))

    # set first element of phi and psi to zero matrices
    psi[0] = np.zeros_like(psi[0])
    phi[0] = np.zeros_like(phi[0])

    # take inverse of singular values and diagonalise
    sig[1:] = 1.0/sig[1:]

    # diagonalize singular value matrix and convert all to lists
    sig = [np.diag(sig[i, :]) for i in range(res_inv.shape[0])]
    psi = array2list(psi)
    phi = array2list(phi)

    # loop over swapping rows and columns as necessary
    for i in range(1, res_inv.shape[0]):
        sig[i] = np.flip(sig[i])
        psi[i] = np.transpose(np.conj(np.flip(psi[i], axis = 0)))
        phi[i] = np.flip(phi[i], axis = 1)

    # cut off the desired number of singular values
    if cut != 0:
        for i in range(res_inv.shape[0]):
            sig[i] = sig[i][:-cut, :-cut]
            psi[i] = psi[i][:, :-cut]
            phi[i] = phi[i][:, :-cut]

    # initialise as trajectory instances and return
    return Trajectory(psi), Trajectory(sig), Trajectory(phi)
