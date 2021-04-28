# This file contains the definitions required to compute the singular value
# decomposition of the resolvent operator.

import numpy as np
from traj_util import list2array, array2list
from Trajectory import Trajectory
from trajectory_functions import transpose, conj

def resolvent_modes_full(res_inv):
    # perform full svd
    phi, sig, psi = np.linalg.svd(list2array(res_inv.mode_list))

    # set first element of phi and psi to zero matrices
    phi[0] = np.zeros_like(phi[0])
    psi[0] = np.zeros_like(psi[0])

    # take inverse of singular values and diagonalise
    sig[1:] = 1.0/sig[1:]
    sig = [np.diag(sig[i, :]) for i in range(np.shape(sig)[0])]

    # convert to instances of trajectory
    phi = Trajectory(array2list(phi))
    sig = Trajectory(sig)
    psi = Trajectory(array2list(psi))

    # loop over swapping rows and columns as necessary
    for i in range(res_inv.shape[0]):
        phi[i] = np.flip(phi[i], axis = 1)
        sig[i] = np.flip(sig[i])
        psi[i] = np.flip(psi[i], axis = 0)

    # take conjugate transpose of left singular matrix (psi)
    psi = transpose(conj(psi))

    return phi, sig, psi

def resolvent_modes_reduced(resolvent):
    pass

if __name__ == '__main__':
    import random as rand
    from residual_functions import resolvent_inv
    from systems import lorenz

    mean = [0, 0, 0]
    jac_at_mean = lorenz.jacobian(mean)

    no_modes = rand.randint(2, 50)
    res_inv = resolvent_inv(no_modes, 1, jac_at_mean)

    phi, sig, psi = resolvent_modes_full(res_inv)

    for i in range(1, no_modes):
        if not np.allclose(np.linalg.inv(res_inv[i]), psi[i] @ sig[i] @ np.transpose(np.conj(phi[i]))):
            print(False)
