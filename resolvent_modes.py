# This file contains the definitions required to compute the singular value
# decomposition of the resolvent operator.

import numpy as np
from traj_util import list2array, array2list
from Trajectory import Trajectory
from trajectory_functions import transpose, conj

def resolvent(freq, n, jac_at_mean, B = None):
    # evaluate the number of dimensions using the size of the jacobian
    dim = np.shape(jac_at_mean)[0]

    # is B given as an input
    if type(B) != np.ndarray:
        B = np.eye(dim)

    # calculate single resolvent matrix if n is an integer
    if type(n) == int:
        H_n = np.linalg.inv(1j*n*freq*np.eye(dim) - jac_at_mean) @ B
    elif type(n) == range:
        H_n = [None]*(n[-1] + 1)
        for i in n:
            H_n[i] = np.linalg.inv(1j*i*freq*np.eye(dim) - jac_at_mean) @ B
        H_n = Trajectory(H_n)

    return H_n

def resolvent_modes(resolvent, cut = 0):
    # perform full svd
    psi, sig, phi = np.linalg.svd(list2array(resolvent.mode_list), full_matrices = False)

    # diagonalize singular value matrix and convert all to lists
    sig = [np.diag(sig[i, :]) for i in range(resolvent.shape[0])]
    psi = array2list(psi)
    phi = array2list(phi)

    # cut off the desired number of singular values
    if cut != 0:
        for i in range(resolvent.shape[0]):
            sig[i] = sig[i][:-cut, :-cut]
            psi[i] = psi[i][:, :-cut]
            phi[i] = phi[i][:-cut, :]

    return Trajectory(psi), Trajectory(sig), conj(transpose(Trajectory(phi)))

def resolvent_modes_inv(res_inv, cut = 0):
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

if __name__ == '__main__':
    from systems import lorenz

    freq = 1
    n = range(5)
    jac_at_mean = lorenz.jacobian([0, 0, 0])
    B = np.array([[0, 0], [-1, 0], [0, 1]])

    H_n = resolvent(freq, n, jac_at_mean, B)

    cut = 1
    psi_n, sig_n, phi_n = resolvent_modes(H_n, cut = cut)

    # for i in n:
    #     print(np.allclose(H_n[i], psi_n[i] @ sig_n[i] @ np.conj(np.transpose(phi_n[i]))))

    a = 1
    print(H_n[a])
    print(psi_n[a] @ sig_n[a] @ np.conj(np.transpose(phi_n[a])))
