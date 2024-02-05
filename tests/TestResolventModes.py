# This file contains the unit tests for the SVD functions of the resolvent
# (inverse).

import unittest
import random as rand

import numpy as np

import pyReSolver

from pyReSolver.trajectory_functions import transpose, conj

class TestResolventModes(unittest.TestCase):

    def setUp(self):
        self.no_modes = rand.randint(2, 50)
        self.dim = rand.randint(2, 5)
        self.array = pyReSolver.Trajectory(np.random.rand(self.no_modes, self.dim, self.dim) + 1j*np.random.rand(self.no_modes, self.dim, self.dim))
        self.sys = pyReSolver.systems.lorenz

    def tearDown(self):
        del self.no_modes
        del self.dim
        del self.array
        del self.sys

    def test_resolvent_inv(self):
            rho = rand.uniform(0, 30)
            beta = rand.uniform(0, 10)
            sigma = rand.uniform(0, 30)
            z_mean = rand.uniform(0, 50)
            freq = rand.uniform(0, 10)
            self.sys.parameters['sigma'] = sigma
            self.sys.parameters['beta'] = beta
            self.sys.parameters['rho'] = rho
            jac_at_mean_sys3 = self.sys.jacobian(np.array([[0, 0, z_mean]]))
            H_sys3 = pyReSolver.resolvent_inv(self.array.shape[0], freq, jac_at_mean_sys3)
            resolvent_true = pyReSolver.Trajectory(np.zeros([self.array.shape[0], 3, 3], dtype = complex))
            for n in range(1, self.array.shape[0]):
                D_n = ((1j*n*freq) + sigma)*((1j*n*freq) + 1) + sigma*(z_mean - rho)
                resolvent_true[n, 0, 0] = ((1j*n*freq) + 1)/D_n
                resolvent_true[n, 1, 0] = (rho - z_mean)/D_n
                resolvent_true[n, 0, 1] = sigma/D_n
                resolvent_true[n, 1, 1] = ((1j*n*freq) + sigma)/D_n
                resolvent_true[n, 2, 2] = 1/((1j*n*freq) + beta)
                resolvent_true[n] = np.linalg.inv(np.copy(resolvent_true[n]))
            self.assertEqual(H_sys3, resolvent_true)

    def test_resolvent(self):
        # define parameters
        no_modes = rand.randint(2, 50)
        rho = rand.uniform(0, 30)
        beta = rand.uniform(0, 10)
        sigma = rand.uniform(0, 30)
        z_mean = rand.uniform(0, 50)
        freq = rand.uniform(0, 10)
        self.sys.parameters['sigma'] = sigma
        self.sys.parameters['beta'] = beta
        self.sys.parameters['rho'] = rho

        # evaluate resolvent matrices
        jac_at_mean = self.sys.jacobian(np.array([[0, 0, z_mean]]))
        B = np.array([[0, 0], [-1, 0], [0, 1]])
        H = pyReSolver.resolvent(freq, range(no_modes), jac_at_mean, B = B)

        # true value
        resolvent_true = pyReSolver.Trajectory(np.zeros([no_modes, 3, 2], dtype = complex))
        for n in range(no_modes):
            D_n = ((1j*n*freq) + sigma)*((1j*n*freq) + 1) + sigma*(z_mean - rho)
            resolvent_true[n, 0, 0] = -sigma/D_n
            resolvent_true[n, 1, 0] = -((1j*n*freq) + sigma)/D_n
            resolvent_true[n, 2, 1] = 1/((1j*n*freq) + beta)

        # correct values
        self.assertEqual(H, resolvent_true)

    def test_resolvent_svd_lorenz(self):
        # define parameters
        no_modes = rand.randint(2, 50)
        # rho = rand.uniform(0, 30)
        # beta = rand.uniform(0, 10)
        # sigma = rand.uniform(0, 30)
        # z_mean = rand.uniform(0, 50)
        rho = 28
        beta = 8/3
        sigma = 10
        z_mean = 26.58
        freq = rand.uniform(0, 10)
        self.sys.parameters['sigma'] = sigma
        self.sys.parameters['beta'] = beta
        self.sys.parameters['rho'] = rho

        # evaluate resolvent matrices
        jac_at_mean = self.sys.jacobian(np.array([[0, 0, z_mean]]))
        B = np.array([[0, 0], [-1, 0], [0, 1]])
        H = pyReSolver.resolvent(freq, range(no_modes), jac_at_mean, B = B)

        # perform singular value decomposition
        psi, sig, phi = pyReSolver.resolvent_modes(H)

        # generate true singular modes values
        psi_true = np.zeros_like(psi)
        sig_true = np.zeros_like(sig)
        phi_true = np.zeros_like(phi)
        for n in range(H.shape[0]):
            D_n = ((1j*n*freq) + sigma)*((1j*n*freq) + 1) + sigma*(z_mean - rho)
            alpha_n = -sigma/D_n
            beta_n = -((1j*n*freq) + sigma)/D_n
            gamma_n = 1/((1j*n*freq) + beta)
            sing1 = np.sqrt(1/(((n*freq)**2) + (beta ** 2)))
            sing2 = np.sqrt((((n*freq)**2) + (2*(sigma**2)))/(abs(D_n)**2))
            if sing1 > sing2:
                sig_true[n, 0, 0] = sing1
                sig_true[n, 1, 1] = sing2
            else:
                sig_true[n, 0, 0] = sing2
                sig_true[n, 1, 1] = sing1
            psi_true[n, 0, 0] = -alpha_n/sig_true[n, 0, 0]
            psi_true[n, 1, 0] = -beta_n/sig_true[n, 0, 0]
            psi_true[n, 2, 1] = -gamma_n/sig_true[n, 1, 1]
            phi_true[n, 0, 0] = -1
            phi_true[n, 1, 1] = -1

        # correct values
        self.assertEqual(psi, psi_true)
        self.assertEqual(sig, sig_true)
        self.assertEqual(phi, phi_true)

        # reconstruct resolvend and its inverse

    def test_resolvent_svd_random(self):
        # perform decomposition
        psi, sig, phi = pyReSolver.resolvent_modes(self.array)

        # check singular values are in correct order
        for i in range(self.no_modes):
            for j in range(self.dim - 1):
                self.assertTrue(sig[i, j, j] >= sig[i, j + 1, j + 1])

        # take inverse of array at each mode
        array_inv_true = np.zeros_like(self.array)
        for i in range(self.no_modes):
            array_inv_true[i] = np.linalg.inv(self.array[i])

        # invert singular values matrix
        sig_inv = np.zeros_like(sig)
        for i in range(self.no_modes):
            sig_inv[i] = np.linalg.inv(sig[i])

        def traj_matmul_tmp(traj1, traj2):
            return np.einsum('ikl,ilj->ikj', traj1, traj2, out=None)

        # multiply singular matrices together to get array and its inverse
        array_recon = traj_matmul_tmp(psi, traj_matmul_tmp(sig, transpose(conj(phi))))
        array_inv_recon = traj_matmul_tmp(phi, traj_matmul_tmp(sig_inv, transpose(conj(psi))))

        # compare arrays and reconstructed arrays
        self.assertEqual(self.array, array_recon)

        # do the same for the inverse arrays
        self.assertEqual(array_inv_true, array_inv_recon)

    def test_resolvent_modes_truncated(self):
        # perform truncated svd
        cut = rand.randint(0, self.dim - 1)
        psi, sig, phi = pyReSolver.resolvent_modes(self.array, cut = cut)

        # check correct size
        self.assertEqual(psi.shape, (self.no_modes, self.dim, self.dim - cut))
        self.assertEqual(sig.shape, (self.no_modes, self.dim - cut, self.dim - cut))
        self.assertEqual(phi.shape, (self.no_modes, self.dim, self.dim - cut))


if __name__ == '__main__':
    unittest.main()
