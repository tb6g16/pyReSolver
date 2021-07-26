# This file contains the unit tests for the SVD functions of the resolvent
# (inverse).

import unittest
import random as rand
import numpy as np

from Trajectory import Trajectory
from trajectory_functions import transpose, conj
from resolvent_modes import resolvent, resolvent_modes
from systems import lorenz

class TestResolventModes(unittest.TestCase):

    def setUp(self):
        # self.no_modes = rand.randint(2, 50)
        # self.dim = rand.randint(2, 5)
        self.no_modes = 5
        self.dim = 2
        self.array = Trajectory(np.random.rand(self.no_modes, self.dim, self.dim) + 1j*np.random.rand(self.no_modes, self.dim, self.dim))
        self.sys = lorenz

    def tearDown(self):
        del self.no_modes
        del self.dim
        del self.array
        del self.sys

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
        jac_at_mean = self.sys.jacobian([0, 0, z_mean])
        B = np.array([[0, 0], [-1, 0], [0, 1]])
        H_sys = resolvent(freq, range(no_modes), jac_at_mean, B = B)

        # true value
        resolvent_true = Trajectory(np.zeros([no_modes, 3, 2], dtype = complex))
        for n in range(no_modes):
            D_n = ((1j*n*freq) + sigma)*((1j*n*freq) + 1) + sigma*(z_mean - rho)
            resolvent_true[n, 0, 0] = -sigma/D_n
            resolvent_true[n, 1, 0] = -((1j*n*freq) + sigma)/D_n
            resolvent_true[n, 2, 1] = 1/((1j*n*freq) + beta)

        # correct values
        self.assertEqual(H_sys, resolvent_true)

    def est_resolvent_modes_full(self):
        # perform decomposition
        psi, sig, phi = resolvent_modes(self.array)

        # check singular values are in correct order
        for i in range(self.no_modes):
            for j in range(self.dim - 1):
                self.assertTrue(sig[i, j, j] >= sig[i, j + 1, j + 1])

        # take inverse of array at each mode
        array_inv_true = Trajectory(np.zeros_like(self.array.modes))
        for i in range(1, self.no_modes):
            array_inv_true[i] = np.linalg.inv(self.array[i])

        # multiply singular matrices together to get array inverse
        # array_inv = psi @ sig @ transpose(conj(phi))
        a = transpose(conj(phi))
        b = a.matmul_left_traj(sig)
        array_inv = b.matmul_left_traj(psi)

        # compare to see if they are the same
        # self.assertAlmostEqual(array_inv, array_inv_true)
        for i in range(1, self.no_modes):
            self.assertTrue(np.allclose(array_inv[i], array_inv_true[i]))

        # is zero mode from reconstruction zero matrix
        self.assertTrue(np.array_equal(np.zeros_like(self.array[0]), array_inv[0]))

    def est_resolvent_modes_truncated(self):
        # perform truncated svd
        cut = rand.randint(0, self.dim - 1)
        psi, sig, phi = resolvent_modes(self.array, cut = cut)

        # check correct size
        self.assertEqual(psi.shape, (self.no_modes, self.dim, self.dim - cut))
        self.assertEqual(sig.shape, (self.no_modes, self.dim - cut, self.dim - cut))
        self.assertEqual(phi.shape, (self.no_modes, self.dim, self.dim - cut))


if __name__ == '__main__':
    unittest.main()
