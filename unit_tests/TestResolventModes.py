# This file contains the unit tests for the SVD functions of the resolvent
# (inverse).

import sys
sys.path.append(r"C:\Users\user\Desktop\PhD\Bruno Paper\ResolventSolver")
import unittest
import random as rand
import numpy as np
from Trajectory import Trajectory
from trajectory_functions import transpose, conj
from resolvent_modes import resolvent_modes

class TestResolventModes(unittest.TestCase):

    def setUp(self):
        self.no_modes = rand.randint(2, 50)
        self.dim = rand.randint(2, 5)
        array = [None]*self.no_modes
        for i in range(self.no_modes):
            array[i] = np.random.rand(self.dim, self.dim) + 1j*np.random.rand(self.dim, self.dim)
        self.array = Trajectory(array)

    def tearDown(self):
        del self.no_modes
        del self.dim
        del self.array

    def test_resolvent_modes_full(self):
        # perform decomposition
        psi, sig, phi = resolvent_modes(self.array)

        # check singular values are in correct order
        for i in range(self.no_modes):
            for j in range(self.dim - 1):
                self.assertTrue(sig[i, j, j] >= sig[i, j + 1, j + 1])

        # take inverse of array at each mode
        array_inv_true = [None]*self.no_modes
        for i in range(self.no_modes):
            array_inv_true[i] = np.linalg.inv(self.array[i])
        array_inv_true = Trajectory(array_inv_true)

        # multiply singular matrices together to get array inverse
        array_inv = psi @ sig @ transpose(conj(phi))

        # compare to see if they are the same
        for i in range(1, self.no_modes):
            self.assertTrue(np.allclose(array_inv[i], array_inv_true[i]))
        
        # is zero mode from reconstruction zero matrix
        self.assertTrue(np.array_equal(np.zeros_like(self.array[0]), array_inv[0]))
    
    def test_resolvent_modes_truncated(self):
        # perform truncated svd
        cut = rand.randint(0, self.dim - 1)
        psi, sig, phi = resolvent_modes(self.array, cut = cut)

        # check correct size
        self.assertEqual(psi.shape, (self.no_modes, self.dim, self.dim - cut))
        self.assertEqual(sig.shape, (self.no_modes, self.dim - cut, self.dim - cut))
        self.assertEqual(phi.shape, (self.no_modes, self.dim, self.dim - cut))


if __name__ == '__main__':
    unittest.main()
