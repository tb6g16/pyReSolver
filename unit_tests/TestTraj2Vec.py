# This file contains the unit tests for traj2vec and vec2traj functions
# required to perform the optimisation.

import sys
sys.path.append(r"C:\Users\user\Desktop\PhD\Bruno Paper\Code\Approach B")
import unittest
import numpy as np
from Trajectory import Trajectory
import traj2vec as t2v
import random as rand
from traj_util import array2list
from trajectory_definitions import unit_circle as uc
from trajectory_definitions import ellipse as elps
from systems import van_der_pol as vpd
from systems import viswanath as vis

class TestTraj2Vec(unittest.TestCase):
    
    def setUp(self):
        modes = rand.randint(1, 256)
        dim1 = rand.randint(1, 5)
        traj = np.random.rand(modes, dim1) + 1j*np.random.rand(modes, dim1)
        traj[0] = 0
        self.traj = Trajectory(array2list(traj))
        self.freq = rand.uniform(-10, 10)
        self.vec = t2v.traj2vec(self.traj, self.freq)

    def tearDown(self):
        del self.traj
        del self.freq
        del self.vec

    def test_traj2vec(self):
        # correct size
        dofs = (2*self.traj.shape[1]*(self.traj.shape[0] - 1)) + 1
        self.assertEqual(np.shape(self.vec), (dofs,))

        # correct values
        vec_true = np.zeros(dofs)
        a = 2*self.traj.shape[1]
        for i in range(dofs - 1):
            if i % a == 0:
                b = 0
            for j in range(a):
                if (i - j) % a == 0:
                    if i % 2 == 0:
                        vec_true[i] = np.real(self.traj[1 + int((i - j)/a), b])
                    elif i % 2 == 1:
                        vec_true[i] = np.imag(self.traj[1 + int((i - j)/a), b])
                        b += 1
        vec_true[-1] = self.freq
        self.assertTrue(np.array_equal(self.vec, vec_true))

    def test_vec2traj(self):
        traj, freq = t2v.vec2traj(self.vec, self.traj.shape[1])

        # check vec2traj returns correct trajectory
        self.assertEqual(traj, self.traj)

        # check vec2traj returns correct frequency
        self.assertEqual(freq, self.freq)


if __name__ == "__main__":
    unittest.main()