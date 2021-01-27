# This file contains the unit tests for traj2vec and vec2traj functions
# required to perform the optimisation.

import sys
sys.path.append(r"C:\Users\user\Desktop\PhD\Bruno Paper\Code\Approach B")
import unittest
import numpy as np
from Trajectory import Trajectory
import traj2vec as t2v
import random as rand
from test_cases import unit_circle as uc
from test_cases import ellipse as elps
from test_cases import van_der_pol as vpd
from test_cases import viswanath as vis

class TestTraj2Vec(unittest.TestCase):
    
    def setUp(self):
        temp = np.random.rand(rand.randint(1, 5), rand.randint(1, 256))
        temp[:, 0] = 0
        self.traj = Trajectory(temp)
        self.freq = rand.uniform(-10, 10)
        self.vec = t2v.traj2vec(self.traj, self.freq)

    def tearDown(self):
        del self.traj
        del self.freq
        del self.vec

    def test_traj2vec(self):
        # correct size
        dofs = (2*self.traj.shape[0]*(self.traj.shape[1] - 1)) + 1
        self.assertEqual(np.shape(self.vec), (dofs,))

        # last element always frequency
        self.assertEqual(self.vec[-1], self.freq)

        # correct values
        vec_true = np.zeros(dofs)
        a = 2*self.traj.shape[0]
        for i in range(dofs - 1):
            if i % a == 0:
                b = 0
            for j in range(a):
                if (i - j) % a == 0:
                    if i % 2 == 0:
                        vec_true[i] = np.real(self.traj[b, 1 + int((i - j)/a)])
                    elif i % 2 == 1:
                        vec_true[i] = np.imag(self.traj[b, 1 + int((i - j)/a)])
                        b += 1
        vec_true[-1] = self.freq
        self.assertTrue(np.array_equal(self.vec, vec_true))

    def test_vec2traj(self):
        traj, freq = t2v.vec2traj(self.vec, self.traj.shape[0])

        # check vec2traj returns correct trajectory
        self.assertEqual(traj, self.traj)

        # check vec2traj returns correct frequency
        self.assertEqual(freq, self.freq)


if __name__ == "__main__":
    unittest.main()