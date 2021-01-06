# This file contains the unit tests for traj2vec and vec2traj functions
# required to perform the optimisation.

import sys
sys.path.append(r"C:\Users\user\Desktop\PhD\Bruno Paper\Code\RA Dynamical System")
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
        self.traj1 = Trajectory(uc.x)
        self.traj2 = Trajectory(elps.x)
        self.freq1 = rand.uniform(-10, 10)
        self.freq2 = rand.uniform(-10, 10)
        self.t1f1_vec = t2v.traj2vec(self.traj1, self.freq1)
        self.t1f2_vec = t2v.traj2vec(self.traj1, self.freq2)
        self.t2f1_vec = t2v.traj2vec(self.traj2, self.freq1)
        self.t2f2_vec = t2v.traj2vec(self.traj2, self.freq2)

    def tearDown(self):
        del self.traj1
        del self.traj2
        del self.freq1
        del self.freq2
        del self.t1f1_vec
        del self.t1f2_vec
        del self.t2f1_vec
        del self.t2f2_vec

    def test_traj2vec(self):
        # correct size
        dofs = (self.traj1.shape[0]*self.traj1.shape[1]) + 1
        self.assertEqual(np.shape(self.t1f1_vec), (dofs,))
        self.assertEqual(np.shape(self.t1f2_vec), (dofs,))
        self.assertEqual(np.shape(self.t2f1_vec), (dofs,))
        self.assertEqual(np.shape(self.t2f2_vec), (dofs,))

        # last element always frequency
        self.assertEqual(self.t1f1_vec[-1], self.freq1)
        self.assertEqual(self.t1f2_vec[-1], self.freq2)
        self.assertEqual(self.t2f1_vec[-1], self.freq1)
        self.assertEqual(self.t2f2_vec[-1], self.freq2)

        # correct values
        t1f1_vec_true = np.zeros([dofs])
        t1f2_vec_true = np.zeros([dofs])
        t2f1_vec_true = np.zeros([dofs])
        t2f2_vec_true = np.zeros([dofs])
        for i in range(dofs - 1):
            if i % 2 == 0:
                t1f1_vec_true[i] = self.traj1[0, i//2]
                t1f2_vec_true[i] = self.traj1[0, i//2]
                t2f1_vec_true[i] = self.traj2[0, i//2]
                t2f2_vec_true[i] = self.traj2[0, i//2]
            else:
                t1f1_vec_true[i] = self.traj1[1, i//2]
                t1f2_vec_true[i] = self.traj1[1, i//2]
                t2f1_vec_true[i] = self.traj2[1, i//2]
                t2f2_vec_true[i] = self.traj2[1, i//2]
        t1f1_vec_true[-1] = self.freq1
        t1f2_vec_true[-1] = self.freq2
        t2f1_vec_true[-1] = self.freq1
        t2f2_vec_true[-1] = self.freq2
        
        self.assertTrue(np.array_equal(self.t1f1_vec, t1f1_vec_true))
        self.assertTrue(np.array_equal(self.t1f2_vec, t1f2_vec_true))
        self.assertTrue(np.array_equal(self.t2f1_vec, t2f1_vec_true))
        self.assertTrue(np.array_equal(self.t2f2_vec, t2f2_vec_true))

    def test_vec2traj(self):
        traj_t1f1, freq_t1f1 = t2v.vec2traj(self.t1f1_vec, 2)
        traj_t1f2, freq_t1f2 = t2v.vec2traj(self.t1f2_vec, 2)
        traj_t2f1, freq_t2f1 = t2v.vec2traj(self.t2f1_vec, 2)
        traj_t2f2, freq_t2f2 = t2v.vec2traj(self.t2f2_vec, 2)

        # check vec2traj returns correct trajectory
        self.assertEqual(traj_t1f1, self.traj1)
        self.assertEqual(traj_t1f2, self.traj1)
        self.assertEqual(traj_t2f1, self.traj2)
        self.assertEqual(traj_t2f2, self.traj2)

        # check vec2traj returns correct frequency
        self.assertEqual(freq_t1f1, self.freq1)
        self.assertEqual(freq_t1f2, self.freq2)
        self.assertEqual(freq_t2f1, self.freq1)
        self.assertEqual(freq_t2f2, self.freq2)


if __name__ == "__main__":
    unittest.main()