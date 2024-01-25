# This file contains the unit tests for traj2vec and vec2traj functions
# required to perform the optimisation.

import unittest
import random as rand

import numpy as np

from pyReSolver.Trajectory import Trajectory
import pyReSolvertraj2vec as t2v

class TestTraj2Vec(unittest.TestCase):
    
    def setUp(self):
        modes = rand.randint(1, 256)
        dim = rand.randint(1, 5)
        traj_array = np.random.rand(modes, dim) + 1j*np.random.rand(modes, dim)
        traj_array[0] = 0
        self.traj = Trajectory(traj_array)
        self.vec = t2v.init_comp_vec(self.traj)
        t2v.traj2vec(self.traj, self.vec)

    def tearDown(self):
        del self.traj
        del self.vec

    def test_traj2vec(self):
        # correct values
        a = 0
        b = (self.traj.shape[0] - 1)*self.traj.shape[1]
        for i in range(self.traj.shape[0] - 1):
            for j in range(self.traj.shape[1]):
                self.assertEqual(self.vec[a], self.traj[i + 1, j].real)
                self.assertEqual(self.vec[b], self.traj[i + 1, j].imag)
                a += 1
                b += 1

    def test_vec2traj(self):
        # initialise temporary trajectory
        tmp_traj = np.zeros_like(self.traj)

        # convert from vector to trajectory
        t2v.vec2traj(tmp_traj, self.vec)

        # check vec2traj returns correct trajectory
        self.assertEqual(tmp_traj, self.traj)


if __name__ == "__main__":
    unittest.main()
