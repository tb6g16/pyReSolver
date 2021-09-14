# This file contains the unit tests for traj2vec and vec2traj functions
# required to perform the optimisation.

import unittest
import random as rand
import numpy as np

from ResolventSolver.Trajectory import Trajectory
import ResolventSolver.traj2vec as t2v

class TestTraj2Vec(unittest.TestCase):
    
    def setUp(self):
        modes = rand.randint(1, 256)
        dim = rand.randint(1, 5)
        traj_array = np.random.rand(modes, dim) + 1j*np.random.rand(modes, dim)
        traj_array[0] = 0
        traj_array[-1] = 0
        self.traj = Trajectory(traj_array)
        self.freq = rand.uniform(-10, 10)
        self.vec = t2v.traj2vec(self.traj, self.freq)

    def tearDown(self):
        del self.traj
        del self.freq
        del self.vec

    def test_traj2vec(self):
        # correct size
        dofs = (2*self.traj.shape[1]*(self.traj.shape[0] - 2)) + 1
        self.assertEqual(np.shape(self.vec), (dofs,))

        # correct values
        a = 0
        b = (self.traj.shape[0] - 2)*self.traj.shape[1]
        for i in range(self.traj.shape[0] - 2):
            for j in range(self.traj.shape[1]):
                self.assertEqual(self.vec[a], self.traj[i + 1, j].real)
                self.assertEqual(self.vec[b], self.traj[i + 1, j].imag)
                a += 1
                b += 1
        self.assertEqual(self.vec[-1], self.freq)

    def test_vec2traj(self):
        traj, freq = t2v.vec2traj(self.vec, self.traj.shape[1])

        # check vec2traj returns correct trajectory
        self.assertEqual(traj, self.traj)

        # check vec2traj returns correct frequency
        self.assertEqual(freq, self.freq)


if __name__ == "__main__":
    unittest.main()
