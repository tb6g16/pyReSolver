# This file contains the unit tests for the functions defined in traj_util.py.

import unittest
import random as rand

import numpy as np

from pyReSolver.traj_util import func2curve

from tests.test_trajectories import unit_circle as uc
from tests.test_trajectories import unit_circle_3d as uc3d
from tests.test_trajectories import ellipse as elps

class TestTrajUtil(unittest.TestCase):

    def setUp(self):
        self.rand1 = rand.randint(2, 50)
        self.rand2 = rand.randint(2, 50)
        self.rand3 = rand.randint(2, 50)
        self.array1 = func2curve(uc, modes = self.rand1)
        self.array2 = func2curve(elps, modes = self.rand2)
        self.array3 = func2curve(uc3d, modes = self.rand3)

    def tearDown(self):
        del self.rand1
        del self.rand2
        del self.rand3
        del self.array1
        del self.array2
        del self.array3

    def test_func2curve_traj(self):
        # correct shape
        self.assertEqual(self.array1.shape, (2*(self.rand1 - 1), 2))
        self.assertEqual(self.array2.shape, (2*(self.rand2 - 1), 2))
        self.assertEqual(self.array3.shape, (2*(self.rand3 - 1), 3))

        # correct values
        for i in range(2*(self.rand1 - 1)):
            s = (2*np.pi)/(2*(self.rand1 - 1))*i
            self.assertTrue(np.allclose(self.array1[i], uc(s)))
        for i in range(2*(self.rand2 - 1)):
            s = (2*np.pi)/(2*(self.rand2 - 1))*i
            self.assertTrue(np.allclose(self.array2[i], elps(s)))
        for i in range(2*(self.rand3 - 1)):
            s = (2*np.pi)/(2*(self.rand3 - 1))*i
            self.assertTrue(np.allclose(self.array3[i], uc3d(s)))


if __name__ == "__main__":
    unittest.main()
