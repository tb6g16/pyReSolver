# This file contains the unit tests for the functions defined in traj_util.py.

import sys
sys.path.append(r"C:\Users\user\Desktop\PhD\Bruno Paper\Code\ResolventSolver")
import unittest
import numpy as np
import random as rand
from traj_util import func2curve, list2array, array2list
from trajectory_definitions import unit_circle as uc
from trajectory_definitions import unit_circle_3d as uc3d
from trajectory_definitions import ellipse as elps
from systems import van_der_pol as vpd

class TestTrajUtil(unittest.TestCase):

    def setUp(self):
        self.rand1 = rand.randint(2, 50)
        self.rand2 = rand.randint(2, 50)
        self.rand3 = rand.randint(2, 50)
        self.array1 = func2curve(uc.x, modes = self.rand1)
        self.array2 = func2curve(elps.x, modes = self.rand2)
        self.array3 = func2curve(uc3d.x, modes = self.rand3)
        self.list1 = array2list(self.array1)
        self.list2 = array2list(self.array2)
        self.list3 = array2list(self.array3)

    def tearDown(self):
        del self.rand1
        del self.rand2
        del self.rand3
        del self.array1
        del self.array2
        del self.array3
        del self.list1
        del self.list2
        del self.list3

    def test_func2curve_traj(self):
        # correct shape
        self.assertEqual(self.array1.shape, (2*(self.rand1 - 1), 2))
        self.assertEqual(self.array2.shape, (2*(self.rand2 - 1), 2))
        self.assertEqual(self.array3.shape, (2*(self.rand3 - 1), 3))

        # correct values
        for i in range(2*(self.rand1 - 1)):
            s = (2*np.pi)/(2*(self.rand1 - 1))*i
            self.assertTrue(np.allclose(self.array1[i], uc.x(s)))
        for i in range(2*(self.rand2 - 1)):
            s = (2*np.pi)/(2*(self.rand2 - 1))*i
            self.assertTrue(np.allclose(self.array2[i], elps.x(s)))
        for i in range(2*(self.rand3 - 1)):
            s = (2*np.pi)/(2*(self.rand3 - 1))*i
            self.assertTrue(np.allclose(self.array3[i], uc3d.x(s)))

    def test_array2list(self):
        # correct shape
        self.assertEqual(len(self.list1), 2*(self.rand1 - 1))
        self.assertEqual(len(self.list2), 2*(self.rand2 - 1))
        self.assertEqual(len(self.list3), 2*(self.rand3 - 1))
        for i in range(2*(self.rand1 - 1)):
            self.assertEqual(self.list1[i].shape, (2,))
        for i in range(2*(self.rand2 - 1)):
            self.assertEqual(self.list2[i].shape, (2,))
        for i in range(2*(self.rand3 - 1)):
            self.assertEqual(self.list3[i].shape, (3,))

        # correct values at each index of the list
        for i in range(2*(self.rand1 - 1)):
            self.assertTrue(np.array_equal(self.list1[i], self.array1[i]))
        for i in range(2*(self.rand2 - 1)):
            self.assertTrue(np.array_equal(self.list2[i], self.array2[i]))
        for i in range(2*(self.rand3 - 1)):
            self.assertTrue(np.array_equal(self.list3[i], self.array3[i]))

    def test_list2array(self):
        # recoveres original array
        self.assertTrue(np.array_equal(self.array1, list2array(self.list1)))
        self.assertTrue(np.array_equal(self.array2, list2array(self.list2)))
        self.assertTrue(np.array_equal(self.array3, list2array(self.list3)))


if __name__ == "__main__":
    unittest.main()