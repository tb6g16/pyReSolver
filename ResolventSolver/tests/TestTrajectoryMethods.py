# This file contains the tests for the Trajectory class and its associated
# methods.

import unittest
import random as rand

import numpy as np

from ResolventSolver.Trajectory import Trajectory

class TestTrajectoryMethods(unittest.TestCase):

    def setUp(self):
        self.modes = rand.randint(2, 50)
        self.dim1 = rand.randint(1, 5)
        self.dim2 = rand.randint(1, 5)
        self.dim3 = rand.randint(1, 5)
        self.rand_array1 = np.random.rand(self.modes, self.dim1) + 1j*np.random.rand(self.modes, self.dim1)
        self.rand_array2 = np.random.rand(self.modes, self.dim2, self.dim3) + 1j*np.random.rand(self.modes, self.dim2, self.dim3)
        self.traj1 = Trajectory(self.rand_array1)
        self.traj2 = Trajectory(self.rand_array2)

    def tearDown(self):
        del self.modes
        del self.dim1
        del self.dim2
        del self.dim3
        del self.rand_array1
        del self.rand_array2
        del self.traj1
        del self.traj2

    def test_add_sub(self):
        rand_factor1 = rand.uniform(0, 10)
        rand_factor2 = rand.uniform(0, 10)
        self.assertEqual(self.traj1 + Trajectory(rand_factor1*self.rand_array1),
                            self.rand_array1 + rand_factor1*self.rand_array1)
        self.assertEqual(self.traj1 + Trajectory(rand_factor2*self.rand_array1),
                            self.rand_array1 + rand_factor2*self.rand_array1)

    def test_mul_float(self):
        # random numbers to multiply by
        rand1 = rand.uniform(-10, 10)
        rand2 = rand.uniform(-10, 10)

        # perform multiplication via magic method
        float_mul1 = rand1*self.traj1
        float_mul2 = rand2*self.traj2

        # correct elements
        for i in range(self.traj1.shape[0]):
                self.assertTrue(np.array_equal(float_mul1[i], self.traj1[i]*rand1))
                self.assertTrue(np.array_equal(float_mul2[i], self.traj2[i]*rand2))
        
        # check commutativity (__rmul__)
        self.assertEqual(float_mul1, self.traj1*rand1)
        self.assertEqual(float_mul2, self.traj2*rand2)

    def test_matmul_traj(self):
        # random trajectory to left multiply
        no_modes = self.traj1.shape[0]
        rand_row = rand.randint(2, 5)
        rand_array = np.random.rand(no_modes, rand_row, self.dim1) + 1j*np.random.rand(no_modes, rand_row, self.dim1)
        rand_traj = Trajectory(rand_array)

        # perform left multiplication
        traj_mult = self.traj1.matmul_left_traj(rand_traj)

        # correct size
        self.assertEqual(traj_mult.shape, (no_modes, rand_row))

        # correct values
        for i in range(no_modes):
            self.assertEqual(traj_mult[i], np.matmul(rand_traj[i], self.traj1[i]))

    def test_getitem(self):
        rand_mode = rand.randint(0, self.modes - 1)
        rand_el1 = rand.randint(0, self.dim1 - 1)
        rand_el2 = rand.randint(0, self.dim2 - 1)
        rand_el3 = rand.randint(0, self.dim3 - 1)
        self.assertEqual(self.traj1[rand_mode, rand_el1], self.rand_array1[rand_mode, rand_el1])
        self.assertEqual(self.traj2[rand_mode, rand_el2, rand_el3], self.rand_array2[rand_mode][rand_el2, rand_el3])
        self.assertTrue(np.array_equal(self.traj1[rand_mode, :], self.rand_array1[rand_mode]))
        self.assertTrue(np.array_equal(self.traj2[rand_mode, rand_el2, :], self.rand_array2[rand_mode][rand_el2, :]))
        self.assertTrue(np.array_equal(self.traj2[rand_mode, :, rand_el3], self.rand_array2[rand_mode][:, rand_el3]))

    def test_round(self):
        rand_mode = rand.randint(0, self.modes - 1)
        rand_el1 = rand.randint(0, self.dim1 - 1)
        rand_el2 = rand.randint(0, self.dim2 - 1)
        rand_el3 = rand.randint(0, self.dim3 - 1)
        rand_round = rand.randint(0, 10)
        traj1_round = round(self.traj1, rand_round)
        rand_array1_round = np.around(self.rand_array1[rand_mode, rand_el1], rand_round)
        traj2_round = round(self.traj2, rand_round)
        rand_array2_round = np.around(self.rand_array2[rand_mode, rand_el2, rand_el3], rand_round)
        self.assertEqual(traj1_round[rand_mode, rand_el1], rand_array1_round)
        self.assertEqual(traj2_round[rand_mode, rand_el2, rand_el3], rand_array2_round)


if __name__ == "__main__":
    unittest.main()
