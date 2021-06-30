# This file contains the tests for the Trajectory class and its associated
# methods.

import sys
sys.path.append(r"C:\Users\user\Desktop\PhD\Bruno Paper\ResolventSolver")
import unittest
import numpy as np
import random as rand
from Trajectory import Trajectory
from traj_util import array2list

class TestTrajectoryMethods(unittest.TestCase):
    
    def setUp(self):
        self.modes = rand.randint(2, 50)
        self.dim1 = rand.randint(1, 5)
        self.dim2 = rand.randint(1, 5)
        self.dim3 = rand.randint(1, 5)
        rand_array1 = np.random.rand(self.modes, self.dim1) + 1j*np.random.rand(self.modes, self.dim1)
        rand_array2 = np.random.rand(self.modes, self.dim2, self.dim3) + 1j*np.random.rand(self.modes, self.dim2, self.dim3)
        self.rand_list1 = array2list(rand_array1)
        self.rand_list2 = array2list(rand_array2)
        self.traj1 = Trajectory(self.rand_list1)
        self.traj2 = Trajectory(self.rand_list2)

    def tearDown(self):
        del self.modes
        del self.dim1
        del self.dim2
        del self.dim3
        del self.rand_list1
        del self.rand_list2
        del self.traj1
        del self.traj2

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

    def test_matmul_array(self):
        # random matrices to multiple by
        rand_row1 = rand.randint(1, 5)
        rand_row2 = rand.randint(1, 5)
        rand_col1 = rand.randint(1, 5)
        rand1 = np.random.rand(rand_row1, self.dim1)
        rand2 = np.random.rand(rand_row2, self.dim2)
        rand3 = np.random.rand(self.dim3, rand_col1)
        rand4 = np.random.rand(self.dim1)

        # perform multiplication via magic method
        mat_mul1 = rand1 @ self.traj1
        mat_mul2 = rand2 @ self.traj2
        mat_mul3 = self.traj2 @ rand3
        mat_mul4 = rand4 @ self.traj1
        mat_mul5 = self.traj1 @ rand4

        # correct size
        self.assertEqual(mat_mul1.shape, (self.modes, rand_row1))
        self.assertEqual(mat_mul2.shape, (self.modes, rand_row2, self.dim3))
        self.assertEqual(mat_mul3.shape, (self.modes, self.dim2, rand_col1))
        self.assertEqual(mat_mul4.shape, (self.modes,))
        self.assertEqual(mat_mul5.shape, (self.modes,))

        # correct elements
        for i in range(self.traj1.shape[0]):
            self.assertTrue(np.allclose(mat_mul1[i], rand1 @ self.traj1[i]))
            self.assertTrue(np.allclose(mat_mul2[i], rand2 @ self.traj2[i]))
            self.assertTrue(np.allclose(mat_mul3[i], self.traj2[i] @ rand3))
            self.assertTrue(np.allclose(mat_mul4[i], rand4 @ self.traj1[i]))
            self.assertTrue(np.allclose(mat_mul5[i], mat_mul4[i]))

    def test_check_type_shape(self):
        # generate lists with all same types
        type_list = [complex, float, int, str, bool]
        rands = [[x(rand.uniform(-10, 10)) for i in range(self.modes)] for x in type_list]
        rands[4][rand.randint(0, 2)] = False
        for i in range(len(rands)):
            self.assertTrue(self.traj1.check_type_shape(rands[i])[0])

        # generate lists with single different type
        for i in range(len(rands)):
            rand_el = rand.randint(0, self.modes - 1)
            for j in range(self.modes):
                if j == rand_el:
                    if i == len(rands) - 1:
                        rands[i][j] = type_list[0](rand.uniform(-10, 10))
                    else:
                        rands[i][j] = type_list[i + 1](rand.uniform(-10, 10))
        for i in range(len(rands)):
            self.assertFalse(self.traj1.check_type_shape(rands[i])[0])

        # generate lists with numpy arrays of all same shape
        rand_shape = [rand.randint(1, 5) for i in range(rand.randint(1, 5))]
        rand_np = [np.random.rand(*rand_shape) for i in range(self.modes)]
        self.assertTrue(self.traj1.check_type_shape(rand_np)[0])
        self.assertTrue(self.traj1.check_type_shape(rand_np)[1])

        # generate lists with numpy arrays where one is a different shape
        temp = True
        i = 1
        while temp:
            rand_shape_diff = [rand.randint(1, 5) for i in range(rand.randint(1, 5))]
            if rand_shape != rand_shape_diff:
                temp = False
            if i > 5:
                break
            i += 1
        rand_np[rand.randint(0, self.modes - 1)] = np.random.rand(*rand_shape_diff)
        self.assertTrue(self.traj1.check_type_shape(rand_np)[0])
        self.assertFalse(self.traj1.check_type_shape(rand_np)[1])

    def test_getitem(self):
        rand_mode = rand.randint(0, self.modes - 1)
        rand_el1 = rand.randint(0, self.dim1 - 1)
        rand_el2 = rand.randint(0, self.dim2 - 1)
        rand_el3 = rand.randint(0, self.dim3 - 1)
        self.assertEqual(self.traj1[rand_mode, rand_el1], self.rand_list1[rand_mode][rand_el1])
        self.assertEqual(self.traj2[rand_mode, rand_el2, rand_el3], self.rand_list2[rand_mode][rand_el2, rand_el3])
        self.assertTrue(np.array_equal(self.traj1[rand_mode, :], self.rand_list1[rand_mode]))
        self.assertTrue(np.array_equal(self.traj2[rand_mode, rand_el2, :], self.rand_list2[rand_mode][rand_el2, :]))
        self.assertTrue(np.array_equal(self.traj2[rand_mode, :, rand_el3], self.rand_list2[rand_mode][:, rand_el3]))

    def test_round(self):
        rand_mode = rand.randint(0, self.modes - 1)
        rand_el1 = rand.randint(0, self.dim1 - 1)
        rand_el2 = rand.randint(0, self.dim2 - 1)
        rand_el3 = rand.randint(0, self.dim3 - 1)
        rand_round = rand.randint(0, 10)
        traj1_round = round(self.traj1, rand_round)
        rand_list1_round = round(self.rand_list1[rand_mode][rand_el1], rand_round)
        traj2_round = round(self.traj2, rand_round)
        rand_list2_round = round(self.rand_list2[rand_mode][rand_el2, rand_el3], rand_round)
        self.assertEqual(traj1_round[rand_mode, rand_el1], rand_list1_round)
        self.assertEqual(traj2_round[rand_mode, rand_el2, rand_el3], rand_list2_round)


if __name__ == "__main__":
    unittest.main()
