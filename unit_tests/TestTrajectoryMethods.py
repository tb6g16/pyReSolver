# This file contains the tests for the Trajectory class and its associated
# methods

import sys
sys.path.append(r"C:\Users\user\Desktop\PhD\Bruno Paper\Code\Approach B")
import unittest
import numpy as np
import random as rand
from Trajectory import Trajectory
import trajectory_functions as traj_funcs
from System import System
from test_cases import unit_circle as uc
from test_cases import ellipse as elps
from test_cases import van_der_pol as vpd
from test_cases import viswanath as vis

class TestTrajectoryMethods(unittest.TestCase):
    
    def setUp(self):
        self.traj1 = Trajectory(uc.x, disc = 64)
        self.freq1 = 1
        self.traj2 = Trajectory(elps.x, disc = 64)
        self.freq2 = 1
        self.sys1 = System(vpd)
        self.sys2 = System(vis)

    def tearDown(self):
        del self.traj1
        del self.traj2
        del self.sys1
        del self.sys2

    # def test_mul_float(self):
    #     # random numbers to multiply by
    #     rand1 = rand.uniform(-10, 10)
    #     rand2 = rand.uniform(-10, 10)

    #     # perform multiplication via magic method
    #     float_mul1 = rand1*self.traj1
    #     float_mul2 = rand2*self.traj2

    #     # correct elements
    #     for i in range(self.traj1.shape[0]):
    #         for j in range(self.traj1.shape[1]):
    #             self.assertEqual(float_mul1[i, j], self.traj1[i, j]*rand1)
    #             self.assertEqual(float_mul2[i, j], self.traj2[i, j]*rand2)
        
    #     # check commutativity (__rmul__)
    #     self.assertEqual(float_mul1, self.traj1*rand1)
    #     self.assertEqual(float_mul2, self.traj2*rand2)

    # def test_matmul_array(self):
    #     # random matrices to multiple by
    #     rand1 = np.random.rand(2, 2)
    #     rand2 = np.random.rand(2, 2)

    #     # perform multiplication via magic method
    #     mat_mul1 = rand1 @ self.traj1
    #     mat_mul2 = rand2 @ self.traj2

    #     # correct elements
    #     for i in range(self.traj1.shape[1]):
    #         self.assertTrue(np.allclose(mat_mul1[:, i], rand1 @ self.traj1[:, i], rtol = 1e-12))
    #         self.assertTrue(np.allclose(mat_mul2[:, i], rand2 @ self.traj2[:, i], rtol = 1e-12))

    #     # check commutativity (__rmatmul__)
    #     self.assertEqual(mat_mul1, self.traj1 @ rand1)
    #     self.assertEqual(mat_mul2, self.traj2 @ rand2)

    def test_matmul_func(self):
        # initialise jacobian matrices
        mat_func_t1s1 = traj_funcs.jacob_init(self.traj1, self.sys1)
        mat_func_t2s1 = traj_funcs.jacob_init(self.traj2, self.sys1)
        mat_func_t1s2 = traj_funcs.jacob_init(self.traj1, self.sys2)
        mat_func_t2s2 = traj_funcs.jacob_init(self.traj2, self.sys2)

        # random index
        # rind_t1s1 = int(rand.uniform(0, 64))
        # rind_t2s1 = int(rand.uniform(0, 64))
        # rind_t1s2 = int(rand.uniform(0, 64))
        # rind_t2s2 = int(rand.uniform(0, 64))

        # calculate product via magin method
        matmul_t1s1 = mat_func_t1s1 @ self.traj1
        matmul_t2s1 = mat_func_t2s1 @ self.traj2
        matmul_t1s2 = mat_func_t1s2 @ self.traj1
        matmul_t2s2 = mat_func_t2s2 @ self.traj2

        # outputs are trajectories
        self.assertIsInstance(matmul_t1s1, Trajectory)
        self.assertIsInstance(matmul_t2s1, Trajectory)
        self.assertIsInstance(matmul_t1s2, Trajectory)
        self.assertIsInstance(matmul_t2s2, Trajectory)

        # outputs are correct size
        # self.assertEqual(matmul_t1s1.shape, self.traj1.shape)
        # self.assertEqual(matmul_t2s1.shape, self.traj2.shape)
        # self.assertEqual(matmul_t1s2.shape, self.traj1.shape)
        # self.assertEqual(matmul_t2s2.shape, self.traj2.shape)

        # outputs are real numbers
        # temp = True
        # if matmul_t1s1.mode_array.dtype != np.int64 and matmul_t1s1.mode_array.dtype != np.float64:
        #     temp = False
        # if matmul_t2s1.mode_array.dtype != np.int64 and matmul_t2s1.mode_array.dtype != np.float64:
        #     temp = False
        # if matmul_t1s2.mode_array.dtype != np.int64 and matmul_t1s2.mode_array.dtype != np.float64:
        #     temp = False
        # if matmul_t2s2.mode_array.dtype != np.int64 and matmul_t2s2.mode_array.dtype != np.float64:
        #     temp = False
        # self.assertTrue(temp)

        # CONVERT BOTH TO TIME AND COMPARE!!!
        # correct values at random vector
        # self.assertTrue(np.allclose(matmul_t1s1[:, rind_t1s1], mat_func_t1s1(rind_t1s1) @ self.traj1[:, rind_t1s1]))
        # self.assertTrue(np.allclose(matmul_t2s1[:, rind_t2s1], mat_func_t2s1(rind_t2s1) @ self.traj2[:, rind_t2s1]))
        # self.assertTrue(np.allclose(matmul_t1s2[:, rind_t1s2], mat_func_t1s2(rind_t1s2) @ self.traj1[:, rind_t1s2]))
        # self.assertTrue(np.allclose(matmul_t2s2[:, rind_t2s2], mat_func_t2s2(rind_t2s2) @ self.traj2[:, rind_t2s2]))

    # def test_eq(self):
    #     self.assertTrue(self.traj1 + self.traj1 == 2*self.traj1)
    #     self.assertTrue(self.traj2 + self.traj2 == 2*self.traj2)


if __name__ == "__main__":
    unittest.main()
