# This file contains the testing methods for the functions defined in
# trajectory_functions.

import sys
sys.path.append(r"C:\Users\user\Desktop\PhD\Bruno Paper\ResolventSolver")
import unittest
import numpy as np
import random as rand
from Trajectory import Trajectory
import trajectory_functions as traj_funcs
from traj_util import array2list, list2array
from trajectory_definitions import unit_circle as uc
from trajectory_definitions import ellipse as elps
from trajectory_definitions import unit_circle_3d as uc3
from systems import van_der_pol as vpd
from systems import viswanath as vis
from systems import lorenz

class TestTrajectoryFunctions(unittest.TestCase):

    def setUp(self):
        self.traj1 = Trajectory(uc.x, modes = 33)
        self.traj2 = Trajectory(elps.x, modes = 33)
        self.sys1 = vpd
        self.sys2 = vis

    def tearDown(self):
        del self.traj1
        del self.traj2
        del self.sys1
        del self.sys2

    def test_transpose(self):
        # set up random trajectories
        i1 = rand.randint(1, 100)
        i2 = rand.randint(1, 10)
        i3 = rand.randint(1, 10)
        array1_real = np.random.rand(i1, i2, i3)
        array2_real = np.random.rand(i1, i2, i3)
        array3_real = np.random.rand(i1, i2, i3)
        array1_imag = np.random.rand(i1, i2, i3)
        array2_imag = np.random.rand(i1, i2, i3)
        array3_imag = np.random.rand(i1, i2, i3)
        traj1 = Trajectory(array2list(array1_real + 1j*array1_imag))
        traj2 = Trajectory(array2list(array2_real + 1j*array2_imag))
        traj3 = Trajectory(array2list(array3_real + 1j*array3_imag))

        # take transpose
        traj1_tran = traj_funcs.transpose(traj1)
        traj2_tran = traj_funcs.transpose(traj2)
        traj3_tran = traj_funcs.transpose(traj3)

        # double application same as original
        self.assertEqual(traj1, traj_funcs.transpose(traj1_tran))
        self.assertEqual(traj2, traj_funcs.transpose(traj2_tran))
        self.assertEqual(traj3, traj_funcs.transpose(traj3_tran))

        # swapped indices match
        for i in range(traj1.shape[0]):
            for j in range(traj1.shape[1]):
                for k in range(traj1.shape[2]):
                    self.assertEqual(traj1[i, j, k], traj1_tran[i, k, j])
                    self.assertEqual(traj2[i, j, k], traj2_tran[i, k, j])
                    self.assertEqual(traj3[i, j, k], traj3_tran[i, k, j])

    def test_conj(self):
        # set up random trajectories
        i1 = rand.randint(1, 100)
        i2 = rand.randint(1, 10)
        i3 = rand.randint(1, 10)
        array1_real = np.random.rand(i1, i2, i3)
        array2_real = np.random.rand(i1, i2, i3)
        array3_real = np.random.rand(i1, i2, i3)
        array1_imag = np.random.rand(i1, i2, i3)
        array2_imag = np.random.rand(i1, i2, i3)
        array3_imag = np.random.rand(i1, i2, i3)
        traj1 = Trajectory(array2list(array1_real + 1j*array1_imag))
        traj2 = Trajectory(array2list(array2_real + 1j*array2_imag))
        traj3 = Trajectory(array2list(array3_real + 1j*array3_imag))

        # take conjugate
        traj1_conj = traj_funcs.conj(traj1)
        traj2_conj = traj_funcs.conj(traj2)
        traj3_conj = traj_funcs.conj(traj3)

        # double application same as original
        self.assertEqual(traj1, traj_funcs.conj(traj1_conj))
        self.assertEqual(traj2, traj_funcs.conj(traj2_conj))
        self.assertEqual(traj3, traj_funcs.conj(traj3_conj))

        # swapped indices match
        for i in range(traj1.shape[0]):
            for j in range(traj1.shape[1]):
                for k in range(traj1.shape[2]):
                    self.assertEqual(traj1[i, j, k], np.conj(traj1_conj[i, j, k]))
                    self.assertEqual(traj2[i, j, k], np.conj(traj2_conj[i, j, k]))
                    self.assertEqual(traj3[i, j, k], np.conj(traj3_conj[i, j, k]))

    def test_gradient(self):
        traj1_grad = traj_funcs.traj_grad(self.traj1)
        traj2_grad = traj_funcs.traj_grad(self.traj2)

        # same shape as original trajectories
        self.assertEqual(self.traj1.shape, traj1_grad.shape)
        self.assertEqual(self.traj2.shape, traj2_grad.shape)

        # outputs are complex numbers
        temp = True
        for i in range(traj1_grad.shape[0]):
            if traj1_grad.mode_list[i].dtype != np.complex128:
                temp = False
            if traj2_grad.mode_list[i].dtype != np.complex128:
                temp = False
        self.assertTrue(temp)

        # correct values
        traj1_grad_time = traj_funcs.traj_irfft(traj1_grad)
        traj2_grad_time = traj_funcs.traj_irfft(traj2_grad)
        traj1_grad_true = np.zeros(traj1_grad_time.shape)
        traj2_grad_true = np.zeros(traj2_grad_time.shape)
        for i in range(traj2_grad_time.shape[0]):
            s = ((2*np.pi)/traj2_grad_time.shape[0])*i
            traj1_grad_true[i, 0] = -np.sin(s)
            traj1_grad_true[i, 1] = -np.cos(s)
            traj2_grad_true[i, 0] = -2*np.sin(s)
            traj2_grad_true[i, 1] = -np.cos(s)
        self.assertTrue(np.allclose(traj1_grad_true, traj1_grad_time))
        self.assertTrue(np.allclose(traj2_grad_true, traj2_grad_time))

    def test_traj_response(self):
        # response to full system
        traj1_response1 = traj_funcs.traj_response(self.traj1, self.sys1.response)
        traj1_response2 = traj_funcs.traj_response(self.traj1, self.sys2.response)
        traj2_response1 = traj_funcs.traj_response(self.traj2, self.sys1.response)
        traj2_response2 = traj_funcs.traj_response(self.traj2, self.sys2.response)
        traj1_nl1 = traj_funcs.traj_response(self.traj1, self.sys1.nl_factor)
        traj1_nl2 = traj_funcs.traj_response(self.traj1, self.sys2.nl_factor)
        traj2_nl1 = traj_funcs.traj_response(self.traj2, self.sys1.nl_factor)
        traj2_nl2 = traj_funcs.traj_response(self.traj2, self.sys2.nl_factor)
        
        # output is of the Trajectory class
        self.assertIsInstance(traj1_response1, Trajectory)
        self.assertIsInstance(traj1_response2, Trajectory)
        self.assertIsInstance(traj2_response1, Trajectory)
        self.assertIsInstance(traj2_response2, Trajectory)
        self.assertIsInstance(traj1_nl1, Trajectory)
        self.assertIsInstance(traj1_nl2, Trajectory)
        self.assertIsInstance(traj2_nl1, Trajectory)
        self.assertIsInstance(traj2_nl2, Trajectory)

        # outputs are complex numbers
        temp1 = True
        for i in range(traj1_response1.shape[0]):
            if traj1_response1.mode_list[i].dtype != np.complex128:
                temp1 = False
            if traj1_response2.mode_list[i].dtype != np.complex128:
                temp1 = False
            if traj2_response1.mode_list[i].dtype != np.complex128:
                temp1 = False
            if traj2_response2.mode_list[i].dtype != np.complex128:
                temp1 = False
            temp2 = True
            if traj1_nl1.mode_list[i].dtype != np.complex128:
                temp2 = False
            if traj1_nl2.mode_list[i].dtype != np.complex128:
                temp2 = False
            if traj2_nl1.mode_list[i].dtype != np.complex128:
                temp2 = False
            if traj2_nl2.mode_list[i].dtype != np.complex128:
                temp2 = False
        self.assertTrue(temp1)
        self.assertTrue(temp2)

        # same response for trajectories at crossing points in time domain
        t1r1_time = traj_funcs.traj_irfft(traj1_response1)
        # t1r2_time = traj_funcs.traj_irfft(traj1_response2)
        # t2r1_time = traj_funcs.traj_irfft(traj2_response1)
        # t2r2_time = traj_funcs.traj_irfft(traj2_response2)
        cross_i1 = int(((np.shape(t1r1_time)[0])/(2*np.pi))*(np.pi/2))
        cross_i2 = int(((np.shape(t1r1_time)[0])/(2*np.pi))*((3*np.pi)/2))
        # traj1_cross1_resp1 = t1r1_time[cross_i1]
        # traj2_cross1_resp1 = t2r1_time[cross_i1]
        # traj1_cross2_resp1 = t1r1_time[cross_i2]
        # traj2_cross2_resp1 = t2r1_time[cross_i2]
        # traj1_cross1_resp2 = t1r2_time[cross_i1]
        # traj2_cross1_resp2 = t2r2_time[cross_i1]
        # traj1_cross2_resp2 = t1r2_time[cross_i2]
        # traj2_cross2_resp2 = t2r2_time[cross_i2]
        # self.assertTrue(np.allclose(traj1_cross1_resp1, traj2_cross1_resp1))
        # self.assertTrue(np.allclose(traj1_cross2_resp1, traj2_cross2_resp1))
        # self.assertTrue(np.allclose(traj1_cross1_resp2, traj2_cross1_resp2))
        # self.assertTrue(np.allclose(traj1_cross2_resp2, traj2_cross2_resp2))
        t1nl1_time = traj_funcs.traj_irfft(traj1_nl1)
        t1nl2_time = traj_funcs.traj_irfft(traj1_nl2)
        t2nl1_time = traj_funcs.traj_irfft(traj2_nl1)
        t2nl2_time = traj_funcs.traj_irfft(traj2_nl2)
        traj1_cross1_nl1 = t1nl1_time[cross_i1]
        traj2_cross1_nl1 = t2nl1_time[cross_i1]
        traj1_cross2_nl1 = t1nl1_time[cross_i2]
        traj2_cross2_nl1 = t2nl1_time[cross_i2]
        traj1_cross1_nl2 = t1nl2_time[cross_i1]
        traj2_cross1_nl2 = t2nl2_time[cross_i1]
        traj1_cross2_nl2 = t1nl2_time[cross_i2]
        traj2_cross2_nl2 = t2nl2_time[cross_i2]
        self.assertTrue(np.allclose(traj1_cross1_nl1, traj2_cross1_nl1))
        self.assertTrue(np.allclose(traj1_cross2_nl1, traj2_cross2_nl1))
        self.assertTrue(np.allclose(traj1_cross1_nl2, traj2_cross1_nl2))
        self.assertTrue(np.allclose(traj1_cross2_nl2, traj2_cross2_nl2))

        # extra test for matrix function
        traj3 = Trajectory(uc3.x)
        sys3 = lorenz
        t3_lor_jac = traj_funcs.traj_response(traj3, sys3.jacobian)
        t3_lor_jac_true = np.zeros([traj3.shape[0], 3, 3], dtype = complex)
        t3_lor_jac_true[0, 0, 0] = -sys3.parameters['sigma']
        t3_lor_jac_true[0, 0, 1] = sys3.parameters['sigma']
        t3_lor_jac_true[0, 1, 0] = sys3.parameters['rho']
        t3_lor_jac_true[0, 1, 1] = -1
        t3_lor_jac_true[0, 2, 2] = -sys3.parameters['beta']
        t3_lor_jac_true[1, 1, 2] = -0.5
        t3_lor_jac_true[1, 2, 0] = 1j*0.5
        t3_lor_jac_true[1, 2, 1] = 0.5
        t3_lor_jac_true = Trajectory(array2list(t3_lor_jac_true))
        self.assertAlmostEqual(t3_lor_jac, t3_lor_jac_true)


if __name__ == "__main__":
    unittest.main()
