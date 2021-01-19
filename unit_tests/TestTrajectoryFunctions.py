# This file contains the testing methods for the functions defined in
# trajectory_functions

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

class TestTrajectoryFunctions(unittest.TestCase):

    def setUp(self):
        self.traj1 = Trajectory(uc.x, disc = 64)
        self.traj2 = Trajectory(elps.x, disc = 64)
        self.sys1 = System(vpd)
        self.sys2 = System(vis)

    def tearDown(self):
        del self.traj1
        del self.traj2
        del self.sys1
        del self.sys2

    def est_traj_inner_prod(self):
        traj1_traj1_prod = traj_funcs.traj_inner_prod(self.traj1, self.traj1)
        traj2_traj2_prod = traj_funcs.traj_inner_prod(self.traj2, self.traj2)
        traj1_traj2_prod = traj_funcs.traj_inner_prod(self.traj1, self.traj2)
        traj2_traj1_prod = traj_funcs.traj_inner_prod(self.traj2, self.traj1)

        # output is of the Trajectory class
        self.assertIsInstance(traj1_traj2_prod, Trajectory)
        self.assertIsInstance(traj2_traj1_prod, Trajectory)

        # does the operation commute
        self.assertEqual(traj1_traj2_prod, traj2_traj1_prod)

        # inner product equal to norm
        traj1_norm = np.ones([1, self.traj1.shape[1]])
        traj1_norm = Trajectory(traj1_norm)
        traj2_norm = np.zeros([1, self.traj2.shape[1]])
        for i in range(self.traj2.shape[1]):
            s = ((2*np.pi)/self.traj2.shape[1])*i
            traj2_norm[0, i] = (4*(np.cos(s)**2)) + (np.sin(s)**2)
        traj2_norm = Trajectory(traj2_norm)
        self.assertEqual(traj1_norm, traj1_traj1_prod)
        self.assertEqual(traj2_norm, traj2_traj2_prod)

        # single number at each index
        temp1 = True
        for i in range(traj1_traj2_prod.shape[1]):
            if traj1_traj2_prod[:, i].shape[0] != 1:
                temp1 = False
        for i in range(traj2_traj1_prod.shape[1]):
            if traj2_traj1_prod[:, i].shape[0] != 1:
                temp1 = False
        self.assertTrue(temp1)

        # outputs are numbers
        # temp2 = True
        # if traj1_traj2_prod.curve_array.dtype != np.int64 and \
        #     traj1_traj2_prod.curve_array.dtype != np.float64:
        #     temp2 = False
        # if traj2_traj1_prod.curve_array.dtype != np.int64 and \
        #     traj2_traj1_prod.curve_array.dtype != np.float64:
        #     temp2 = False
        # self.assertTrue(temp2)

    def st_gradient(self):
        traj1_grad = traj_funcs.traj_grad(self.traj1)
        traj2_grad = traj_funcs.traj_grad(self.traj2)

        # same shape as original trajectories
        self.assertEqual(self.traj1.shape, traj1_grad.shape)
        self.assertEqual(self.traj2.shape, traj2_grad.shape)

        # outputs are real numbers
        # temp = True
        # if traj1_grad.curve_array.dtype != np.int64 and \
        #     traj1_grad.curve_array.dtype != np.float64:
        #     temp = False
        # if traj2_grad.curve_array.dtype != np.int64 and \
        #     traj2_grad.curve_array.dtype != np.float64:
        #     temp = False
        # self.assertTrue(temp)

        # correct values
        traj1_grad_true = np.zeros(self.traj1.shape)
        traj2_grad_true = np.zeros(self.traj2.shape)
        for i in range(self.traj2.shape[1]):
            s = ((2*np.pi)/self.traj2.shape[1])*i
            traj1_grad_true[0, i] = -np.sin(s)
            traj1_grad_true[1, i] = -np.cos(s)
            traj2_grad_true[0, i] = -2*np.sin(s)
            traj2_grad_true[1, i] = -np.cos(s)
        traj1_grad_true = Trajectory(traj1_grad_true)
        traj2_grad_true = Trajectory(traj2_grad_true)
        self.assertEqual(traj1_grad_true, traj1_grad)
        self.assertEqual(traj2_grad_true, traj2_grad)

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
        if traj1_response1.mode_array.dtype != np.complex128:
            temp1 = False
        if traj1_response2.mode_array.dtype != np.complex128:
            temp1 = False
        if traj2_response1.mode_array.dtype != np.complex128:
            temp1 = False
        if traj2_response2.mode_array.dtype != np.complex128:
            temp1 = False
        temp2 = True
        if traj1_nl1.mode_array.dtype != np.complex128:
            temp2 = False
        if traj1_nl2.mode_array.dtype != np.complex128:
            temp2 = False
        if traj2_nl1.mode_array.dtype != np.complex128:
            temp2 = False
        if traj2_nl2.mode_array.dtype != np.complex128:
            temp2 = False
        self.assertTrue(temp1)
        self.assertTrue(temp2)

        # same response for trajectories at crossing points
        t1r1_time = self.traj1.modes2curve(traj1_response1.mode_array)
        t1r2_time = self.traj1.modes2curve(traj1_response2.mode_array)
        t2r1_time = self.traj2.modes2curve(traj2_response1.mode_array)
        t2r2_time = self.traj2.modes2curve(traj2_response2.mode_array)
        cross_i1 = int(((np.shape(t1r1_time)[1])/(2*np.pi))*(np.pi/2))
        cross_i2 = int(((np.shape(t1r1_time)[1])/(2*np.pi))*((3*np.pi)/2))
        traj1_cross1_resp1 = t1r1_time[:, cross_i1]
        traj2_cross1_resp1 = t2r1_time[:, cross_i1]
        traj1_cross2_resp1 = t1r1_time[:, cross_i2]
        traj2_cross2_resp1 = t2r1_time[:, cross_i2]
        traj1_cross1_resp2 = t1r2_time[:, cross_i1]
        traj2_cross1_resp2 = t2r2_time[:, cross_i1]
        traj1_cross2_resp2 = t1r2_time[:, cross_i2]
        traj2_cross2_resp2 = t2r2_time[:, cross_i2]
        self.assertTrue(np.allclose(traj1_cross1_resp1, traj2_cross1_resp1))
        self.assertTrue(np.allclose(traj1_cross2_resp1, traj2_cross2_resp1))
        self.assertTrue(np.allclose(traj1_cross1_resp2, traj2_cross1_resp2))
        self.assertTrue(np.allclose(traj1_cross2_resp2, traj2_cross2_resp2))
        t1nl1_time = self.traj1.modes2curve(traj1_nl1.mode_array)
        t1nl2_time = self.traj1.modes2curve(traj1_nl2.mode_array)
        t2nl1_time = self.traj2.modes2curve(traj2_nl1.mode_array)
        t2nl2_time = self.traj2.modes2curve(traj2_nl2.mode_array)
        traj1_cross1_nl1 = t1nl1_time[:, cross_i1]
        traj2_cross1_nl1 = t2nl1_time[:, cross_i1]
        traj1_cross2_nl1 = t1nl1_time[:, cross_i2]
        traj2_cross2_nl1 = t2nl1_time[:, cross_i2]
        traj1_cross1_nl2 = t1nl2_time[:, cross_i1]
        traj2_cross1_nl2 = t2nl2_time[:, cross_i1]
        traj1_cross2_nl2 = t1nl2_time[:, cross_i2]
        traj2_cross2_nl2 = t2nl2_time[:, cross_i2]
        self.assertTrue(np.allclose(traj1_cross1_nl1, traj2_cross1_nl1))
        self.assertTrue(np.allclose(traj1_cross2_nl1, traj2_cross2_nl1))
        self.assertTrue(np.allclose(traj1_cross1_nl2, traj2_cross1_nl2))
        self.assertTrue(np.allclose(traj1_cross2_nl2, traj2_cross2_nl2))

    def est_jacob_init(self):
        self.sys1.parameters['mu'] = 1
        self.sys2.parameters['mu'] = 1
        sys1_jac = traj_funcs.jacob_init(self.traj1, self.sys1)
        sys2_jac = traj_funcs.jacob_init(self.traj2, self.sys2)
        sys1_jac_tran = traj_funcs.jacob_init(self.traj1, self.sys1, if_transp = True)
        sys2_jac_tran = traj_funcs.jacob_init(self.traj2, self.sys2, if_transp = True)

        # outputs are numbers
        temp1 = True
        rindex1 = int(rand.random()*(self.traj1.shape[1]))
        rindex2 = int(rand.random()*(self.traj2.shape[1]))
        output1 = sys1_jac(rindex1)
        output2 = sys2_jac(rindex2)
        if output1.dtype != np.int64 and output1.dtype != np.float64:
            temp1 = False
        if output2.dtype != np.int64 and output2.dtype != np.float64:
            temp1 = False
        self.assertTrue(temp1)

        # output is correct size
        temp2 = True
        if output1.shape != (2, 2):
            temp2 = False
        if output2.shape != (2, 2):
            temp2 = False
        self.assertTrue(temp2)

        # correct values
        rstate1 = self.traj1[:, rindex1]
        rstate2 = self.traj2[:, rindex2]
        sys1_jac_true = vpd.jacobian(rstate1)
        sys2_jac_true = vis.jacobian(rstate2)
        self.assertTrue(np.allclose(output1, sys1_jac_true))
        self.assertTrue(np.allclose(output2, sys2_jac_true))

        # transpose correct
        output3 = sys1_jac_tran(rindex1)
        output4 = sys2_jac_tran(rindex2)
        self.assertTrue(np.array_equal(output3, np.transpose(sys1_jac_true)))
        self.assertTrue(np.array_equal(output4, np.transpose(sys2_jac_true)))


if __name__ == "__main__":
    unittest.main()
