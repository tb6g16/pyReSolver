# This file contains the testing methods for the functions defined in
# trajectory_functions.

import unittest
import random as rand

import numpy as np

from ResolventSolver.FFTPlans import FFTPlans
from ResolventSolver.traj_util import func2curve
from ResolventSolver.my_fft import my_rfft
from ResolventSolver.Trajectory import Trajectory
import ResolventSolver.trajectory_functions as traj_funcs
from ResolventSolver.trajectory_definitions import unit_circle as uc
from ResolventSolver.trajectory_definitions import ellipse as elps
from ResolventSolver.trajectory_definitions import unit_circle_3d as uc3
from ResolventSolver.systems import van_der_pol as vpd
from ResolventSolver.systems import viswanath as vis
from ResolventSolver.systems import lorenz

class TestTrajectoryFunctions(unittest.TestCase):

    def setUp(self):
        curve1 = func2curve(uc.x, 33)
        curve2 = func2curve(elps.x, 33)
        self.traj1 = Trajectory(my_rfft(curve1))
        self.traj2 = Trajectory(my_rfft(curve2))
        self.plans_t1 = FFTPlans(curve1.shape, flag = 'FFTW_ESTIMATE')
        self.plans_t2 = FFTPlans(curve2.shape, flag = 'FFTW_ESTIMATE')
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
        i4 = rand.randint(1, 10)
        array1_real = np.random.rand(i1, i2)
        array2_real = np.random.rand(i1, i2, i3)
        array3_real = np.random.rand(i1, i2, i3, i4)
        array1_imag = np.random.rand(i1, i2)
        array2_imag = np.random.rand(i1, i2, i3)
        array3_imag = np.random.rand(i1, i2, i3, i4)
        traj1 = Trajectory(array1_real + 1j*array1_imag)
        traj2 = Trajectory(array2_real + 1j*array2_imag)
        traj3 = Trajectory(array3_real + 1j*array3_imag)

        # take transpose
        traj1_tran = traj_funcs.transpose(traj1)
        traj2_tran = traj_funcs.transpose(traj2)
        traj3_tran = traj_funcs.transpose(traj3)

        # double application same as original
        self.assertEqual(traj1, traj_funcs.transpose(traj1_tran))
        self.assertEqual(traj2, traj_funcs.transpose(traj2_tran))
        self.assertEqual(traj3, traj_funcs.transpose(traj3_tran))

        # correct values
        for i in range(traj1.shape[0]):
            for j in range(traj1.shape[1]):
                self.assertEqual(traj1[i, j], traj1_tran[i, j])
                for k in range(traj2.shape[2]):
                    self.assertEqual(traj2[i, j, k], traj2_tran[i, k, j])
                    for l in range(traj3.shape[3]):
                        self.assertEqual(traj3[i, j, k, l], traj3_tran[i, l, k, j])

    def test_conj(self):
        # set up random trajectories
        i1 = rand.randint(1, 100)
        i2 = rand.randint(1, 10)
        i3 = rand.randint(1, 10)
        i4 = rand.randint(1, 10)
        array1_real = np.random.rand(i1, i2)
        array2_real = np.random.rand(i1, i2, i3)
        array3_real = np.random.rand(i1, i2, i3, i4)
        array1_imag = np.random.rand(i1, i2)
        array2_imag = np.random.rand(i1, i2, i3)
        array3_imag = np.random.rand(i1, i2, i3, i4)
        traj1 = Trajectory(array1_real + 1j*array1_imag)
        traj2 = Trajectory(array2_real + 1j*array2_imag)
        traj3 = Trajectory(array3_real + 1j*array3_imag)

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
                self.assertEqual(traj1[i, j], np.conj(traj1_conj[i, j]))
                for k in range(traj2.shape[2]):
                    self.assertEqual(traj2[i, j, k], np.conj(traj2_conj[i, j, k]))
                    for l in range(traj3.shape[3]):
                        self.assertEqual(traj3[i, j, k, l], np.conj(traj3_conj[i, j, k, l]))

    def test_traj_grad(self):
        traj1_grad = traj_funcs.traj_grad(self.traj1)
        traj2_grad = traj_funcs.traj_grad(self.traj2)

        # same shape as original trajectories
        self.assertEqual(self.traj1.shape, traj1_grad.shape)
        self.assertEqual(self.traj2.shape, traj2_grad.shape)

        # outputs are complex numbers
        temp = True
        if traj1_grad.modes.dtype != np.complex128:
            temp = False
        if traj2_grad.modes.dtype != np.complex128:
            temp = False
        self.assertTrue(temp)

        # correct values
        traj1_grad_true = Trajectory(np.zeros_like(traj1_grad.modes))
        traj2_grad_true = Trajectory(np.zeros_like(traj2_grad.modes))
        traj1_grad_true[1, 0] = 0.5*1j
        traj1_grad_true[1, 1] = -0.5
        traj2_grad_true[1, 0] = 1j
        traj2_grad_true[1, 1] = -0.5
        self.assertAlmostEqual(traj1_grad_true, traj1_grad)
        self.assertAlmostEqual(traj2_grad_true, traj2_grad)

    def test_traj_response(self):
        # response to full system
        traj1_response1 = traj_funcs.traj_response(self.traj1, self.plans_t1, self.sys1.response)
        traj1_response2 = traj_funcs.traj_response(self.traj1, self.plans_t1, self.sys2.response)
        traj2_response1 = traj_funcs.traj_response(self.traj2, self.plans_t2, self.sys1.response)
        traj2_response2 = traj_funcs.traj_response(self.traj2, self.plans_t2, self.sys2.response)
        traj1_nl1 = traj_funcs.traj_response(self.traj1, self.plans_t1, self.sys1.nl_factor)
        traj1_nl2 = traj_funcs.traj_response(self.traj1, self.plans_t1, self.sys2.nl_factor)
        traj2_nl1 = traj_funcs.traj_response(self.traj2, self.plans_t2, self.sys1.nl_factor)
        traj2_nl2 = traj_funcs.traj_response(self.traj2, self.plans_t2, self.sys2.nl_factor)
        
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
        if traj1_response1.modes.dtype != np.complex128:
            temp1 = False
        if traj1_response2.modes.dtype != np.complex128:
            temp1 = False
        if traj2_response1.modes.dtype != np.complex128:
            temp1 = False
        if traj2_response2.modes.dtype != np.complex128:
            temp1 = False
        temp2 = True
        if traj1_nl1.modes.dtype != np.complex128:
            temp2 = False
        if traj1_nl2.modes.dtype != np.complex128:
            temp2 = False
        if traj2_nl1.modes.dtype != np.complex128:
            temp2 = False
        if traj2_nl2.modes.dtype != np.complex128:
            temp2 = False
        self.assertTrue(temp1)
        self.assertTrue(temp2)

        # same response for trajectories at crossing points in time domain
        t1r1_time = traj_funcs.traj_irfft(traj1_response1, self.plans_t1)
        # t1r2_time = traj_funcs.traj_irfft(traj1_response2)
        # t2r1_time = traj_funcs.traj_irfft(traj2_response1)
        # t2r2_time = traj_funcs.traj_irfft(traj2_response2)
        cross_i1 = int(((np.shape(t1r1_time)[0])/(2*np.pi))*(np.pi/2))
        cross_i2 = int(((np.shape(t1r1_time)[0])/(2*np.pi))*((3*np.pi)/2))
        # self.assertTrue(np.allclose(t1r1_time[cross_i1], t2r1_time[cross_i1]))
        # self.assertTrue(np.allclose(t1r1_time[cross_i2], t2r1_time[cross_i2]))
        # self.assertTrue(np.allclose(t1r2_time[cross_i1], t2r2_time[cross_i1]))
        # self.assertTrue(np.allclose(t1r2_time[cross_i2], t2r2_time[cross_i2]))
        t1nl1_time = traj_funcs.traj_irfft(traj1_nl1, self.plans_t1)
        t1nl2_time = traj_funcs.traj_irfft(traj1_nl2, self.plans_t1)
        t2nl1_time = traj_funcs.traj_irfft(traj2_nl1, self.plans_t2)
        t2nl2_time = traj_funcs.traj_irfft(traj2_nl2, self.plans_t2)
        self.assertTrue(np.allclose(t1nl1_time[cross_i1], t2nl1_time[cross_i1]))
        self.assertTrue(np.allclose(t1nl1_time[cross_i2], t2nl1_time[cross_i2]))
        self.assertTrue(np.allclose(t1nl2_time[cross_i1], t2nl2_time[cross_i1]))
        self.assertTrue(np.allclose(t1nl2_time[cross_i2], t2nl2_time[cross_i2]))

        # extra test for matrix function
        curve3 = func2curve(uc3.x, 33)
        curve4 = np.ones_like(curve3)
        traj3 = Trajectory(my_rfft(curve3))
        traj4 = Trajectory(my_rfft(curve4))
        plan_t3 = FFTPlans(curve3.shape, flag = 'FFTW_ESTIMATE')
        t3_lor_jac = traj_funcs.traj_response2(traj3, traj4, plan_t3, lorenz.jac_conv)
        t3_lor_jac_true = np.zeros([traj3.shape[0], 3], dtype = complex)
        t3_lor_jac_true[0, 1] = lorenz.parameters['rho'] - 1
        t3_lor_jac_true[0, 2] = -lorenz.parameters['beta']
        # NOTE: should this be 0.5 or -0.5 (get the actual answer)
        t3_lor_jac_true[1, 1] = 0.5
        t3_lor_jac_true[1, 2] = 0.5*(1 + 1j)
        t3_lor_jac_true = Trajectory(t3_lor_jac_true)
        for i in range(traj3.shape[0]):
            if not np.allclose(t3_lor_jac[i], t3_lor_jac_true[i]):
                print(t3_lor_jac[i])
                print(t3_lor_jac_true[i])
        # self.assertAlmostEqual(t3_lor_jac, t3_lor_jac_true)


if __name__ == "__main__":
    unittest.main()
