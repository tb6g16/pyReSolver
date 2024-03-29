# This file contains the testing methods for the functions defined in
# trajectory_functions.

import unittest
import random as rand

import numpy as np

import pyReSolver

import pyReSolver.trajectory_functions as traj_funcs

from tests.test_trajectories import unit_circle as uc
from tests.test_trajectories import ellipse as elps
from tests.test_trajectories import unit_circle_3d as uc3

class TestTrajectoryFunctions(unittest.TestCase):

    def setUp(self):
        curve1 = pyReSolver.utils.func2curve(uc, 33)
        curve2 = pyReSolver.utils.func2curve(elps, 33)
        self.plans_t1 = pyReSolver.FFTPlans(curve1.shape, flag = 'FFTW_ESTIMATE')
        self.plans_t2 = pyReSolver.FFTPlans(curve2.shape, flag = 'FFTW_ESTIMATE')
        self.traj1 = pyReSolver.Trajectory(np.fft.rfft(curve1, axis = 0)/np.shape(curve1)[0])
        self.traj2 = pyReSolver.Trajectory(np.fft.rfft(curve2, axis = 0)/np.shape(curve2)[0])
        self.sys1 = pyReSolver.systems.van_der_pol
        self.sys2 = pyReSolver.systems.viswanath

    def tearDown(self):
        del self.traj1
        del self.traj2
        del self.plans_t1
        del self.plans_t2
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
        traj1 = pyReSolver.Trajectory(array1_real + 1j*array1_imag)
        traj2 = pyReSolver.Trajectory(array2_real + 1j*array2_imag)
        traj3 = pyReSolver.Trajectory(array3_real + 1j*array3_imag)

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
        traj1 = pyReSolver.Trajectory(array1_real + 1j*array1_imag)
        traj2 = pyReSolver.Trajectory(array2_real + 1j*array2_imag)
        traj3 = pyReSolver.Trajectory(array3_real + 1j*array3_imag)

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
        traj1_grad = pyReSolver.Trajectory(np.zeros_like(self.traj1))
        traj2_grad = pyReSolver.Trajectory(np.zeros_like(self.traj2))
        traj_funcs.traj_grad(self.traj1, traj1_grad)
        traj_funcs.traj_grad(self.traj2, traj2_grad)

        # same shape as original trajectories
        self.assertEqual(self.traj1.shape, traj1_grad.shape)
        self.assertEqual(self.traj2.shape, traj2_grad.shape)

        # correct values
        traj1_grad_true = pyReSolver.Trajectory(np.zeros_like(traj1_grad))
        traj2_grad_true = pyReSolver.Trajectory(np.zeros_like(traj2_grad))
        traj1_grad_true[1, 0] = 0.5*1j
        traj1_grad_true[1, 1] = -0.5
        traj2_grad_true[1, 0] = 1j
        traj2_grad_true[1, 1] = -0.5
        self.assertAlmostEqual(traj1_grad_true, traj1_grad)
        self.assertAlmostEqual(traj2_grad_true, traj2_grad)

    def test_traj_response(self):
        # initialise arrays to be modified in-place
        curve1_response1 = np.zeros_like(self.plans_t1.tmp_t)
        curve1_response2 = np.zeros_like(self.plans_t1.tmp_t)
        traj1_response1 = np.zeros_like(self.traj1)
        traj1_response2 = np.zeros_like(self.traj1)
        traj2_response1 = np.zeros_like(self.traj2)
        traj2_response2 = np.zeros_like(self.traj2)
        curve1_nl1 = np.zeros_like(self.plans_t1.tmp_t)
        curve1_nl2 = np.zeros_like(self.plans_t1.tmp_t)
        curve2_nl1 = np.zeros_like(self.plans_t2.tmp_t)
        curve2_nl2 = np.zeros_like(self.plans_t2.tmp_t)
        traj1_nl1 = np.zeros_like(self.traj1)
        traj1_nl2 = np.zeros_like(self.traj1)
        traj2_nl1 = np.zeros_like(self.traj2)
        traj2_nl2 = np.zeros_like(self.traj2)

        # response to full system
        traj_funcs.traj_response(self.traj1, self.plans_t1, self.sys1.response, traj1_response1, curve1_response1)
        traj_funcs.traj_response(self.traj1, self.plans_t1, self.sys2.response, traj1_response2, curve1_response2)
        traj_funcs.traj_response(self.traj2, self.plans_t2, self.sys1.response, traj2_response1, curve1_response1)
        traj_funcs.traj_response(self.traj2, self.plans_t2, self.sys2.response, traj2_response2, curve1_response2)
        traj_funcs.traj_response(self.traj1, self.plans_t1, self.sys1.nl_factor, traj1_nl1, curve1_nl1)
        traj_funcs.traj_response(self.traj1, self.plans_t1, self.sys2.nl_factor, traj1_nl2, curve1_nl2)
        traj_funcs.traj_response(self.traj2, self.plans_t2, self.sys1.nl_factor, traj2_nl1, curve2_nl1)
        traj_funcs.traj_response(self.traj2, self.plans_t2, self.sys2.nl_factor, traj2_nl2, curve2_nl2)
        
        # output is of the Trajectory class
        self.assertIsInstance(traj1_response1, pyReSolver.Trajectory)
        self.assertIsInstance(traj1_response2, pyReSolver.Trajectory)
        self.assertIsInstance(traj2_response1, pyReSolver.Trajectory)
        self.assertIsInstance(traj2_response2, pyReSolver.Trajectory)
        self.assertIsInstance(traj1_nl1, pyReSolver.Trajectory)
        self.assertIsInstance(traj1_nl2, pyReSolver.Trajectory)
        self.assertIsInstance(traj2_nl1, pyReSolver.Trajectory)
        self.assertIsInstance(traj2_nl2, pyReSolver.Trajectory)

        # same response for trajectories at crossing points in time domain
        t1r1_time = np.zeros_like(self.plans_t1.tmp_t)
        traj_funcs.traj_irfft(traj1_response1, t1r1_time, self.plans_t1)
        cross_i1 = int(((np.shape(t1r1_time)[0])/(2*np.pi))*(np.pi/2))
        cross_i2 = int(((np.shape(t1r1_time)[0])/(2*np.pi))*((3*np.pi)/2))
        t1nl1_time = np.zeros_like(self.plans_t1.tmp_t)
        t1nl2_time = np.zeros_like(self.plans_t1.tmp_t)
        t2nl1_time = np.zeros_like(self.plans_t2.tmp_t)
        t2nl2_time = np.zeros_like(self.plans_t2.tmp_t)
        traj_funcs.traj_irfft(traj1_nl1, t1nl1_time, self.plans_t1)
        traj_funcs.traj_irfft(traj1_nl2, t1nl2_time, self.plans_t1)
        traj_funcs.traj_irfft(traj2_nl1, t2nl1_time, self.plans_t2)
        traj_funcs.traj_irfft(traj2_nl2, t2nl2_time, self.plans_t2)
        self.assertTrue(np.allclose(t1nl1_time[cross_i1], t2nl1_time[cross_i1]))
        self.assertTrue(np.allclose(t1nl1_time[cross_i2], t2nl1_time[cross_i2]))
        self.assertTrue(np.allclose(t1nl2_time[cross_i1], t2nl2_time[cross_i1]))
        self.assertTrue(np.allclose(t1nl2_time[cross_i2], t2nl2_time[cross_i2]))

    def test_traj_response2(self):
        # test for lorenz
        curve5 = pyReSolver.utils.func2curve(uc3, 5)
        curve6 = np.tile(np.random.rand(curve5.shape[1]), [curve5.shape[0], 1])
        plan_t5 = pyReSolver.FFTPlans(curve5.shape, flag = 'FFTW_ESTIMATE')
        traj5 = pyReSolver.Trajectory(np.zeros_like(plan_t5.tmp_f))
        traj6 = pyReSolver.Trajectory(np.zeros_like(plan_t5.tmp_f))
        traj_funcs.traj_rfft(traj5, curve5, plan_t5)
        traj_funcs.traj_rfft(traj6, curve6, plan_t5)
        t5_lor_jac = np.zeros_like(traj5)
        c5_lor_jac = np.zeros_like(plan_t5.tmp_t)
        tmp_curve = np.zeros_like(plan_t5.tmp_t)
        traj_funcs.traj_response2(traj5, traj6, plan_t5, pyReSolver.systems.lorenz.jac_conv, t5_lor_jac, c5_lor_jac, tmp_curve)
        t5_lor_jac_true = np.zeros([traj5.shape[0], 3], dtype = complex)
        t5_lor_jac_true[0, 0] = pyReSolver.systems.lorenz.parameters['sigma']*(curve6[0, 1] - curve6[0, 0])
        t5_lor_jac_true[0, 1] = (pyReSolver.systems.lorenz.parameters['rho']*curve6[0, 0]) - curve6[0, 1]
        t5_lor_jac_true[0, 2] = -pyReSolver.systems.lorenz.parameters['beta']*curve6[0, 2]
        t5_lor_jac_true[1, 1] = -0.5*curve6[0, 2]
        t5_lor_jac_true[1, 2] = 0.5*(curve6[0, 1] + curve6[0,0]*1j)
        t5_lor_jac_true = pyReSolver.Trajectory(t5_lor_jac_true)
        self.assertAlmostEqual(t5_lor_jac, t5_lor_jac_true)


if __name__ == "__main__":
    unittest.main()
