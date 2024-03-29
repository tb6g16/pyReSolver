# This file contains the tests for the residual calculation functions defined
# in the residual_functions file.

import unittest
import random as rand

import numpy as np

import pyReSolver

from pyReSolver.Cache import Cache
import pyReSolver.trajectory_functions as traj_funcs
import pyReSolver.residual_functions as res_funcs

from tests.test_trajectories import unit_circle as uc
from tests.test_trajectories import ellipse as elps
from tests.test_trajectories import unit_circle_3d as uc3

def init_H_n_inv(traj, sys, freq, mean):
    jac_at_mean = sys.jacobian(mean)
    return pyReSolver.resolvent_inv(traj.shape[0], freq, jac_at_mean)

class TestResidualFunctions(unittest.TestCase):

    def setUp(self):
        curve1 = pyReSolver.utils.func2curve(uc, 33)
        curve2 = pyReSolver.utils.func2curve(elps, 33)
        curve3 = pyReSolver.utils.func2curve(uc3, 33)
        self.plans_t1 = pyReSolver.FFTPlans(curve1.shape, flag = 'FFTW_ESTIMATE')
        self.plans_t2 = pyReSolver.FFTPlans(curve2.shape, flag = 'FFTW_ESTIMATE')
        self.plans_t3 = pyReSolver.FFTPlans(curve3.shape, flag = 'FFTW_ESTIMATE')
        self.traj1 = pyReSolver.Trajectory(np.zeros_like(self.plans_t1.tmp_f))
        self.traj2 = pyReSolver.Trajectory(np.zeros_like(self.plans_t2.tmp_f))
        self.traj3 = pyReSolver.Trajectory(np.zeros_like(self.plans_t3.tmp_f))
        self.mean1 = np.zeros([1, 2])
        self.mean2 = np.zeros([1, 2])
        traj_funcs.traj_rfft(self.traj1, curve1, self.plans_t1)
        traj_funcs.traj_rfft(self.traj2, curve2, self.plans_t2)
        traj_funcs.traj_rfft(self.traj3, curve3, self.plans_t3)
        self.sys1 = pyReSolver.systems.van_der_pol
        self.sys2 = pyReSolver.systems.viswanath
        self.sys3 = pyReSolver.systems.lorenz
        self.cache1 = Cache(self.traj1, self.mean1, self.sys1, self.plans_t1)
        self.cache2 = Cache(self.traj2, self.mean2, self.sys2, self.plans_t2)

    def tearDown(self):
        del self.traj1
        del self.traj2
        del self.traj3
        del self.mean1
        del self.mean2
        del self.plans_t1
        del self.plans_t2
        del self.plans_t3
        del self.sys1
        del self.sys2
        del self.sys3
        del self.cache1
        del self.cache2

    def test_local_residual(self):
        # generating random frequencies and system parameters
        freq1 = rand.uniform(-10, 10)
        freq2 = rand.uniform(-10, 10)
        mu1 = rand.uniform(0, 10)
        mu2 = rand.uniform(0, 10)
        r = rand.uniform(0, 10)

        # apply parameters
        self.sys1.parameters['mu'] = mu1
        self.sys2.parameters['mu'] = mu2
        self.sys2.parameters['r'] = r

        # generate inverse resolvent matrices
        H_n_inv_t1s1 = init_H_n_inv(self.traj1, self.sys1, freq1, self.mean1)
        H_n_inv_t2s1 = init_H_n_inv(self.traj2, self.sys1, freq2, self.mean2)

        # generate local residual trajectories
        lr_traj1_sys1 = res_funcs.local_residual(self.cache1, self.sys1, H_n_inv_t1s1, self.plans_t1)
        lr_traj2_sys1 = res_funcs.local_residual(self.cache2, self.sys1, H_n_inv_t2s1, self.plans_t2)

        # output is of Trajectory class
        self.assertIsInstance(lr_traj1_sys1, pyReSolver.Trajectory)
        self.assertIsInstance(lr_traj2_sys1, pyReSolver.Trajectory)

        # output is of correct shape
        self.assertEqual(lr_traj1_sys1.shape, self.traj1.shape)
        self.assertEqual(lr_traj2_sys1.shape, self.traj2.shape)

        # correct values
        lr_traj1_sys1_true = np.zeros_like(self.plans_t1.tmp_t)
        lr_traj2_sys1_true = np.zeros_like(self.plans_t2.tmp_t)
        traj_funcs.traj_irfft(self.traj1, lr_traj1_sys1_true, self.plans_t1)
        traj_funcs.traj_irfft(self.traj2, lr_traj2_sys1_true, self.plans_t2)
        for i in range(np.shape(lr_traj1_sys1_true)[0]):
            s = ((2*np.pi)/np.shape(lr_traj1_sys1_true)[0])*i
            lr_traj1_sys1_true[i, 0] = (1 - freq1)*np.sin(s)
            lr_traj1_sys1_true[i, 1] = (mu1*(1 - (np.cos(s)**2))*np.sin(s)) + ((1 - freq1)*np.cos(s))
        lr_traj1_sys1_true_f = np.zeros_like(self.traj1)
        traj_funcs.traj_rfft(lr_traj1_sys1_true_f, lr_traj1_sys1_true, self.plans_t1)
        for i in range(np.shape(lr_traj2_sys1_true)[0]):
            s = ((2*np.pi)/np.shape(lr_traj2_sys1_true)[0])*i
            lr_traj2_sys1_true[i, 0] = (1 - (2*freq2))*np.sin(s)
            lr_traj2_sys1_true[i, 1] = ((2 - freq2)*np.cos(s)) + (mu1*(1 - (4*(np.cos(s)**2)))*np.sin(s))
        lr_traj2_sys1_true_f = np.zeros_like(self.traj2)
        traj_funcs.traj_rfft(lr_traj2_sys1_true_f, lr_traj2_sys1_true, self.plans_t2)
        self.assertEqual(lr_traj1_sys1, lr_traj1_sys1_true_f)
        self.assertEqual(lr_traj2_sys1, lr_traj2_sys1_true_f)

    def test_global_residual(self):
        # generating random frequencies and system parameters
        freq1 = rand.uniform(-10, 10)
        freq2 = rand.uniform(-10, 10)
        mu1 = rand.uniform(0, 10)
        mu2 = rand.uniform(0, 10)
        r = rand.uniform(0, 10)

        # apply parameters
        self.sys1.parameters['mu'] = mu1
        self.sys2.parameters['mu'] = mu2
        self.sys2.parameters['r'] = r

        # calculate global residuals
        H_n_inv_t1s1 = init_H_n_inv(self.traj1, self.sys1, freq1, self.mean1)
        H_n_inv_t2s1 = init_H_n_inv(self.traj2, self.sys1, freq2, self.mean2)
        _ = res_funcs.local_residual(self.cache1, self.sys1, H_n_inv_t1s1, self.plans_t1)
        _ = res_funcs.local_residual(self.cache2, self.sys1, H_n_inv_t2s1, self.plans_t2)
        gr_traj1_sys1 = res_funcs.global_residual(self.cache1)
        gr_traj2_sys1 = res_funcs.global_residual(self.cache2)

        # correct values
        gr_traj1_sys1_true = ((5*(mu1**2))/32) + (((freq1 - 1)**2)/2)
        gr_traj2_sys1_true = (1/4)*((((2*freq2) - 1)**2) + ((2 - freq2)**2) + mu1**2)
        self.assertAlmostEqual(gr_traj1_sys1, gr_traj1_sys1_true, places = 6)
        self.assertAlmostEqual(gr_traj2_sys1, gr_traj2_sys1_true, places = 6)

    def test_global_residual_grad(self):
        # generating random frequencies and system parameters
        freq1 = rand.uniform(-10, 10)
        freq2 = rand.uniform(-10, 10)
        mu1 = rand.uniform(0, 10)
        mu2 = rand.uniform(0, 10)
        r = rand.uniform(0, 10)

        # apply parameters
        self.sys1.parameters['mu'] = mu1
        self.sys2.parameters['mu'] = mu2
        self.sys2.parameters['r'] = r

        # calculate local residuals
        H_n_inv_t1s1 = init_H_n_inv(self.traj1, self.sys1, freq1, self.mean1)
        H_n_inv_t2s1 = init_H_n_inv(self.traj2, self.sys1, freq2, self.mean2)
        lr_t1s1 = res_funcs.local_residual(self.cache1, self.sys1, H_n_inv_t1s1, self.plans_t1)
        lr_t2s1 = res_funcs.local_residual(self.cache2, self.sys1, H_n_inv_t2s1, self.plans_t2)

        # calculate global residual gradients
        gr_grad_traj_t1s1 = res_funcs.gr_traj_grad(self.cache1, self.sys1, freq1, self.mean1, self.plans_t1)
        gr_grad_traj_t2s1 = res_funcs.gr_traj_grad(self.cache2, self.sys1, freq2, self.mean2, self.plans_t2)
        gr_grad_freq_t1s1 = res_funcs.gr_freq_grad(self.traj1, lr_t1s1)
        gr_grad_freq_t2s1 = res_funcs.gr_freq_grad(self.traj2, lr_t2s1)

        # outputs are numbers
        temp_freq = True
        if type(gr_grad_freq_t1s1) != np.float64 and type(gr_grad_freq_t1s1) != float:
            temp_freq = False
        if type(gr_grad_freq_t2s1) != np.float64 and type(gr_grad_freq_t2s1) != float:
            temp_freq = False
        self.assertTrue(temp_freq)

        # correct values (compared with FD approximation)
        gr_grad_traj_t1s1_FD = self.gen_gr_grad_FD(self.traj1, self.sys1, freq1, self.mean1, self.plans_t1)
        gr_grad_traj_t2s1_FD = self.gen_gr_grad_FD(self.traj2, self.sys1, freq2, self.mean2, self.plans_t2)

        self.assertAlmostEqual(gr_grad_traj_t1s1, gr_grad_traj_t1s1_FD, places = 3)
        self.assertAlmostEqual(gr_grad_traj_t2s1, gr_grad_traj_t2s1_FD, places = 3)

    @staticmethod
    def gen_gr_grad_FD(traj, sys, freq, mean, fftplans, step = 1e-9):
        """
            This function uses finite differencing to compute the gradients of
            the global residual for all the DoFs (the discrete time coordinated
            and the frequency).
        """
        # initialise arrays
        gr_grad_FD_traj_real = np.zeros(traj.shape)
        gr_grad_FD_traj_imag = np.zeros(traj.shape)
        cache = Cache(traj, mean, sys, fftplans)

        # generate resolvent trajectory
        H_n_inv = init_H_n_inv(traj, sys, freq, mean)

        # loop over trajectory DoFs and use CD scheme
        for i in range(traj.shape[0]):
            for j in range(traj.shape[1]):
                for k in range(2):
                    if k == 1:
                        step2 = 1j*step
                    else:
                        step2 = step
                    traj_for = traj
                    traj_for[i, j] = traj[i, j] + step2
                    _ = res_funcs.local_residual(cache, sys, H_n_inv, fftplans)
                    gr_traj_for = res_funcs.global_residual(cache)
                    traj_back = traj
                    traj_back[i, j] = traj[i, j] - step2
                    _ = res_funcs.local_residual(cache, sys, H_n_inv, fftplans)
                    gr_traj_back = res_funcs.global_residual(cache)
                    if k == 0:
                        gr_grad_FD_traj_real[i, j] = (gr_traj_for - gr_traj_back)/(2*step)
                    else:
                        gr_grad_FD_traj_imag[i, j] = (gr_traj_for - gr_traj_back)/(2*step)

        return pyReSolver.Trajectory(gr_grad_FD_traj_real + 1j*gr_grad_FD_traj_imag)


if __name__ == "__main__":
    unittest.main()
