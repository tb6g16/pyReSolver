# This file contains the unit test for the optimise file that initialises the
# objective function, constraints, and all their associated gradients

import unittest
import random as rand

import numpy as np

import pyReSolver

from pyReSolver.traj2vec import init_comp_vec, traj2vec, vec2traj
from pyReSolver.Cache import Cache
from pyReSolver.init_opt_funcs import init_opt_funcs
import pyReSolver.residual_functions as res_funcs

def init_H_n_inv(traj, sys, freq, mean):
    jac_at_mean = sys.jacobian(mean)
    return pyReSolver.resolvent_inv(traj.shape[0], freq, jac_at_mean)

class TestInitOptFuncs(unittest.TestCase):

    def setUp(self):
        self.sys1 = pyReSolver.systems.van_der_pol
        self.sys2 = pyReSolver.systems.lorenz

        modes = rand.randint(3, 65)

        temp1 = np.random.rand(modes, 2) + 1j*np.random.rand(modes, 2)
        temp1[0] = 0
        temp1[-1] = 0
        self.traj1 = pyReSolver.Trajectory(temp1)
        self.mean1 = np.random.rand(1, 2)
        self.plan_t1 = pyReSolver.FFTPlans([(self.traj1.shape[0] - 1) << 1, self.traj1.shape[1]], flag = 'FFTW_ESTIMATE')
        self.freq1 = rand.uniform(0, 10)
        self.traj1_vec = init_comp_vec(self.traj1)
        traj2vec(self.traj1, self.traj1_vec)
        self.cache1 = Cache(self.traj1, self.mean1, self.sys1, self.plan_t1)

        temp2 = np.random.rand(modes, 2) + 1j*np.random.rand(modes, 2)
        temp2[0] = 0
        temp2[-1] = 0
        self.traj2 = pyReSolver.Trajectory(temp2)
        self.mean2 = np.random.rand(1, 2)
        self.plan_t2 = pyReSolver.FFTPlans([(self.traj2.shape[0] - 1) << 1, self.traj2.shape[1]], flag = 'FFTW_ESTIMATE')
        self.freq2 = rand.uniform(0, 10)
        self.traj2_vec = init_comp_vec(self.traj2)
        traj2vec(self.traj2, self.traj2_vec)
        self.cache2 = Cache(self.traj2, self.mean2, self.sys1, self.plan_t2)

        temp3 = np.random.rand(modes, 3) + 1j*np.random.rand(modes, 3)
        temp3[0] = 0
        temp3[-1] = 0
        self.traj3 = pyReSolver.Trajectory(temp3)
        self.mean3 = np.random.rand(1, 3)
        self.plan_t3 = pyReSolver.FFTPlans([(self.traj3.shape[0] - 1) << 1, self.traj3.shape[1]], flag = 'FFTW_ESTIMATE')
        self.freq3 = rand.uniform(0, 10)
        self.traj3_vec = init_comp_vec(self.traj3)
        traj2vec(self.traj3, self.traj3_vec)
        self.cache3 = Cache(self.traj3, self.mean3, self.sys2, self.plan_t3)

    def tearDown(self):
        del self.sys1
        del self.sys2
        del self.traj1
        del self.mean1
        del self.plan_t1
        del self.freq1
        del self.traj1_vec
        del self.cache1
        del self.traj2
        del self.mean2
        del self.plan_t2
        del self.freq2
        del self.traj2_vec
        del self.cache2
        del self.traj3
        del self.mean3
        del self.plan_t3
        del self.freq3
        del self.traj3_vec
        del self.cache3

    def test_traj_global_res(self):
        res_func_t1s1, _ = init_opt_funcs(self.cache1, self.freq1, self.plan_t1, self.sys1, self.mean1)
        res_func_t2s1, _ = init_opt_funcs(self.cache2, self.freq2, self.plan_t2, self.sys1, self.mean2)
        res_func_t3s2, _ = init_opt_funcs(self.cache3, self.freq3, self.plan_t3, self.sys2, self.mean3)
        gr_t1s1 = res_func_t1s1(self.traj1_vec)
        gr_t2s1 = res_func_t2s1(self.traj2_vec)
        gr_t3s2 = res_func_t3s2(self.traj3_vec)

        # correct value
        H_n_inv_t1s1 = init_H_n_inv(self.traj1, self.sys1, self.freq1, self.mean1)
        H_n_inv_t2s1 = init_H_n_inv(self.traj2, self.sys1, self.freq2, self.mean2)
        H_n_inv_t3s2 = init_H_n_inv(self.traj3, self.sys2, self.freq3, self.mean3)
        res_funcs.local_residual(self.cache1, self.sys1, H_n_inv_t1s1, self.plan_t1)
        res_funcs.local_residual(self.cache2, self.sys1, H_n_inv_t2s1, self.plan_t2)
        res_funcs.local_residual(self.cache3, self.sys2, H_n_inv_t3s2, self.plan_t3)
        gr_t1s1_true = res_funcs.global_residual(self.cache1)
        gr_t2s1_true = res_funcs.global_residual(self.cache2)
        gr_t3s2_true = res_funcs.global_residual(self.cache3)
        self.assertEqual(gr_t1s1, gr_t1s1_true)
        self.assertEqual(gr_t2s1, gr_t2s1_true)
        self.assertEqual(gr_t3s2, gr_t3s2_true)

    def test_traj_global_res_jac(self):
        tmp_fun1, res_grad_func_t1s1 = init_opt_funcs(self.cache1, self.freq1, self.plan_t1, self.sys1, self.mean1)
        tmp_fun2, res_grad_func_t2s1 = init_opt_funcs(self.cache2, self.freq2, self.plan_t2, self.sys1, self.mean2)
        tmp_fun3, res_grad_func_t3s2 = init_opt_funcs(self.cache3, self.freq3, self.plan_t3, self.sys2, self.mean3)
        gr_traj_t1s1 = np.zeros_like(self.traj1)
        gr_traj_t2s1 = np.zeros_like(self.traj2)
        gr_traj_t3s2 = np.zeros_like(self.traj3)
        tmp_fun1(self.traj1_vec)
        tmp_fun2(self.traj2_vec)
        tmp_fun3(self.traj3_vec)
        vec2traj(gr_traj_t1s1, res_grad_func_t1s1(self.traj1_vec))
        vec2traj(gr_traj_t2s1, res_grad_func_t2s1(self.traj2_vec))
        vec2traj(gr_traj_t3s2, res_grad_func_t3s2(self.traj3_vec))

        # correct values
        H_n_inv_t1s1 = init_H_n_inv(self.traj1, self.sys1, self.freq1, self.mean1)
        H_n_inv_t2s1 = init_H_n_inv(self.traj2, self.sys1, self.freq2, self.mean2)
        H_n_inv_t3s2 = init_H_n_inv(self.traj3, self.sys2, self.freq3, self.mean3)
        res_funcs.local_residual(self.cache1, self.sys1, H_n_inv_t1s1, self.plan_t1)
        res_funcs.local_residual(self.cache2, self.sys1, H_n_inv_t2s1, self.plan_t2)
        res_funcs.local_residual(self.cache3, self.sys2, H_n_inv_t3s2, self.plan_t3)
        gr_traj_t1s1_true = res_funcs.gr_traj_grad(self.cache1, self.sys1, self.freq1, self.mean1, self.plan_t1)
        gr_traj_t2s1_true = res_funcs.gr_traj_grad(self.cache2, self.sys1, self.freq2, self.mean2, self.plan_t2)
        gr_traj_t3s2_true = res_funcs.gr_traj_grad(self.cache3, self.sys2, self.freq3, self.mean3, self.plan_t3)
        tmp1 = init_comp_vec(self.traj1)
        tmp2 = init_comp_vec(self.traj2)
        tmp3 = init_comp_vec(self.traj3)
        traj2vec(gr_traj_t1s1_true, tmp1)
        traj2vec(gr_traj_t2s1_true, tmp2)
        traj2vec(gr_traj_t3s2_true, tmp3)
        vec2traj(gr_traj_t1s1_true, tmp1)
        vec2traj(gr_traj_t2s1_true, tmp2)
        vec2traj(gr_traj_t3s2_true, tmp3)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        gr_traj_t1s1_true[0] = 0
        gr_traj_t2s1_true[0] = 0
        gr_traj_t3s2_true[0] = 0
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.assertEqual(gr_traj_t1s1, gr_traj_t1s1_true)
        self.assertEqual(gr_traj_t2s1, gr_traj_t2s1_true)
        self.assertEqual(gr_traj_t3s2, gr_traj_t3s2_true)


if __name__ == "__main__":
    unittest.main()