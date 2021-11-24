# This file contains the unit test for the optimise file that initialises the
# objective function, constraints, and all their associated gradients

import unittest
import random as rand

import numpy as np

from ResolventSolver.FFTPlans import FFTPlans
from ResolventSolver.Trajectory import Trajectory
from ResolventSolver.traj2vec import traj2vec, vec2traj, init_comp_vec
from ResolventSolver.my_min import init_opt_funcs
import ResolventSolver.residual_functions as res_funcs
from ResolventSolver.systems import van_der_pol as vpd
from ResolventSolver.systems import lorenz

class TestMyMin(unittest.TestCase):

    def setUp(self):
        modes = rand.randint(3, 65)

        temp1 = np.random.rand(modes, 2) + 1j*np.random.rand(modes, 2)
        temp1[0] = 0
        temp1[-1] = 0
        self.traj1 = Trajectory(temp1)
        self.plan_t1 = FFTPlans([(self.traj1.shape[0] - 1) << 1, self.traj1.shape[1]], flag = 'FFTW_ESTIMATE')
        self.freq1 = rand.uniform(0, 10)
        self.traj1_vec = init_comp_vec(self.traj1)
        traj2vec(self.traj1, self.traj1_vec)

        temp2 = np.random.rand(modes, 2) + 1j*np.random.rand(modes, 2)
        temp2[0] = 0
        temp2[-1] = 0
        self.traj2 = Trajectory(temp2)
        self.plan_t2 = FFTPlans([(self.traj2.shape[0] - 1) << 1, self.traj2.shape[1]], flag = 'FFTW_ESTIMATE')
        self.freq2 = rand.uniform(0, 10)
        self.traj2_vec = init_comp_vec(self.traj2)
        traj2vec(self.traj2, self.traj2_vec)

        temp3 = np.random.rand(modes, 3) + 1j*np.random.rand(modes, 3)
        temp3[0] = 0
        temp3[-1] = 0
        self.traj3 = Trajectory(temp3)
        self.plan_t3 = FFTPlans([(self.traj3.shape[0] - 1) << 1, self.traj3.shape[1]], flag = 'FFTW_ESTIMATE')
        self.freq3 = rand.uniform(0, 10)
        self.traj3_vec = init_comp_vec(self.traj3)
        traj2vec(self.traj3, self.traj3_vec)

        self.sys1 = vpd
        self.mean1 = np.random.rand(1, 2)
        self.sys2 = lorenz
        self.mean2 = np.random.rand(1, 3)

    def tearDown(self):
        del self.traj1
        del self.plan_t1
        del self.freq1
        del self.traj1_vec
        del self.traj2
        del self.plan_t2
        del self.freq2
        del self.traj2_vec
        del self.traj3
        del self.plan_t3
        del self.freq3
        del self.traj3_vec
        del self.sys1
        del self.mean1
        del self.sys2
        del self.mean2

    def test_traj_global_res(self):
        res_func_t1s1, _ = init_opt_funcs(self.traj1, self.freq1, self.plan_t1, self.sys1, self.mean1)
        res_func_t2s1, _ = init_opt_funcs(self.traj2, self.freq2, self.plan_t2, self.sys1, self.mean1)
        res_func_t3s2, _ = init_opt_funcs(self.traj3, self.freq3, self.plan_t3, self.sys2, self.mean2)
        gr_t1s1 = res_func_t1s1(self.traj1_vec)
        gr_t2s1 = res_func_t2s1(self.traj2_vec)
        gr_t3s2 = res_func_t3s2(self.traj3_vec)

        # correct value
        H_n_inv_t1s1 = res_funcs.init_H_n_inv(self.traj1, self.sys1, self.freq1, self.mean1)
        H_n_inv_t2s1 = res_funcs.init_H_n_inv(self.traj2, self.sys1, self.freq2, self.mean1)
        H_n_inv_t3s2 = res_funcs.init_H_n_inv(self.traj3, self.sys2, self.freq3, self.mean2)
        lr_t1s1_true = res_funcs.local_residual(self.traj1, self.sys1, self.mean1, H_n_inv_t1s1, self.plan_t1)
        lr_t2s1_true = res_funcs.local_residual(self.traj2, self.sys1, self.mean1, H_n_inv_t2s1, self.plan_t2)
        lr_t3s2_true = res_funcs.local_residual(self.traj3, self.sys2, self.mean2, H_n_inv_t3s2, self.plan_t3)
        gr_t1s1_true = res_funcs.global_residual(lr_t1s1_true)
        gr_t2s1_true = res_funcs.global_residual(lr_t2s1_true)
        gr_t3s2_true = res_funcs.global_residual(lr_t3s2_true)
        self.assertEqual(gr_t1s1, gr_t1s1_true)
        self.assertEqual(gr_t2s1, gr_t2s1_true)
        self.assertEqual(gr_t3s2, gr_t3s2_true)

    # FIXME: problem due to first node not artificially being set to zero
    def test_traj_global_res_jac(self):
        tmp_fun1, res_grad_func_t1s1 = init_opt_funcs(self.traj1, self.freq1, self.plan_t1, self.sys1, self.mean1)
        tmp_fun2, res_grad_func_t2s1 = init_opt_funcs(self.traj2, self.freq2, self.plan_t2, self.sys1, self.mean1)
        tmp_fun3, res_grad_func_t3s2 = init_opt_funcs(self.traj3, self.freq3, self.plan_t3, self.sys2, self.mean2)
        gr_traj_t1s1 = np.zeros_like(self.traj1)
        gr_traj_t2s1 = np.zeros_like(self.traj2)
        gr_traj_t3s2 = np.zeros_like(self.traj3)
        vec2traj(gr_traj_t1s1, res_grad_func_t1s1(self.traj1_vec))
        vec2traj(gr_traj_t2s1, res_grad_func_t2s1(self.traj2_vec))
        vec2traj(gr_traj_t3s2, res_grad_func_t3s2(self.traj3_vec))

        # correct values
        H_n_inv_t1s1 = res_funcs.init_H_n_inv(self.traj1, self.sys1, self.freq1, self.mean1)
        H_n_inv_t2s1 = res_funcs.init_H_n_inv(self.traj2, self.sys1, self.freq2, self.mean1)
        H_n_inv_t3s2 = res_funcs.init_H_n_inv(self.traj3, self.sys2, self.freq3, self.mean2)
        lr_t1s1_true = res_funcs.local_residual(self.traj1, self.sys1, self.mean1, H_n_inv_t1s1, self.plan_t1)
        lr_t2s1_true = res_funcs.local_residual(self.traj2, self.sys1, self.mean1, H_n_inv_t2s1, self.plan_t2)
        lr_t3s2_true = res_funcs.local_residual(self.traj3, self.sys2, self.mean2, H_n_inv_t3s2, self.plan_t3)
        tmp1 = init_comp_vec(self.traj1)
        tmp2 = init_comp_vec(self.traj2)
        tmp3 = init_comp_vec(self.traj3)
        gr_traj_t1s1_true = res_funcs.gr_traj_grad(self.traj1, self.sys1, self.freq1, self.mean1, lr_t1s1_true, self.plan_t1)
        gr_traj_t2s1_true = res_funcs.gr_traj_grad(self.traj2, self.sys1, self.freq2, self.mean1, lr_t2s1_true, self.plan_t2)
        gr_traj_t3s2_true = res_funcs.gr_traj_grad(self.traj3, self.sys2, self.freq3, self.mean2, lr_t3s2_true, self.plan_t3)
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