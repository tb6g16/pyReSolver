# This file contains the unit test for the optimise file that initialises the
# objective function, constraints, and all their associated gradients

import sys
sys.path.append(r"C:\Users\user\Desktop\PhD\Bruno Paper\ResolventSolver")
import unittest
import numpy as np
import random as rand
from Trajectory import Trajectory
import trajectory_functions as traj_funcs
from traj2vec import traj2vec, vec2traj
from my_min import init_opt_funcs
import residual_functions as res_funcs
from traj_util import array2list
from systems import van_der_pol as vpd
from systems import lorenz

class TestOptimise(unittest.TestCase):

    def setUp(self):
        modes = rand.randint(3, 65)

        temp1 = np.random.rand(modes, 2) + 1j*np.random.rand(modes, 2)
        temp1[0] = 0
        temp1[-1] = 0
        self.traj1 = Trajectory(array2list(temp1))
        self.freq1 = rand.uniform(0, 10)
        self.traj1_vec = traj2vec(self.traj1, self.freq1)

        temp2 = np.random.rand(modes, 2) + 1j*np.random.rand(modes, 2)
        temp2[0] = 0
        temp2[-1] = 0
        self.traj2 = Trajectory(array2list(temp2))
        self.freq2 = rand.uniform(0, 10)
        self.traj2_vec = traj2vec(self.traj2, self.freq2)

        temp3 = np.random.rand(modes, 3) + 1j*np.random.rand(modes, 3)
        temp3[0] = 0
        temp3[-1] = 0
        self.traj3 = Trajectory(array2list(temp3))
        self.freq3 = rand.uniform(0, 10)
        self.traj3_vec = traj2vec(self.traj3, self.freq3)

        self.sys1 = vpd
        self.mean1 = np.random.rand(2)
        self.sys2 = lorenz
        self.mean2 = np.random.rand(3)

    def tearDown(self):
        del self.traj1
        del self.freq1
        del self.traj1_vec
        del self.traj2
        del self.freq2
        del self.traj2_vec
        del self.traj3
        del self.freq3
        del self.traj3_vec
        del self.sys1
        del self.mean1
        del self.sys2
        del self.mean2

    def test_traj_global_res(self):
        res_func_s1, _ = init_opt_funcs(self.sys1, 2, self.mean1)
        res_func_s2, _ = init_opt_funcs(self.sys2, 3, self.mean2)
        gr_t1s1 = res_func_s1(self.traj1_vec)
        gr_t2s1 = res_func_s1(self.traj2_vec)
        gr_t3s2 = res_func_s2(self.traj3_vec)

        # correct value
        gr_t1s1_true = res_funcs.global_residual(self.traj1, self.sys1, self.freq1, self.mean1)
        gr_t2s1_true = res_funcs.global_residual(self.traj2, self.sys1, self.freq2, self.mean1)
        gr_t3s2_true = res_funcs.global_residual(self.traj3, self.sys2, self.freq3, self.mean2)
        self.assertEqual(gr_t1s1, gr_t1s1_true)
        self.assertEqual(gr_t2s1, gr_t2s1_true)
        self.assertEqual(gr_t3s2, gr_t3s2_true)

    def test_traj_global_res_jac(self):
        _, res_grad_func_s1 = init_opt_funcs(self.sys1, 2, self.mean1)
        _, res_grad_func_s2 = init_opt_funcs(self.sys2, 3, self.mean2)
        gr_traj_t1s1, gr_freq_t1s1 = vec2traj(res_grad_func_s1(self.traj1_vec), 2)
        gr_traj_t2s1, gr_freq_t2s1 = vec2traj(res_grad_func_s1(self.traj2_vec), 2)
        gr_traj_t3s2, gr_freq_t3s2 = vec2traj(res_grad_func_s2(self.traj3_vec), 3)

        # correct values
        gr_traj_t1s1_true = vec2traj(traj2vec(res_funcs.gr_traj_grad(self.traj1, self.sys1, self.freq1, self.mean1), np.nan), 2)[0]
        gr_freq_t1s1_true = res_funcs.gr_freq_grad(self.traj1, self.sys1, self.freq1, self.mean1)
        gr_traj_t2s1_true = vec2traj(traj2vec(res_funcs.gr_traj_grad(self.traj2, self.sys1, self.freq2, self.mean1), np.nan), 2)[0]
        gr_freq_t2s1_true = res_funcs.gr_freq_grad(self.traj2, self.sys1, self.freq2, self.mean1)
        gr_traj_t3s2_true = vec2traj(traj2vec(res_funcs.gr_traj_grad(self.traj3, self.sys2, self.freq3, self.mean2), np.nan), 3)[0]
        gr_freq_t3s2_true = res_funcs.gr_freq_grad(self.traj3, self.sys2, self.freq3, self.mean2)
        self.assertEqual(gr_traj_t1s1, gr_traj_t1s1_true)
        self.assertEqual(gr_traj_t2s1, gr_traj_t2s1_true)
        self.assertEqual(gr_traj_t3s2, gr_traj_t3s2_true)
        self.assertEqual(gr_freq_t1s1, gr_freq_t1s1_true)
        self.assertEqual(gr_freq_t2s1, gr_freq_t2s1_true)
        self.assertEqual(gr_freq_t3s2, gr_freq_t3s2_true)


if __name__ == "__main__":
    unittest.main()