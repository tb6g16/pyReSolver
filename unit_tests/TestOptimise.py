# This file contains the unit test for the optimise file that initialises the
# objective function, constraints, and all their associated gradients

import sys
sys.path.append(r"C:\Users\user\Desktop\PhD\Bruno Paper\Code\RA Dynamical System")
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

class TestOptimise(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_traj_global_res(self):
        pass

    def test_traj_global_res_jac(self):
        pass

    def test_constraints(self):
        pass

    def test_constraints_grad(self):
        pass


if __name__ == "__main__":
    unittest.main()