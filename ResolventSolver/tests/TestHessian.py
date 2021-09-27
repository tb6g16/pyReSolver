# This file contains the unit tests for the Hessian matrix generation around a
# minimum.

import unittest

import numpy as np

from ResolventSolver.HessianOperator import HessianOperator
from ResolventSolver.gen_rand_traj import gen_rand_traj
from ResolventSolver.my_min import my_min
from ResolventSolver.systems import lorenz

class TestHessian(unittest.TestCase):

    T = 1.55
    traj = gen_rand_traj(3, 15)
    freq = (2*np.pi)/T
    mean = np.array([[0, 0, 23.64]])
    traj, _, _, _ = my_min(traj, freq, lorenz, mean, method = 'CG', quiet = True)

    def setUp(self):
        self.hess_operator = HessianOperator(self.traj, lorenz, self.freq, self.mean)

    def tearDown(self):
        del self.hess_operator

    def test_hessian_symmetric(self):
        # initialise random vector
        v = np.random.rand(self.hess_operator.shape[0] + 1)

        # matvec and rmatvec produce the same result
        matvec = self.hess_operator._matvec(v)
        rmatvec = self.hess_operator._rmatvec(v)

        # are they the same?
        self.assertTrue(np.array_equal(matvec, rmatvec))

    def test_hessian_dot_test(self):
        # initialise two random vectors#
        u = np.random.rand(self.hess_operator.shape[0] + 1)
        v = np.random.rand(self.hess_operator.shape[0] + 1)

        # perform each side of the dot test
        lhs = np.dot(self.hess_operator.matvec(u), v)
        rhs = np.dot(u, self.hess_operator.rmatvec(v))

        # are they equal?
        self.assertTrue(np.allclose(lhs, rhs))

    def test_hessian_positive_definite(self):
        pass


if __name__ == '__main__':
    unittest.main()
