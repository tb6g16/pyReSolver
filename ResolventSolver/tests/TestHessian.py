# This file contains the unit tests for the Hessian matrix generation around a
# minimum.

from os.path import exists
import unittest

import numpy as np

from ResolventSolver.HessianOperator import HessianOperator
from ResolventSolver.gen_rand_traj import gen_rand_traj
from ResolventSolver.my_min import my_min
from ResolventSolver.systems import lorenz
from ResolventSolver.traj_hdf5 import write_traj, read_traj

def get_opt(T, filename = 'hess_traj.hdf5'):
    if exists('./' + filename):
        traj, freq = read_traj(filename)
    else:
        traj = gen_rand_traj(3, 10*T)
        freq = (2*np.pi)/T
        mean = np.array([[0, 0, 23.64]])
        traj, _, _ = my_min(traj, freq, lorenz, mean, method = 'CG')
        write_traj(filename, traj, freq)

    return traj, freq

# MAY NEED TO CHANGE FFT TYPE TO GET PROPER CONVERGENCE TO MINIMUM
class TestHessian(unittest.TestCase):

    def setUp(self):
        traj, freq = get_opt(1.55)
        self.hess_operator = HessianOperator(traj, lorenz, freq, np.array([[0, 0, 23.64]]))

    def tearDown(self):
        del self.hess_operator

    def test_hessian_symmetric(self):
        # initialise random vector
        v = np.random.rand(self.hess_operator.shape[0])

        # matvec and rmatvec produce the same result
        matvec = self.hess_operator._matvec(v)
        rmatvec = self.hess_operator._rmatvec(v)

        # are they the same?
        self.assertTrue(np.array_equal(matvec, rmatvec))

    # THIS TEST FAILING IMPLIES THE HESSIAN OPERATOR IS NOT SYMMETRIC
    def test_hessian_dot_test(self):
        # initialise two random vectors
        # NOTE: small perturbation
        u = np.random.rand(self.hess_operator.shape[0])*1e-6
        v = np.random.rand(self.hess_operator.shape[0])*1e-6

        # perform each side of the dot test
        lhs = np.dot(self.hess_operator.matvec(u), v)
        rhs = np.dot(u, self.hess_operator.rmatvec(v))

        # are they equal?
        self.assertEqual(lhs, rhs)

    def est_hessian_positive_definite(self):
        pass


if __name__ == '__main__':
    unittest.main()
