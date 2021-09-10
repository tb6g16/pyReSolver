# This file contains the unit tests for the Hessian matrix generation around a
# minimum.

import unittest
import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator
import random as rand

import traj_hdf5
from Trajectory import Trajectory
from hessian import hess
from systems import lorenz

class RandomDiagonal(LinearOperator):
    
    def __init__(self, N):
        self.diag_vec = np.random.rand(N)
        self.shape = (N, N)
        self.dtype = np.dtype('float64')

    def _matvec(self, v):
        return self.diag_vec*v

    def _rmatvec(self, v):
        return self.diag_vec*v

    @property
    def diag_matrix(self):
        return np.diag(self.diag_vec)


class TestSparseLinalgEigs(unittest.TestCase):

    def setUp(self):
        # N = rand.randint(2, 50)
        N = 5
        self.diag = RandomDiagonal(N)

    def tearDown(self):
        del self.diag

    def test_diag_symmetric(self):
        # initialise random vector
        v = np.random.rand(self.diag.shape[0])

        # matvec and rmatvec produce the same result
        matvec = self.diag._matvec(v)
        rmatvec = self.diag._rmatvec(v)

        # are they the same?
        self.assertTrue(np.array_equal(matvec, rmatvec))

    def test_diag_dot_test(self):
        # initialise two random vectors#
        u = np.random.rand(self.diag.shape[0])
        v = np.random.rand(self.diag.shape[0])

        # perform each side of the dot test
        lhs = np.dot(self.diag.matvec(u), v)
        rhs = np.dot(u, self.diag.rmatvec(v))

        # are they equal?
        self.assertTrue(np.allclose(lhs, rhs))

    def test_diag_eigs(self):
        pass


if __name__ == '__main__':
    unittest.main()
