# This file contains the unit tests for the Hessian matrix generation around a
# minimum.

import unittest
import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator

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

class TestSparseLinalgEigs(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_diag_eigs(self):
        pass

    def test_diag_symmetric(self):
        pass

    def test_diag_dot_test(self):
        pass


if __name__ == '__main__':
    unittest.main()
