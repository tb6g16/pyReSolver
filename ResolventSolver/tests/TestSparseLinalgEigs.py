# This file contains the unit tests for the Hessian matrix generation around a
# minimum.

import unittest
import random as rand

import numpy as np
import scipy.sparse.linalg as sparse

class RandomDiagonal(sparse.LinearOperator):
    
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
        N = rand.randint(3, 50)
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
        self.assertEqual(lhs, rhs)

    def test_diag_eigs(self):
        # how many eigenvalues to find
        no_eigs = rand.randint(1, self.diag.shape[0] - 2)

        # find largest
        evals_large, _ = sparse.eigs(self.diag, k = no_eigs, which = 'LM')

        # find smallest with shift invert
        evals_small, _ = sparse.eigs(self.diag, k = no_eigs, which = 'LM', sigma = 0)

        # are they correct
        self.assertTrue(np.allclose(np.sort(self.diag.diag_vec)[self.diag.shape[0] - no_eigs:], np.sort(evals_large)))
        self.assertTrue(np.allclose(np.sort(self.diag.diag_vec)[:no_eigs], np.sort(evals_small)))

    def test_diag_eigsh(self):
        # how many eigenvalues to find
        no_eigs = rand.randint(1, self.diag.shape[0] - 1)

        # find largest
        evals_large, _ = sparse.eigsh(self.diag, k = no_eigs, which = 'LM')

        # find smallest with shift invert
        evals_small, _ = sparse.eigsh(self.diag, k = no_eigs, which = 'LM', sigma = 0)

        # are they correct
        self.assertTrue(np.allclose(np.sort(self.diag.diag_vec)[self.diag.shape[0] - no_eigs:], np.sort(evals_large)))
        self.assertTrue(np.allclose(np.sort(self.diag.diag_vec)[:no_eigs], np.sort(evals_small)))


if __name__ == '__main__':
    unittest.main()
