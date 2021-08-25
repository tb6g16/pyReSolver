# This file contains the unit tests for the Hessian matrix generation around a
# minimum.

import unittest
import numpy as np
import scipy

import traj_hdf5
from Trajectory import Trajectory
from hessian import hess
from systems import lorenz

# The approximation only works when the trajectory is close to a minimum already,
# so there are two possible ways to ensure that:
#   1: use a UPO converged by Davide;
#   2: converge my own.

class TestHessian(unittest.TestCase):

    def setUp(self):
        upo, freq = traj_hdf5.read_traj_davide('upo01.orb')
        upo = Trajectory(upo[:20])
        mean = list(np.real(upo[0]))
        upo[0] = 0
        self.hess = hess(upo, freq, lorenz, mean)

    def tearDown(self):
        del self.hess

    def test_hessian_symmetric(self):
        # is hessian invariant under transpose operation
        # THE PROBLEM IS THE ARTIFITIAL SETTING TO ZERO! OR MAYBE THE ZERO FREQUENCY GRADIENT?
        self.assertTrue(np.allclose(self.hess, np.transpose(self.hess)))
        # ONLY SYMMETRIC TO LOW A TOLERANCE, GETS WORSE AS NUMBER OF MODES IS INCREASED

    def est_hessian_positive_definite(self):
        # evaluate eigenvalues
        hess_spectra = scipy.linalg.eigvals(self.hess)

        # check if eigenvalues are all positive
        for i in range(np.shape(hess_spectra)[0]):
            self.assertTrue(np.real(hess_spectra[i]) > 0)


if __name__ == '__main__':
    unittest.main()
