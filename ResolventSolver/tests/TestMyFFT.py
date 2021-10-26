# This file contains the unit tests for my_fft functions.

import unittest
import random as rand

import numpy as np

from ResolventSolver.my_fft import my_fft, my_ifft, my_rfft, my_irfft
from ResolventSolver.traj_util import func2curve
from ResolventSolver.trajectory_definitions import unit_circle as uc
from ResolventSolver.trajectory_definitions import ellipse as elps

class TestMyFFT(unittest.TestCase):

    def setUp(self):
        self.modes = rand.randint(2, 256)
        dim = rand.randint(1, 5)
        array1_time = func2curve(uc.x, self.modes)
        self.array1 = np.fft.fft(array1_time, axis = 0)/np.shape(array1_time)[0]
        array2_time = func2curve(elps.x, self.modes)
        self.array2 = np.fft.fft(array2_time, axis = 0)/np.shape(array2_time)[0]
        self.array_rand = np.random.rand(self.modes, dim) + 1j*np.random.rand(self.modes, dim)
        self.rand_time = np.random.rand(rand.randrange(3, 511, 2), dim)

    def tearDown(self):
        del self.modes
        del self.array1
        del self.array2
        del self.array_rand
        del self.rand_time

    def test_my_fft_functions(self):
        # do the random noises convert back and forth properly
        self.assertTrue(np.allclose(self.array_rand, my_fft(my_ifft(self.array_rand))))
        self.assertTrue(np.allclose(self.rand_time, my_ifft(my_fft(self.rand_time))))

        # # correct modes for unit circle
        array1_modes_true = np.zeros_like(self.array1)
        array1_modes_true[1, 0] = 0.5
        array1_modes_true[1, 1] = 1j*0.5
        array1_modes_true[-1, 0] = 0.5
        array1_modes_true[-1, 1] = -1j*0.5
        self.assertTrue(np.allclose(self.array1, array1_modes_true))

        # # correct values for ellipse
        array2_modes_true = np.zeros_like(self.array2)
        array2_modes_true[1, 0] = 1
        array2_modes_true[1, 1] = 1j*0.5
        array2_modes_true[-1, 0] = 1
        array2_modes_true[-1, 1] = -1j*0.5
        self.assertTrue(np.allclose(self.array2, array2_modes_true))

    def test_my_rfft_functions(self):
        # convert random array to maintain rfft symmetries
        self.array_rand[0] = np.real(self.array_rand[0])
        self.array_rand[-1] = np.real(self.array_rand[-1])

        # do the random noises convert back and forth properly
        self.assertTrue(np.allclose(self.array_rand, my_rfft(my_irfft(self.array_rand))))
        self.assertTrue(np.allclose(self.rand_time, my_irfft(my_rfft(self.rand_time))))

        # correct modes for unit circle
        array1_modes_true = np.zeros(np.shape(self.array1[:self.modes, :]), dtype = complex)
        array1_modes_true[1, 0] = 0.5
        array1_modes_true[1, 1] = 1j*0.5
        self.assertTrue(np.allclose(self.array1[:self.modes, :], array1_modes_true))

        # correct values for ellipse
        array2_modes_true = np.zeros(np.shape(self.array1[:self.modes, :]), dtype = complex)
        array2_modes_true[1, 0] = 1
        array2_modes_true[1, 1] = 1j*0.5
        self.assertTrue(np.allclose(self.array2[:self.modes, :], array2_modes_true))


if __name__ == "__main__":
    unittest.main()
