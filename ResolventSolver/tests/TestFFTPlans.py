# This file contains the unit tests for the FFT plans class.

import unittest
import random as rand

import numpy as np

from ResolventSolver.FFTPlans import FFTPlans
from ResolventSolver.trajectory_definitions import unit_circle as uc
from ResolventSolver.trajectory_definitions import ellipse as elps

# The following tests will be performed:
#   - trig functions are correct
#   - delta functions are correct

class TestFFTPlans(unittest.TestCase):

    def setUp(self):
        Nt = rand.randint(3, 64)
        dim = 3
        self.shape = [Nt, dim]
        self.shapef = [(Nt >> 1) + 1, dim]
        self.flag = 'FFTW_ESTIMATE'

    def tearDown(self):
        del self.shape
        del self.shapef
        del self.flag

    def test_init(self):
        plans = FFTPlans(self.shape, flag = self.flag)
        self.assertIsInstance(plans, FFTPlans)
        self.assertEqual(plans.tmp_f.shape[0], (plans.tmp_t.shape[0] >> 1) + 1)
        self.assertEqual(plans.tmp_f.shape[1], plans.tmp_t.shape[1])
        self.assertTrue(plans.fftplan.input_array is plans.ifftplan.output_array)
        self.assertTrue(plans.ifftplan.input_array is plans.fftplan.output_array)

    def test_random(self):
        randt = np.random.rand(*self.shape)
        randf = np.random.rand(*self.shapef) + 1j*np.random.rand(*self.shapef)
        randf[0] = np.real(randf[0])
        if self.shape[0] % 2 == 0:
            randf[-1] = np.real(randf[-1])
        plans = FFTPlans(self.shape, flag = self.flag)
        plans.tmp_t = np.copy(randt)
        plans.fft()
        self.assertTrue(np.allclose(plans.tmp_f, np.fft.rfft(randt, axis = 0)/randt.shape[0]))
        plans.tmp_f = np.copy(randf)
        plans.ifft()
        if self.shape[0] % 2 == 0:
            self.assertTrue(np.allclose(plans.tmp_t, np.fft.irfft(randf*2*(randf.shape[0] - 1), axis = 0)))
        else:
            self.assertTrue(np.allclose(plans.tmp_t, np.fft.irfft(randf*(2*randf.shape[0] - 1), 2*randf.shape[0] - 1, axis = 0)))
        plans.tmp_t = np.copy(randt)
        plans.fft()
        plans.ifft()
        self.assertTrue(np.allclose(plans.tmp_t, randt))
        plans.tmp_f = np.copy(randf)
        plans.ifft()
        plans.fft()
        self.assertTrue(np.allclose(plans.tmp_f, randf))

    def est_trig(self):
        pass

    def est_delta(self):
        pass


if __name__ == '__main__':
    unittest.main()