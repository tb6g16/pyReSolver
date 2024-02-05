# This file contains the unit tests for the FFT plans class.

import unittest
import random as rand

import numpy as np

from pyReSolver.FFTPlans import FFTPlans
from pyReSolver.utils import func2curve

from tests.test_trajectories import unit_circle as uc

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
        randf[0] = np.real(randf[0]) # type: ignore
        if self.shape[0] % 2 == 0:
            randf[-1] = np.real(randf[-1]) # type: ignore
        plans = FFTPlans(self.shape, flag = self.flag)
        plans.fft(randf, randt)
        self.assertTrue(np.allclose(randf, np.fft.rfft(randt, axis = 0)/np.shape(randt)[0]))
        plans.ifft(randf, randt)
        if self.shape[0] % 2 == 0:
            self.assertTrue(np.allclose(randt, np.fft.irfft(randf*2*(np.shape(randf)[0] - 1), axis = 0)))
        else:
            self.assertTrue(np.allclose(randt, np.fft.irfft(randf*(2*np.shape(randf)[0] - 1), 2*np.shape(randf)[0] - 1, axis = 0)))
        tmp_t = np.zeros_like(randt)
        plans.fft(randf, randt)
        plans.ifft(randf, tmp_t)
        self.assertTrue(np.allclose(tmp_t, randt))
        tmp_f = np.zeros_like(randf)
        plans.ifft(randf, randt)
        plans.fft(tmp_f, randt)
        self.assertTrue(np.allclose(tmp_f, randf))

    def test_trig(self):
        uc_t = func2curve(uc, self.shape[0], if_freq = False)
        uc_f_true = np.zeros([self.shapef[0], 2], dtype = complex)
        uc_f = np.zeros_like(uc_f_true)
        uc_f_true[1, 0] = 0.5
        uc_f_true[1, 1] = 1j*0.5
        plans = FFTPlans([self.shape[0], 2], flag = self.flag)
        plans.fft(uc_f, uc_t)
        self.assertTrue(np.allclose(uc_f, uc_f_true))
        tmp_t = np.zeros_like(uc_t)
        plans.ifft(uc_f, tmp_t)
        self.assertTrue(np.allclose(tmp_t, uc_t))

    def test_delta(self):
        delta_t = np.zeros(self.shape)
        delta_t[0, 0] = 1.0
        delta_t[1, 1] = 1.0
        delta_t[2, 2] = 1.0
        delta_f_true = np.fft.rfft(delta_t, axis = 0)/delta_t.shape[0]
        delta_f = np.zeros_like(delta_f_true)
        plans = FFTPlans(self.shape, flag = self.flag)
        plans.fft(delta_f, delta_t)
        self.assertTrue(np.allclose(delta_f, delta_f_true))
        tmp_t = np.zeros_like(delta_t)
        plans.ifft(delta_f, tmp_t)
        self.assertTrue(np.allclose(tmp_t, delta_t))


if __name__ == '__main__':
    unittest.main()