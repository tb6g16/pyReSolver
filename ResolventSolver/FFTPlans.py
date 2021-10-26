# This file contains the class definition to hold the FFTW plans required to
# efficiently transform between physical and spectral space of a state-space
# trajectory.

import pyfftw

class FFTPlans:

    __slots__ = ['tmp_t', 'tmp_f', 'fftplan', 'ifftplan']

    def __init__(self, shape, flag = 'FFTW_EXHAUSTIVE'):
        self.tmp_t = pyfftw.empty_aligned(shape, dtype = 'float64')
        self.tmp_f = pyfftw.empty_aligned([(shape[0] >> 1) + 1, shape[1]], dtype = 'complex128')
        self.fftplan = pyfftw.FFTW(self.tmp_t, self.tmp_f, axes = (0,), direction = 'FFTW_FORWARD', flags = (flag,))
        self.ifftplan = pyfftw.FFTW(self.tmp_f, self.tmp_t, axes = (0,), direction = 'FFTW_BACKWARD', flags = (flag,))

    def fft(self):
        self.tmp_f = self.fftplan(self.tmp_t)/self.tmp_t.shape[0]

    def ifft(self):
        self.tmp_t = self.ifftplan(self.tmp_f)*self.tmp_t.shape[0]
