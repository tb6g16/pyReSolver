# This file contains the class definition to hold the FFTW plans required to
# efficiently transform between physical and spectral space of a state-space
# trajectory.

import pyfftw

class FFTPlans:

    __slots__ = ['tmp_t', 'tmp_f', 'fftplan', 'ifftplan']

    def __init__(shape, flag = 'FFTW_EXHAUSTIVE'):
        self.tmp_t = pyfftw.empty_aligned(shape, dtype = 'float64')
        self.tmp_f = pyfftw.empty_aligned([(shape[0] >> 1) + 1, shape[1]], dtype = 'complex128')
        self.fftplan = pyfftw.FFTW(tmp_t, tmp_f, axes = (0,), direction = 'FFTW_FORWARD', flags = (flag,))
        self.ifftplan = pyfftw.FFTW(tmp_f, tmp_t, axes = (0,), direction = 'FFTW_BACKWARD')

    def fft(self):
        self.fftplan()

    def ifft(self):
        self.ifftplan()
