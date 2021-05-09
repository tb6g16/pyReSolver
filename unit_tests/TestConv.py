# This file contains the unit tests for the convolution function defined in the
# conv.py file.

import sys
sys.path.append(r"C:\Users\user\Desktop\PhD\Bruno Paper\ResolventSolver")
import unittest
import numpy as np
import random as rand
from my_fft import my_rfft, my_irfft
from conv import conv_scalar, conv_array
from trajectory_definitions import unit_circle as uc
from trajectory_definitions import ellipse as elps

class TestConv(unittest.TestCase):

    def setUp(self):
        self.modes = rand.randint(3, 50)

    def tearDown(self):
        del self.modes

    def test_conv_scalar(self):
        # initialise arrays with known convolutions
        a = rand.uniform(0, 10)
        b = rand.uniform(0, 10)
        cos_modes = np.zeros(self.modes, dtype = complex)
        cos_modes[1] = 0.5*a
        sin_modes = np.zeros(self.modes, dtype = complex)
        sin_modes[1] = -1j*0.5*b

        # initialise random arrays to convolve5
        # REQUIRED PADDING WITH M ZEROS TO AVOID END EFFECTS
        rand1 = np.random.rand(self.modes) + 1j*np.random.rand(self.modes)
        rand1[0] = np.real(rand1[0])
        rand1 = np.append(rand1, [0]*self.modes)
        rand2 = np.random.rand(self.modes) + 1j*np.random.rand(self.modes)
        rand2[0] = np.real(rand2[0])
        rand2 = np.append(rand2, [0]*self.modes)

        # perform convolutions
        conv_cos_cos_sum = conv_scalar(cos_modes, cos_modes, method = 'direct')
        conv_cos_sin_sum = conv_scalar(cos_modes, sin_modes, method = 'direct')
        conv_sin_cos_sum = conv_scalar(sin_modes, cos_modes, method = 'direct')
        conv_sin_sin_sum = conv_scalar(sin_modes, sin_modes, method = 'direct')
        conv_rand1_rand1_sum = conv_scalar(rand1, rand1, method = 'direct')
        conv_rand1_rand2_sum = conv_scalar(rand1, rand2, method = 'direct')
        conv_rand2_rand1_sum = conv_scalar(rand2, rand1, method = 'direct')
        conv_rand2_rand2_sum = conv_scalar(rand2, rand2, method = 'direct')
        conv_cos_cos_fft = conv_scalar(cos_modes, cos_modes, method = 'fft')
        conv_cos_sin_fft = conv_scalar(cos_modes, sin_modes, method = 'fft')
        conv_sin_cos_fft = conv_scalar(sin_modes, cos_modes, method = 'fft')
        conv_sin_sin_fft = conv_scalar(sin_modes, sin_modes, method = 'fft')
        conv_rand1_rand1_fft = conv_scalar(rand1, rand1, method = 'fft')
        conv_rand1_rand2_fft = conv_scalar(rand1, rand2, method = 'fft')
        conv_rand2_rand1_fft = conv_scalar(rand2, rand1, method = 'fft')
        conv_rand2_rand2_fft = conv_scalar(rand2, rand2, method = 'fft')

        # do they commute
        self.assertTrue(np.allclose(conv_cos_sin_sum, conv_sin_cos_sum))
        self.assertTrue(np.allclose(conv_rand1_rand2_sum, conv_rand2_rand1_sum))
        self.assertTrue(np.allclose(conv_cos_sin_fft, conv_sin_cos_fft))
        self.assertTrue(np.allclose(conv_rand1_rand2_fft, conv_rand2_rand1_fft))

        # correct values for known
        conv_cos_cos_true = np.zeros(self.modes, dtype = complex)
        conv_cos_cos_true[0] = (a**2)/2
        conv_cos_cos_true[2] = (a**2)/4
        conv_cos_sin_true = np.zeros(self.modes, dtype = complex)
        conv_cos_sin_true[2] = -1j*(a*b)/4
        conv_sin_sin_true = np.zeros(self.modes, dtype = complex)
        conv_sin_sin_true[0] = (b**2)/2
        conv_sin_sin_true[2] = -(b**2)/4
        self.assertTrue(np.array_equal(conv_cos_cos_sum, conv_cos_cos_true))
        self.assertTrue(np.array_equal(conv_cos_sin_sum, conv_cos_sin_true))
        self.assertTrue(np.array_equal(conv_sin_sin_sum, conv_sin_sin_true))
        self.assertTrue(np.allclose(conv_cos_cos_fft, conv_cos_cos_true))
        self.assertTrue(np.allclose(conv_cos_sin_fft, conv_cos_sin_true))
        self.assertTrue(np.allclose(conv_sin_sin_fft, conv_sin_sin_true))

        # sum and fft same for random values
        self.assertTrue(np.allclose(conv_rand1_rand1_sum, conv_rand1_rand1_fft))
        self.assertTrue(np.allclose(conv_rand1_rand2_sum, conv_rand1_rand2_fft))
        self.assertTrue(np.allclose(conv_rand2_rand2_sum, conv_rand2_rand2_fft))

    def test_conv_array(self):
        # initialise general ellipse
        a = rand.uniform(0, 10)
        b = rand.uniform(0, 10)
        elps = np.zeros([self.modes, 2], dtype = complex)
        elps[1, 0] = 0.5*a
        elps[1, 1] = -1j*0.5*b

        # initialise matrix and vector
        sigma = rand.uniform(0, 10)
        beta = rand.uniform(0, 10)
        rho = rand.uniform(0, 50)
        matrix = np.zeros([self.modes, 3, 3], dtype = complex)
        matrix[0, 0, 0] = -sigma
        matrix[0, 0, 1] = sigma
        matrix[0, 1, 0] = rho
        matrix[0, 1, 1] = -1
        matrix[0, 2, 2] = -beta
        matrix[1, 1, 2] = -0.5*a
        matrix[1, 2, 0] = -1j*0.5*b
        matrix[1, 2, 1] = 0.5*a
        vector = np.zeros([self.modes, 3], dtype = complex)
        vector[1, 0] = 0.5*a
        vector[1, 1] = -1j*0.5*b

        # initialise random arrays for general matrix convolution
        # REQUIRED PADDING WITH M ZEROS TO AVOID END EFFECTS
        dim1 = rand.randint(1, 5)
        dim2 = rand.randint(1, 5)
        dim3 = rand.randint(1, 5)
        rand_mat1 = np.random.rand(self.modes, dim1, dim2) + 1j*np.random.rand(self.modes, dim1, dim2)
        rand_mat1[0] = np.real(rand_mat1[0])
        rand_mat1 = np.append(rand_mat1, np.zeros([self.modes, dim1, dim2], dtype = complex), axis = 0)
        rand_mat2 = np.random.rand(self.modes, dim2, dim3) + 1j*np.random.rand(self.modes, dim2, dim3)
        rand_mat2[0] = np.real(rand_mat2[0])
        rand_mat2 = np.append(rand_mat2, np.zeros([self.modes, dim2, dim3], dtype = complex), axis = 0)

        # perform convolutions
        conv_elps_sum = conv_array(elps, elps, method = 'direct')
        conv_matvec_sum = conv_array(matrix, vector, method = 'direct')
        conv_rand_mat12_sum = conv_array(rand_mat1, rand_mat2, method = 'direct')
        conv_elps_fft = conv_array(elps, elps, method = 'fft')
        conv_matvec_fft = conv_array(matrix, vector, method = 'fft')
        conv_rand_mat12_fft = conv_array(rand_mat1, rand_mat2, method = 'fft')

        # initialise known inner product convolution
        conv_elps_true = np.zeros(self.modes, dtype = complex)
        conv_elps_true[0] = 0.5*(a**2 + b**2)
        conv_elps_true[2] = 0.25*(a**2 - b**2)

        # initialise known matrix vector product convolution
        conv_matvec_true = np.zeros([self.modes, 3], dtype = complex)
        conv_matvec_true[1, 0] = -0.5*sigma*(a + 1j*b)
        conv_matvec_true[1, 1] = 0.5*(rho*a + 1j*b)
        conv_matvec_true[2, 2] = -1j*a*b*0.5

        # check inner product
        self.assertTrue(np.array_equal(conv_elps_sum, conv_elps_true))
        self.assertTrue(np.allclose(conv_elps_fft, conv_elps_true))

        # check matrix vector product
        self.assertTrue(np.array_equal(conv_matvec_sum, conv_matvec_true))
        self.assertTrue(np.allclose(conv_matvec_fft, conv_matvec_true))

        # check general matrix multiplication
        self.assertTrue(np.allclose(conv_rand_mat12_sum, conv_rand_mat12_fft))


if __name__ == "__main__":
    unittest.main()
