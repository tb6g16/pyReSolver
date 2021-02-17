# This file contains the function definition for the convolution sum operation
# between two real valued Fourier mode pairs.

import numpy as np
from Trajectory import Trajectory

# Ideal functionality:
#   - takes in instances of Trajectory class
#   - makes sure they are compatible (is the matrix multiplication possible)
#   - performs the convolution and returns a new isntance of Trajectory class
#   - choice between 'fast' and 'slow'

def conv(traj1, traj2):
    """
        This function calculates the convolution sum of the modes for two
        trajectories (corresponding to multiplication in the time domain).

        Parameters
        ----------
        traj1: Trajectory object
            the first trajectory to evaluate the sum over
        traj2: Trajectory object
            the second trajectory to evaluate the sum over
        
        Returns
        -------
        conv_sum: Trajectory object
            the resulting modes from the convolution sum (corresponding to a
            new trajectory)
    """
    # initialise arrays
    conv_modes = np.zeros([1, np.shape(traj1)[1]], dtype = complex)

    # nested loop to perform convolution
    traj1_zero = traj1[:, 0]
    for n in range(traj1.shape[1]):
        traj2_at_n = traj2[:, n]
        for m in range(traj1.shape[1]):
            if n - m < 0:
                factor2_diff = np.conj(traj2[:, n - m])
            else:
                factor2_diff = traj2[:, n - m]
            if n + m > traj1.shape[1] - 1:
                factor2_sum = np.zeros(traj2.shape[0])
            else:
                factor2_sum = traj2[:, n + m]
            conv_modes[:, n] += np.dot(traj1[:, m], factor2_diff) + \
                                np.dot(np.conj(traj1[:, m]), factor2_sum)
        conv_modes[:, n] += np.dot(traj1_zero, traj2_at_n)

    # account for zero mode
    conv_modes[:, 0] = conv_modes[:, 0]*2

    return conv_modes
