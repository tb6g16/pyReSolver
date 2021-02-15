# This file contains the function definition for the convolution sum operation
# between two real valued Fourier mode pairs.

import numpy as np
from Trajectory import Trajectory

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
    conv_modes = np.zeros([1, traj1.shape[1]], dtype = complex)

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
                                np.dot(np.conj(traj1[:, m]), factor2_sum) + \
                                np.dot(traj1_zero, traj2_at_n)

    # account for zero mode
    conv_modes[:, 0] = conv_modes[:, 0]*2

    # initialise arrays
    prod_modes = np.zeros([1, traj1.shape[1]], dtype = complex)

    # nested loop to perform convolution
    for n in range(traj1.shape[1]):
        for m in range(1 - traj1.shape[1], traj1.shape[1]):
            if m < 0:
                vec1 = np.conj(traj1[:, -m])
            else:
                vec1 = traj1[:, m]
            if n - m < traj1.shape[1]:
                if n - m < 0:
                    vec2 = np.conj(traj2[:, m - n])
                else:
                    vec2 = traj2[:, n - m]
            else:
                vec2 = np.zeros(traj1.shape[0])
            prod_modes[:, n] += np.dot(vec1, vec2)

    return Trajectory(prod_modes), Trajectory(conv_modes)
