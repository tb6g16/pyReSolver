# This file contains the class definition for a general trajectory in some
# vector space.

import numpy as np

class Trajectory(np.ndarray):
    """
        A trajectory in state-space stored as an array of Fourier modes.

        Attributes
        ----------
        modes : ndarray
            2D array containing data of float type.
        shape : tuple of int
            Shape of the trajectory equivelent array.
    """

    def __new__(subtype, input_array):
        return input_array.view(subtype)

    # TODO: use out argument in einsums
    def traj_inner(self, other):
        """Inner product of current instance and another trajectory instances."""
        return np.einsum('ik,ik->i', self, other, optimize=False)

    def matmul_left_traj(self, other):
        """Left multiply current instance by another trajectory instance."""
        return Trajectory(np.einsum('ikl,il->ik', other, self))

    def __eq__(self, other_traj, rtol = 1e-5, atol = 1e-8):
        """Evaluate (approximate) equality of trajectory and current instance."""
        return np.allclose(self, other_traj, rtol = rtol, atol = atol)

    def __round__(self, decimals = 6):
        """Return a new trajectory with rounded modes."""
        return np.around(self, decimals = decimals)

    def __abs__(self):
        """Define the behaviour of the in-built absolute function."""
        return np.linalg.norm(self)
