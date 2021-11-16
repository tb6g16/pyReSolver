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

    def matmul_left_const(self, factor):
        """Left multiply current instance by constant array."""
        return Trajectory(np.transpose(np.matmul(factor, np.transpose(self))))

    def matmul_left_traj(self, other):
        """Left multiply current instance by another trajectory instance."""
        if len(self.shape) == 2 and len(other.shape) == 2:
            return Trajectory(np.diag(np.inner(other, self)))
        elif len(self.shape) == 3 and len(other.shape) == 3:
            return Trajectory(np.matmul(other, self))
        else:
            return Trajectory(np.squeeze(np.matmul(other, np.reshape(self, (*self.shape, 1)))))

    # def __eq__(self, other_traj, rtol = 1e-5, atol = 1e-8):
    #     """Evaluate (approximate) equality of trajectory and current instance."""
    #     return np.allclose(self.modes, other_traj.modes, rtol = rtol, atol = atol)

    # def __round__(self, decimals = 6):
    #     """Return a new trajectory with rounded modes."""
    #     return Trajectory(np.around(self.modes, decimals = decimals))

    # def __abs__(self):
    #     """Define the behaviour of the in-built absolute function."""
    #     return np.linalg.norm(self.modes)
