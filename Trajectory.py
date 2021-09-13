# This file contains the class definition for a general trajectory in some
# vector space.

import numpy as np

from my_fft import my_rfft, my_irfft
from traj_util import func2curve

class Trajectory:
    """
        A trajectory in state-space stored as an array of Fourier modes.

        Attributes
        ----------
        modes : ndarray
            2D array containing data of float type.
        shape : tuple of int
            Shape of the trajectory equivelent array.
    """

    # add type attribute
    __slots__ = ['modes', 'shape']
    __array_priority__ = 1e100

    def __init__(self, curve, modes = 33):
        """
            Initialise a trajectory with a curve definition.

            Parameters
            ----------
            curve : function or ndarray
                Function or list that defines a trajectory in state-space
            modes : positive int, default=33
                Number of modes to represent the trajectory, ignored is curve
                is an array.
        """
        if type(curve) == np.ndarray:
            self.modes = curve
            self.shape = np.shape(curve)
        elif hasattr(curve, '__call__'):
            self.modes = my_rfft(func2curve(curve, modes))
            self.shape = np.shape(self.modes)
        else:
            raise TypeError("Curve variable has to be either a function or a numpy array!")

    def __add__(self, other_traj):
        """Add trajectory to current instance."""
        return Trajectory(self.modes + other_traj.modes)

    def __sub__(self, other_traj):
        """Substract trajectory from current instance."""
        return Trajectory(self.modes - other_traj.modes)

    def __mul__(self, factor):
        """Multiply current instance by scalar."""
        return Trajectory(self.modes*factor)

    def __rmul__(self, factor):
        return self.__mul__(factor)

    def matmul_left_const(self, factor):
        """Left multiply current instance by constant array."""
        return Trajectory(np.transpose(np.matmul(factor, np.transpose(self.modes))))

    def matmul_left_traj(self, other):
        """Left multiply current instance by another trajectory instance."""
        if len(self.shape) == 2 and len(other.shape) == 2:
            return Trajectory(np.diag(np.inner(other.modes, self.modes)))
        elif len(self.shape) == 3 and len(other.shape) == 3:
            return Trajectory(np.matmul(other.modes, self.modes))
        else:
            return Trajectory(np.squeeze(np.matmul(other.modes, np.reshape(self.modes, (*self.shape, 1)))))

    def __eq__(self, other_traj, rtol = 1e-5, atol = 1e-8):
        """Evaluate (approximate) equality of trajectory and current instance."""
        return np.allclose(self.modes, other_traj.modes, rtol = rtol, atol = atol)

    def __getitem__(self, key):
        """Return the element of the mode list indexed by the given key."""
        return self.modes[key]

    def __setitem__(self, key, value):
        """Set the value of the mode list indexed by the given key."""
        self.modes[key] = value

    def __round__(self, decimals = 6):
        """Return a new trajectory with rounded modes."""
        return Trajectory(np.around(self.modes, decimals = decimals))

    def __abs__(self):
        """Define the behaviour of the in-built absolute function."""
        return np.linalg.norm(self.modes)

    def __repr__(self):
        """Return the modes of the instance."""
        return np.array_repr(self.modes)
