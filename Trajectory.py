# This file contains the class definition for a general trajectory in some
# vector space.

import numpy as np
from my_fft import my_rfft, my_irfft
from traj_util import func2curve, list2array, array2list

class Trajectory:
    """
        A trajectory in state-space stored as an array of Fourier modes.

        Attributes
        ----------
        mode_life : ndarray
            2D array containing data of float type.
        shape : tuple of int
            Shape of the trajectory equivelent array.
        type : type
            Data type of the contents of the mode_list.
    """

    # add type attribute
    __slots__ = ['mode_list', 'shape', 'type']
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
        if type(curve) == list:
            type_same, shape_same = self.check_type_shape(curve)
            if type_same == False:
                raise TypeError("Data types of list elements must all be the same!")
            if shape_same == False:
                raise ValueError("Arrays must all be the same shape!")
            self.mode_list = curve
            self.shape = (len(curve), *np.shape(curve[0]))
            self.type = type(curve[0])
        elif hasattr(curve, '__call__'):
            self.mode_list = array2list(my_rfft(func2curve(curve, modes)))
            self.shape = (len(self.mode_list), *np.shape(self.mode_list[0]))
            self.type = type(self.mode_list[0])
        else:
            raise TypeError("Curve variable has to be either a function or a list!")

    def __add__(self, other_traj):
        """Add trajectory to current instance."""
        if not isinstance(other_traj, Trajectory):
            raise TypeError("Inputs are not of the correct type!")
        return Trajectory([self.mode_list[i] + other_traj.mode_list[i] for i in range(self.shape[0])])

    def __sub__(self, other_traj):
        """Substract trajectory from current instance."""
        if not isinstance(other_traj, Trajectory):
            raise TypeError("Inputs are not of the correct type!")
        return Trajectory([self.mode_list[i] - other_traj.mode_list[i] for i in range(self.shape[0])])

    def __mul__(self, factor):
        """Multiply current istance by scalar."""
        if type(factor) == float or type(factor) == int or \
            type(factor) == np.float64 or type(factor) == np.int64:
            return Trajectory([self.mode_list[i]*factor for i in range(self.shape[0])])
        else:
            raise TypeError("Inputs are not of the correct type!")

    def __rmul__(self, factor):
        return self.__mul__(factor)

    def __matmul__(self, factor):
        """Right muyltiply current instance by array or another trajectory."""
        if type(factor) == np.ndarray:
            return Trajectory([np.matmul(self.mode_list[i], factor) \
                               for i in range(self.shape[0])])
        elif type(factor) == Trajectory:
            return Trajectory([np.matmul(self.mode_list[i], factor.mode_list[i]) \
                               for i in range(self.shape[0])])
        else:
            raise TypeError("Inputs are not of the correct type!")

    def __rmatmul__(self, factor):
        """Left multiply current instance by array or another trajectory."""
        if type(factor) == np.ndarray:
            return Trajectory([np.matmul(factor, self.mode_list[i]) \
                               for i in range(self.shape[0])])
        elif type(factor) == Trajectory:
            return Trajectory([np.matmul(self.mode_list[i], factor.mode_list[i]) for i in range(self.shape[0])])
        else:
            raise TypeError("Inputs are not of the correct type!")

    def __eq__(self, other_traj, rtol = 1e-6, atol = 1e-6):
        """Evaluate (approximate) equality of trajectory and current instance."""
        if not isinstance(other_traj, Trajectory):
            raise TypeError("Inputs are not of the correct type!")
        for i in range(self.shape[0]):
            if not np.allclose(self.mode_list[i], other_traj.mode_list[i], rtol, atol):
                return False
        return True

    def __getitem__(self, key):
        """Return the element of the mode list indexed by the given key."""
        if type(key) == int or type(key) == slice:
            return self.mode_list[key]
        else:
            return self.mode_list[key[0]][key[1:]]

    def __setitem__(self, key, value):
        """Set the value of the mode list indexed by the given key."""
        if type(key) == int or type(key) == slice:
            self.mode_list[key] = value
        else:
            self.mode_list[key[0]][key[1:]] = value
        if not self.check_type_shape(self.mode_list)[0]:
            raise ValueError("Invalid assignment!")

    def __round__(self, decimals = 6):
        """Return a new trajectory with rounded modes."""
        traj_round = [None]*self.shape[0]
        for i in range(self.shape[0]):
            traj_round[i] = np.round(self[i], decimals = decimals)
        return Trajectory(traj_round)

    @staticmethod
    def check_type_shape(my_list):
        """
            Return whether the elements of a given list are the same shape and
            type.

            Parameters
            ----------
            my_list : list

            Returns
            -------
            type_same, shape_same : bool
        """
        # initialise bools
        type_same = True
        shape_same = True

        # loop over elements of the list to check type
        for i in range(len(my_list)):
            if type(my_list[i]) != type(my_list[0]):
                type_same = False

        # check the shape if the list type is ndarray
        if type(my_list[0]) == np.ndarray:
            for i in range(len(my_list)):
                if np.shape(my_list[i]) != np.shape(my_list[0]):
                    shape_same = False

        return type_same, shape_same
