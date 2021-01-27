# This file contains the class definition for a general trajectory in some
# vector space. This will most commonly be a periodic state-space trajectory.

import numpy as np
import scipy.integrate as integ
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from System import System

class Trajectory:
    """
        A trajectory in some finite-dimensional vector space parameterised with
        respect to a standard 'time' unit assumed to range from 0 to 2*pi.

        Attributes
        ----------
        curve_array: numpy.ndarray
            the discretised trajectory, number of rows equal to the dimension
            of the vector space and columns equal to the number of 'time' steps
            taken
        curve_fun: function
            function defining the trajectory, given as an input to __init__ to
            generate curve_array attribute
        closed: bool
            prescribing whether the trajectory is closed, i.e. is it periodic
            in 'time'
        
        Methods
        -------
        func2array(curve_func, time_disc = 200)
        gradient()
        plot()
    """

    __slots__ = ['mode_array', 'curve_func', 'shape']
    __array_priority__ = 1e16

    def __init__(self, curve, modes = 33):
        """
            Initialise an instance of the Trajectory object, with either a
            continuous of discrete time function.

            Parameters
            ----------
            curve: function or numpy.ndarray
                function defining the trajectory, either given by a python
                function (continuous) or a numpy array (discrete)
        """
        if type(curve) == np.ndarray:
            if len(np.shape(curve)) == 1:
                curve = np.expand_dims(curve, axis = 0)
            if len(np.shape(curve)) == 2:
                self.mode_array = curve
                self.curve_func = None
                self.shape = np.shape(curve)
            else:
                raise AttributeError("The trajectory array has to 2D (only \
                rows and columns)!")
        elif hasattr(curve, '__call__'):
            self.mode_array = self.func2array(curve, modes)
            self.curve_func = curve
            self.shape = np.shape(self.mode_array)
        else:
            raise TypeError("Curve variable has to be either a function or a \
            2D numpy array!")

    def func2array(self, curve_func, modes):
        """
            Discretise a continuous time representation of a function (given
            as a python function) to a discrete time representation (as a
            numpy array).

            Parameters
            ----------
            curve_func: function
                python function that defines the continuous time representation
                of the trajectory
            time_disc: positive integer
                number of discrete time locations to use
        """
        # import trajectory_functions as traj_funcs
        disc = 2*(modes - 1)
        curve_array = np.zeros([np.shape(curve_func(0))[0], disc])
        t = np.linspace(0, 2*np.pi*(1 - 1/disc), disc)
        for i in range(disc):
            curve_array[:, i] = curve_func(t[i])
        return np.fft.rfft(curve_array, axis = 1)

    def __add__(self, other_traj):
        if not isinstance(other_traj, Trajectory):
            raise TypeError("Inputs are not of the correct type!")
        return Trajectory(self.mode_array + other_traj.mode_array)

    def __sub__(self, other_traj):
        if not isinstance(other_traj, Trajectory):
            raise TypeError("Inputs are not of the correct type!")
        return Trajectory(self.mode_array - other_traj.mode_array)

    def __mul__(self, factor):
        # scalar multiplication
        if type(factor) == float or type(factor) == int or \
            type(factor) == np.float64 or type(factor) == np.int64:
            return Trajectory(factor*self.mode_array)
        else:
            raise TypeError("Inputs are not of the correct type!")

    def __rmul__(self, factor):
        return self.__mul__(factor)

    def __matmul__(self, factor):
        if type(factor) == np.ndarray:
            return Trajectory(np.matmul(factor, self.mode_array))
        elif hasattr(factor, '__call__'):
            curve = np.fft.irfft(self.mode_array, axis = 1)
            for i in range(np.shape(curve)[1]):
                curve[:, i] = np.matmul(factor(i), curve[:, i])
            return Trajectory(np.fft.rfft(curve, axis = 1))
        else:
            raise TypeError("Inputs are not of the correct type!")

    def __rmatmul__(self, factor):
        return self.__matmul__(factor)

    def __pow__(self, exponent):
        # perform element-by-element exponentiation
        curve = np.fft.irfft(self.mode_array, axis = 1)
        return Trajectory(np.fft.rfft(curve**exponent, axis = 1))

    def __eq__(self, other_traj, rtol = 1e-6, atol = 1e-6):
        if not isinstance(other_traj, Trajectory):
            raise TypeError("Inputs are not of the correct type!")
        return np.allclose(self.mode_array, other_traj.mode_array, \
            rtol = rtol, atol = atol)

    def __getitem__(self, key):
        i, j = key
        return self.mode_array[i, j]

    def __setitem__(self, key, value):
        i, j = key
        self.mode_array[i, j] = value

    def plot(self, gradient = None):
        """
            This function is a placeholder and will be used for plotting
            purposes.
        """
        import trajectory_functions as traj_funcs
        
        if self.shape[0] == 2:
            # convert to time domain
            curve = traj_funcs.swap_tf(self)

            # plotting trajectory
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(np.append(curve[0], curve[0, 0]), np.append(curve[1], \
                curve[1, 0]))
            ax.set_aspect('equal')

            # add gradient
            if gradient != None:
                grad = traj_funcs.traj_grad(self)
                grad = traj_funcs.swap_tf(grad)
                for i in range(0, curve.shape[1], int(1/gradient)):
                    ax.quiver(curve[0, i], curve[1, i], grad[0, i], grad[1, i])
            plt.show()
        else:
            raise ValueError("Can't plot!")

if __name__ == '__main__':
    from test_cases import unit_circle as circ
    from test_cases import ellipse as elps

    unit_circle1 = Trajectory(circ.x)
    unit_circle2 = 0.5*Trajectory(circ.x)

    unit_circle3 = np.pi*unit_circle1 + unit_circle2

    unit_circle1.plot(gradient = 16/64)
    unit_circle3.plot(gradient = 16/64)
    
    ellipse = Trajectory(elps.x)
    ellipse.plot(gradient = 16/64)