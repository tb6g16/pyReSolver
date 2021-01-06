# This file contains the class definition for a general trajectory in some
# vector space. This will most commonly be a periodic state-space trajectory

# Thomas Burton - October 2020

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

    __slots__ = ['curve_array', 'curve_func', 'shape']
    __array_priority__ = 1e16

    def __init__(self, curve, disc = 64):
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
                self.curve_array = curve
                self.curve_func = None
                self.shape = np.shape(curve)
            else:
                raise AttributeError("The trajectory array has to 2D (only \
                rows and columns)!")
        elif hasattr(curve, '__call__'):    
            self.curve_array = self.func2array(curve, time_disc = disc)
            self.curve_func = curve
            self.shape = np.shape(self.curve_array)
        else:
            raise TypeError("Curve variable has to be either a function or a \
            2D numpy array!")

    # TIME_DISC HAS TO BE EVEN BECAUSE OF RFFT ALGORITHM
    def func2array(self, curve_func, time_disc = 64):
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
        curve_array = np.zeros([np.shape(curve_func(0))[0], time_disc])
        t = np.linspace(0, 2*np.pi*(1 - 1/time_disc), time_disc)
        for i in range(time_disc):
                curve_array[:, i] = curve_func(t[i])
        return curve_array

    def __add__(self, other_traj):
        if not isinstance(other_traj, Trajectory):
            raise TypeError("Inputs are not of the correct type!")
        return Trajectory(self.curve_array + other_traj.curve_array)

    def __sub__(self, other_traj):
        if not isinstance(other_traj, Trajectory):
            raise TypeError("Inputs are not of the correct type!")
        return Trajectory(self.curve_array - other_traj.curve_array)

    def __mul__(self, factor):
        # scalar multiplication
        if type(factor) == float or type(factor) == int or \
            type(factor) == np.float64 or type(factor) == np.int64:
            return Trajectory(factor*self.curve_array)
        else:
            raise TypeError("Inputs are not of the correct type!")

    def __rmul__(self, factor):
        return self.__mul__(factor)
    
    def __matmul__(self, factor):
        if type(factor) == np.ndarray:
            return Trajectory(np.matmul(factor, self.curve_array))
        elif hasattr(factor, '__call__'):
            s_disc = self.shape
            new_traj = np.zeros(s_disc)
            for i in range(s_disc[1]):
                new_traj[:, i] = np.matmul(factor(i), self[:, i])
            return Trajectory(new_traj)
        else:
            raise TypeError("Inputs are not of the correct type!")

    def __rmatmul__(self, factor):
        return self.__matmul__(factor)

    def __pow__(self, exponent):
        # perform element-by-element exponentiation
        return Trajectory(self.curve_array ** exponent)

    def __eq__(self, other_traj, rtol = 1e-6, atol = 1e-6):
        if not isinstance(other_traj, Trajectory):
            raise TypeError("Inputs are not of the correct type!")
        return np.allclose(self.curve_array, other_traj.curve_array, \
            rtol = rtol, atol = atol)

    def __getitem__(self, key):
        i, j = key
        return self.curve_array[i, j]

    def __setitem__(self, key, value):
        i, j = key
        self.curve_array[i, j] = value

    def plot(self, gradient = None):
        """
            This function is a placeholder and will be used for plotting
            purposes.
        """
        import trajectory_functions as traj_funcs
        
        if self.shape[0] == 2:
            # plotting trajectory
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(np.append(self.curve_array[0], self[0, 0]), \
                np.append(self.curve_array[1], self[1, 0]))
            ax.set_aspect('equal')

            # add gradient
            if gradient != None:
                grad = traj_funcs.traj_grad(self)
                for i in range(0, self.shape[1], int(1/gradient)):
                    ax.quiver(self[0, i], self[1, i], grad[0, i], grad[1, i])
            plt.show()
        else:
            raise ValueError("Bruh!")

    # def plot(self, gradient = False, gradient_density = None):
    #     """
    #         Plot 1D, 2D, or 3D trajectories or gradients.

    #         Paramters
    #         ---------
    #         gradient: bool
    #             boolean to decide whether to plot the gradient (tangent
    #             vectors) with the trajectory curve
    #         gradient_density: float between 0 and 1
    #             amount of gradient vectors to show on the trajectory curve
    #     """
    #     # check if gradient density is between o and 1
    #     if gradient_density < 0 or gradient_density > 1:
            # raise ValueError("gradient_density should be between 0 and 1 \
            # inclusive!")
    #     # check if gradient attribute has value None
    #     # if gradient == True and self.grad is None:
    #     #     self.gradient()
    #     # check dimension of plotting space and then plot (if possible)
    #     if np.shape(self.curve_array)[0] == 1:
    #         t = np.linspace(0, 2*np.pi, np.shape(self.curve_array)[1])
    #         # plot state against parametric time
    #         fig = plt.figure()
    #         ax = fig.gca()
    #         ax.plot(t, self.curve_array[0])

    #         # NEEDS TO IMPLEMENT HERE ON X-AXIS

    #         plt.show()
    #     elif np.shape(self.curve_array)[0] == 3:
    #         # plot in 3D vector space
    #         fig = plt.figure()
    #         ax = fig.gca(projection = '3d')
    #         ax.plot(np.append(self.curve_array[0], self.curve_array[0, 0]), \
    #             np.append(self.curve_array[1], self.curve_array[1, 0]), \
    #             np.append(self.curve_array[2], self.curve_array[2, 0]))
    #         # NEED TO DO GRADIENT PLOT FOR 3D
    #         plt.show()
    #     else:
    #         raise ValueError("Cannot plot trajectories in higher dimensions!")
    #     return None

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