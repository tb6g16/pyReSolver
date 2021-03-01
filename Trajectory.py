# This file contains the class definition for a general trajectory in some
# vector space. This will most commonly be a periodic state-space trajectory.

import numpy as np
import scipy.integrate as integ
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from operator import add, sub, mul
from my_fft import my_rfft, my_irfft
from traj_util import func2curve, list2array, array2list

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

    # add type attribute
    __slots__ = ['mode_list', 'shape']
    __array_priority__ = 1e16

    # need extra tests to make sure everything is of the same (not just correct)
    # type, and all have compatible dimensions (are for the same space)
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
        if type(curve) == list:
            for i in range(len(curve)):
                if type(curve[i]) != np.ndarray:
                    raise TypeError("Has to be numpy array at each location along trajectory!")
            self.mode_list = curve
            self.shape = (len(curve), *np.shape(curve[0]))
        elif hasattr(curve, '__call__'):
            self.mode_list = array2list(my_rfft(func2curve(curve, modes)))
            self.shape = (len(self.mode_list), *np.shape(self.mode_list[0]))
        else:
            raise TypeError("Curve variable has to be either a function or a \
            2D numpy array!")

    def __add__(self, other_traj):
        if not isinstance(other_traj, Trajectory):
            raise TypeError("Inputs are not of the correct type!")
        return Trajectory(list(map(add, self.mode_list, other_traj.mode_list)))

    def __sub__(self, other_traj):
        if not isinstance(other_traj, Trajectory):
            raise TypeError("Inputs are not of the correct type!")
        return Trajectory(list(map(sub, self.mode_list, other_traj.mode_list)))

    def __mul__(self, factor):
        # scalar multiplication
        if type(factor) == float or type(factor) == int or \
            type(factor) == np.float64 or type(factor) == np.int64:
            return Trajectory([self.mode_list[i]*factor for i in range(len(self.mode_list))])
        else:
            raise TypeError("Inputs are not of the correct type!")

    def __rmul__(self, factor):
        return self.__mul__(factor)

    def __matmul__(self, factor):
        if type(factor) == np.ndarray:
            return Trajectory([np.matmul(factor, self.mode_list[i]) \
                               for i in range(len(self.mode_list))])
        else:
            raise TypeError("Inputs are not of the correct type!")

    def __rmatmul__(self, factor):
        return self.__matmul__(factor)

    def __eq__(self, other_traj, rtol = 1e-6, atol = 1e-6):
        if not isinstance(other_traj, Trajectory):
            raise TypeError("Inputs are not of the correct type!")
        if_eq = [False]*self.shape[0]
        for i in range(self.shape[0]):
            if_eq[i] = np.allclose(self.mode_list[i], other_traj.mode_list[i], rtol, atol)
        return all(if_eq)

    def __getitem__(self, key):
        if type(key) == int or type(key) == slice:
            return self.mode_list[key]
        else:
            return self.mode_list[key[0]][key[1:]]

    def __setitem__(self, key, value):
        if type(key) == int or type(key) == slice:
            self.mode_list[key] = value
        else:
            return self.mode_list[key[0]][key[1:]]

    def plot(self, gradient = None):
        """
            This function is a placeholder and will be used for plotting
            purposes.
        """
        import trajectory_functions as traj_funcs
        
        if self.shape[1] == 2:
            # convert to time domain
            curve = my_irfft(list2array(self.mode_list))

            # plotting trajectory
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(np.append(curve[:, 0], curve[0, 0]), np.append(curve[:, 1], curve[0, 1]))
            ax.set_aspect('equal')

            # add gradient
            # if gradient != None:
            #     grad = traj_funcs.traj_grad(self)
            #     grad = my_ifft(list2array(grad.mode_list))
            #     for i in range(0, curve.shape[1], int(1/gradient)):
            #         ax.quiver(curve[0, i], curve[1, i], grad[0, i], grad[1, i])
            
            # plt.xlabel("$x$")
            # plt.ylabel("$\dot{x}$")
            # plt.xlim([-2.2, 2.2])
            # plt.ylim([-4, 4])
            # plt.grid()
            plt.show()
        else:
            raise ValueError("Can't plot!")

if __name__ == '__main__':
    from test_cases import unit_circle as uc
    from test_cases import ellipse as elps

    uc1 = Trajectory(uc.x)
    uc2 = 0.5*Trajectory(uc.x)

    uc3 = np.pi*uc1 + uc2

    uc1.plot(gradient = 16/64)
    uc3.plot(gradient = 16/64)
    
    ellipse = Trajectory(elps.x)
    ellipse.plot(gradient = 16/64)
