# This file contains the function definitions for some trajectory-trajectory
# and trajectory-system interactions

import numpy as np
import scipy.integrate as integ
from Trajectory import Trajectory
from System import System

def traj_grad(traj):
    """
        This function calculates the gradient vectors of a given trajectory and
        returns it as an instance of the Trajectory class. The algorithm used
        is based off of the RFFT

        Parameters
        ----------
        traj: Trajectory object
            the trajectory which we will calculate the gradient for
        
        Returns
        -------
        grad: Trajectory object
            the gradient of the input trajectory
    """
    # number of discretised time locations
    time_disc = traj.shape[1]

    # FFT along the time dimension
    mode_array = np.fft.rfft(traj.curve_array, axis = 1)

    # loop over time and multiply modes by modifiers
    for k in range(time_disc//2):
        mode_array[:, k] *= 1j*k
    
    # force zero mode if symmetric
    if time_disc % 2 == 0:
        mode_array[:, time_disc//2] = 0
    
    # IFFT to get discrete time gradients
    return Trajectory(np.fft.irfft(mode_array, axis = 1))

def average_over_s(traj):
    """
        This function calculates the average of a trajectory over its time
        domain, for each dimension separately.

        Parameters
        ----------
        traj: Trajectory object
            the trajectory for which the integration will be taken
        
        Returns
        -------
        integ_vec: numpy array
            a 1D numpy array (vector) containing the average of the trajectory
            over s for each of its dimensions
    """
    # make trajectory truly periodic
    integ_traj = np.concatenate((traj.curve_array, traj[:, 0:1]), axis = 1)
    
    # initialise vector to hold integration results
    traj_disc = np.linspace(0, 2*np.pi, np.shape(integ_traj)[1])
    integ_vec = np.zeros([traj.shape[0]])

    # loop over each dimension to evaluate the integration
    for i in range(np.shape(integ_traj)[0]):
        integ_vec[i] = (1/(2*np.pi))*integ.trapz(integ_traj[i, :], traj_disc)
    
    return integ_vec

def traj_inner_prod(traj1, traj2):
    """
        This function calculates the Euclidean inner product of two
        trajectories at each location along their domains, s.

        Parameters
        ----------
        traj1: Trajectory object
            the first trajectory in the inner product
        traj2: Trajectory object
            the second trajectory in the inner product
        
        Returns
        -------
        traj_prod: Trajectory object
            the inner product of the two trajectories at each location along
            their domains, s
    """
    # number of time locations
    disc_size = traj1.shape[1]

    # initialise output array
    product_array = np.zeros([1, disc_size])

    # calculate inner product at each location s
    for i in range(disc_size):
        product_array[0, i] = np.dot(traj1[:, i], traj2[:, i])
    
    return Trajectory(product_array)

def traj_response(traj, func):
    """
        This function evaluates the response over the domain of a given
        trajectory due to a given dyanmical system.

        Parameters
        ----------
        traj: Trajectory object
            the trajectory over which the response will be evaluated
        func: function
            the function that defines the response at each location along the
            trajectory, must be a function that inputs a vector and outputs a
            vector of the same size
        
        Returns
        -------
        response_traj: Trajectory object
            the response at each location of the trajectory, given as an
            instance of the Trajectory class
    """
    # initialise arrays
    array_size = traj.shape
    response_array = np.zeros(array_size)
    
    # evaluate response
    for i in range(array_size[1]):
        response_array[:, i] = func(traj[:, i])
    
    return Trajectory(response_array)

def jacob_init(traj, sys, if_transp = False):
    """
        This function initialises a jacobian function that returns the jacobian
        matrix for the given system at each location along the domain of the
        trajectory (through the indexing of the array, i).

        Parameters
        ----------
        traj: Trajectory object
            the trajectory over which the jacobian matrix will be evaluated
        sys: System object
            the system for which the jacobian matrix is for
        
        Returns
        -------
        jacob: function
            the function that returns the jacobian matrix for a given index of
            the array defining the trajectory in state-space
    """
    def jacobian(i):
        """
            This function returns the jacobian matrix for a dynamical system at
            a specified location along a trajectory through the associated
            state-space, indexed by i.

            Parameters
            ----------
            i: positive integer
                the discretised location along the trajectory in state-space
            
            Returns
            -------
            jacobian_matrix: numpy array
                the 2D numpy array for the jacobian of a dynamical system given
                at a specified location of the trajectory
        """
        # test for input
        if i%1 != 0:
            raise TypeError("Inputs are not of the correct type!")
        if i >= traj.shape[1]:
            raise ValueError("Input index is too large!")

        # make sure index is integer
        i = int(i)

        # state at index
        state = traj[:, i]

        # the jocobian for that state
        if if_transp == True:
            return np.transpose(sys.jacobian(state))
        elif if_transp == False:
            return sys.jacobian(state)
    return jacobian
