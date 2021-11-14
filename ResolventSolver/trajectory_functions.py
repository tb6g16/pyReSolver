# This file contains the function definitions for some trajectory-trajectory
# and trajectory-system interactions.

import numpy as np

from ResolventSolver.Trajectory import Trajectory

def transpose(traj):
    """
        Return the transpose of a trajectory.

        Parameters
        ----------
        traj : Trajectory

        Returns
        -------
        Trajectory
    """
    return Trajectory(np.transpose(traj.modes, axes = [0, *range(1, traj.modes.ndim)[::-1][0:]]))

def conj(traj):
    """
        Return the complex conjugate of a trajectory.

        Parameters
        ----------
        traj : Trajectory

        Returns
        -------
        Trajectory
    """
    return Trajectory(np.conj(traj.modes))

# FIXME: the copy function is avoid strange assignment behaviour
def traj_rfft(array, fftplans):
    np.copyto(fftplans.tmp_t, array)
    fftplans.fft()
    return Trajectory(np.copy(fftplans.tmp_f))

def traj_irfft(traj, fftplans):
    np.copyto(fftplans.tmp_f, traj.modes)
    fftplans.ifft()
    return fftplans.tmp_t

def traj_grad(traj):
    """
        Return the gradient of a trajectory.

        Parameters
        ----------
        traj : Trajectory

        Returns
        -------
        Trajectory
    """
    # initialise array of mode modifiers to be multiplied
    modifiers = np.transpose(np.tile(1j*np.arange(traj.shape[0]), (traj.shape[1], 1)))

    # multiply element-wise
    new_modes = modifiers*traj.modes

    # force end mode to be zero to preserve symmetry
    new_modes[-1][:] = 0

    return Trajectory(new_modes)

def traj_response(traj, fftplans, func):
    """
        Return the response of a trajectory over its length due to a function.

        Parameters
        ----------
        traj : Trajectory
        func : function

        Returns
        -------
        Trajectory
    """
    # convert trajectory to time domain
    curve = traj_irfft(traj, fftplans)

    # evaluate response in time domain
    new_curve = func(curve)

    # convert back to frequency domain and return
    return traj_rfft(new_curve, fftplans)

# NOTE: this function is here just for the jacobian function with multiple trajectory
#       trajectory inputs (avoiding python loops at all costs)
def traj_response2(traj1, traj2, fftplans, func):
    # convert trajectories to time domain
    curve1 = traj_irfft(traj1, fftplans)
    curve2 = traj_irfft(traj2, fftplans)

    # evaluate response in time domain
    new_curve = func(curve1, curve2)

    # convert back to frequency domain and return
    return traj_rfft(new_curve, fftplans)
