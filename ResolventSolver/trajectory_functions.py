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
    return np.transpose(traj, axes = [0, *range(1, traj.ndim)[::-1][0:]])

# TODO: in-place here?
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
    return np.conj(traj)

def traj_rfft(traj_f, traj_t, fftplans):
    fftplans.fft(traj_f, traj_t)

def traj_irfft(traj_f, traj_t, fftplans):
    fftplans.ifft(traj_f, traj_t)

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
    return np.transpose(np.tile(1j*np.arange(traj.shape[0]), (traj.shape[1], 1)))*traj

def traj_response(traj, fftplans, func, new_traj, new_curve):
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
    traj_irfft(traj, fftplans.tmp_t, fftplans)

    # evaluate response in time domain
    func(fftplans.tmp_t, new_curve)

    # convert back to frequency domain and return
    traj_rfft(new_traj, new_curve, fftplans)

# TODO: in-place here
def traj_response2(traj1, traj2, fftplans, func):
    curve1 = np.zeros_like(fftplans.tmp_t)
    curve2 = np.zeros_like(fftplans.tmp_t)
    new_traj = np.zeros_like(traj1)

    # convert trajectories to time domain
    traj_irfft(traj1, curve1, fftplans)
    traj_irfft(traj2, curve2, fftplans)

    # evaluate response in time domain
    new_curve = func(curve1, curve2)

    # convert back to frequency domain and return
    traj_rfft(new_traj, new_curve, fftplans)
    return new_traj
