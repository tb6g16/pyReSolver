# This file contains the function definitions to be able to read and write
# trajectories using the HDF5 file format.

import h5py
import numpy as np

from pyReSolver.Trajectory import Trajectory
from pyReSolver.my_fft import my_rfft

def write_traj(filename, traj, freq):
    """
        Write trajectory, with its frequency, to disk using the HDF5 format.

        Parameters
        ----------
        filename : str
            String for the name of the file to which the trajectory is written.
        traj : Trajectory
        freq : positive float
            Base frequency of the given trajectory.
    """
    # open file and write data
    with h5py.File(filename, 'w') as f:
        f.create_dataset('traj', data = traj.modes)
        f.create_dataset('freq', data = np.array(freq))

def read_traj(filename):
    """
        Returns the trajectory frequency pair stored in an HDF5 file.

        Parameters
        ----------
        filename : str

        Returns
        -------
        traj : Trajectory
        freq : positive float
            Base frequency for the given trajectory.
    """
    # open file and read data
    with h5py.File(filename, 'r') as f:
        traj = Trajectory(np.array(f['traj']))
        freq = f['freq'][()]

    return traj, freq

def read_traj_davide(filename):
    """
        Return the trajectory and frequency stored in a HDF5 file (saved by
        Davide).

        Parameters
        ----------
        filename : str

        Returns
        -------
        traj : Trajectory
        freq : positive float
            Base frequency for the given trajectory.
    """
    # open file and read data
    with h5py.File(filename, 'r') as f:
        traj = Trajectory(my_rfft(f['X']))
        freq = f['ω'][()]

    return traj, freq
