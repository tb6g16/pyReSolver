# This file contains the function definitions to be able to read and write
# trajectories using the HDF5 file format.

import h5py
import numpy as np

from Trajectory import Trajectory
from my_fft import my_rfft

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
    # open the file
    with h5py.File(filename, 'w') as f:
        f.create_dataset('traj', data = traj.modes)
        f.create_dataset('freq', data = np.array(freq))

def read_traj(filename):
    pass

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
            Base frequency of the trajectory.
    """
    data = h5py.File(filename, 'r')
    traj = my_rfft(data['X'])
    freq = data['ω'][()]
    return Trajectory(traj), freq

if __name__ == '__main__':
    from gen_rand_traj import gen_rand_traj
    rand_traj = gen_rand_traj(3, 50)
    write_traj('abc.hdf5', rand_traj, 2)
