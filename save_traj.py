# This file contains wrapper functions for the pickle module to easily read and
# write trajectories to files.

import numpy as np
import pickle

def write_traj(base_filename, *args):
    # make sure the trajectories are passed as a mode-frequency pair in an
    # iterable for each argument

    # how many leading zeros should there be
    leading_zeros = int(np.log10(len(args))) + 1

    # loop through provided trajectories
    for i, arg in enumerate(args):
        # determine filename from base filename
        if len(args) == 1:
            filename = base_filename
        else:
            filename = base_filename + str(i + 1).zfill(leading_zeros)

        # open a file
        with open(filename, 'wb') as f:
            # write to the file
            pickle.dump(arg, f)

def read_traj(filename):
    # open the file and load its contents
    with open(filename, 'rb') as f:
        traj, freq = pickle.load(f)

    return traj, freq
