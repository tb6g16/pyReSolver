# This file contains wrapper functions for the pickle module to easily read and
# write trajectories to files. There are also a few other similar functions for
# saving other useful information about the trajectories that may otherwise be
# time consuming to calculate repeatedly.

import numpy as np
import pickle

def gen_filename_from_base(base_filename, no_files, file_no):
    # how many leading should be present
    leading_zeros = int(np.log10(no_files)) + 1

    # concatenate the number of the file with leading zeros
    filename = base_filename + str(file_no).zfill(leading_zeros)

    return filename

def write_traj(base_filename, *args):
    # make sure the trajectories are passed as a mode-frequency pair in an
    # iterable for each argument

    # loop through provided trajectories
    for i, arg in enumerate(args):
        # determine filename from base filename
        if len(args) == 1:
            filename = base_filename
        else:
            filename = gen_filename_from_base(base_filename, len(args), i + 1)

        # open a file
        with open(filename, 'wb') as f:
            # write to the file
            pickle.dump(arg, f)

def read_traj(filename):
    # open the file and load its contents
    with open(filename, 'rb') as f:
        traj, freq = pickle.load(f)

    return traj, freq

def write_hist(base_filename, *args):
    # loop through provided histograms
    for i, arg in enumerate(args):
        # determine filename from base filename
        if len(args) == 1:
            filename = base_filename
        else:
            filename = gen_filename_from_base(base_filename, len(args), i + 1)
    
    # open a file
    with open(filename, 'wb') as f:
        # write to the file
        pickle.dump(arg, f)

def read_hist(filename):
    # open the file and load its contents
    with open(filename, 'rb') as f:
        hist, x = pickle.load(f)
    
    return hist, x
