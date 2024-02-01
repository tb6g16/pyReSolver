# This file contains the definition for a class that holds a cache of all the
# objects that are required at each iteration of the optimisation procedure.
# This is done for the purpose of organising exactly what arrays have been
# allocated in memory.

import numpy as np

from .Trajectory import Trajectory
from .traj2vec import init_comp_vec
from .trajectory_functions import transpose, conj

class Cache:

    __slots__ = ['traj', 'traj_grad', 'lr', 'lr_grad', 'f', 'tmp_conv',
                'red_traj', 'tmp_t1', 'tmp_t2', 'tmp_inner', 'resp_mean']

    def __init__(self, traj, mean, sys, fftplans, psi = None):
        self.traj = traj
        self.tmp_inner = Trajectory(np.einsum('ij,ij->i', traj, traj))
        self.traj_grad = np.zeros_like(self.traj)
        self.lr = np.zeros_like(self.traj)
        self.lr_grad = np.zeros_like(self.traj)
        self.f = np.zeros_like(self.traj)
        self.tmp_conv = np.zeros_like(self.traj)
        self.tmp_t1 = np.copy(fftplans.tmp_t)
        self.tmp_t2 = np.copy(fftplans.tmp_t)
        if psi is not None:
            self.red_traj = Trajectory(np.zeros_like(np.einsum('ikl,il->ik', transpose(conj(psi)), self.traj)))
        else:
            self.red_traj = None
        self.resp_mean = np.zeros_like(mean)
        sys.response(mean, self.resp_mean)
