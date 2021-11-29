# This file contains the definition for a class that holds a cache of all the
# objects that are required at each iteration of the optimisation procedure.
# This is done for the purpose of organising exactly what arrays have been
# allocated in memory.

# To get this going, I need to list ALL of the arrays I need to complete a
# single iteration of the optimisation loop. These arrays are re-used over each
# iteration.

from ResolventSolver.Trajectory import Trajectory
from ResolventSolver.traj2vec import init_comp_vec

class Cache:

    def __init__(self, traj_shape, freq, sys, mean, psi, fftplans):
        self.tmp_traj = Trajectory(np.zeros(traj_shape), dtype=complex)
        self.tmp_traj_grad = np.zeros_like(self.tmp_traj)
        self.lr = np.zeros_like(self.tmp_traj)
        self.lr_grad = np.zeros_like(self.tmp_traj)
        self.gr_grad = np.zeros_like(self.tmp_traj)
        self.f = np.zeros_like(self.tmp_traj)
        self.tmp_conv = np.zeros_like(self.tmp_traj)
        self.tmp_red_traj = Trajectory(np.zeros_like(np.einsum('ikl,il->ik', psi, self.tmp_traj)))
        self.vec = init_comp_vec(self.tmp_traj)
        self.tmp_t1 = np.copy(fftplans.tmp_t)
        self.tmp_t2 = np.copy(fftplans.tmp_t)
