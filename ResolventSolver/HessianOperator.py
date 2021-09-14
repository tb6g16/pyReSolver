# This file contains the definition of the Hessian operator class as a subclass
# of the LinearOperator abstract class given by the SciPy sparse linear algebra
# library.

import numpy as np
from scipy.sparse.linalg import LinearOperator

from ResolventSolver.traj2vec import traj2vec, vec2traj
from ResolventSolver.my_min import init_opt_funcs

class HessianOperator(LinearOperator):

    def __init__(self, traj, sys, freq, mean):
        self.state = traj2vec(traj, freq)
        self.dim = traj.shape[1]
        _, self.grad_func = init_opt_funcs(sys, traj.shape[1], mean)
        self.shape = (np.shape(self.state)[0] - 1, np.shape(self.state)[0] - 1)
        self.dtype = np.dtype('float64')

    def _matvec(self, v):
        """
            Return the Hessian-vector product by taking the difference of the
            gradients.
        """
        return self.grad_func(self.state + v) - self.grad_func(self.state)

    def _rmatvec(self, v):
        """
            Return the matrix-vector product of an arbitrary vector and the
            adjoint of the Hessian (equivalent to left multiplication of a
            row vector).
        """
        return self.grad_func(self.state + v) - self.grad_func(self.state)

    @property
    def traj(self):
        return vec2traj(self.state, self.dim)[0]

    @traj.setter
    def traj(self, new_traj):
        self.dim = new_traj.shape[1]
        self.state = traj2vec(new_traj, self.state[-1])
        self.shape = (np.shape(self.state)[0] - 1, np.shape(self.state)[0] - 1)

    @property
    def freq(self):
        return vec2traj(self.state, self.dim)[1]

    @freq.setter
    def freq(self, new_freq):
        self.state[-1] = new_freq

    # THE PREVIOUS CODE CAN BE PUT HERE FOR SAFE KEEPING
    @property
    def hess_matrix(self):
        pass
