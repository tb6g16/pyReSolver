# This file contains a short test to show that an initial trajectory is
# converged to something that looks correct.

import numpy as np

from ResolventSolver.my_min import my_min
from ResolventSolver.resolvent_modes import resolvent, resolvent_modes
from ResolventSolver.gen_rand_traj import gen_rand_traj
from ResolventSolver.plot_traj import plot_traj
from ResolventSolver.systems import lorenz

def main():
    period = 3.1
    mean = np.array([[0, 0, 23.64]])
    init_traj = gen_rand_traj(3, 10*period)

    jac_at_mean = lorenz.jacobian(mean)
    B = np.array([[0, 0], [-1, 0], [0, 1]])
    resolvent_traj = resolvent((2*np.pi)/period, range(init_traj.shape[0]), jac_at_mean, B)
    psi, _, _ = resolvent_modes(resolvent_traj)

    opt_traj, _, _, _ = my_min(init_traj, (2*np.pi)/period, lorenz, mean, iter = 5000, method = 'CG', psi = psi)
    plot_traj(opt_traj, discs = [100000], means = [mean], proj = 'xz')

if __name__ == '__main__':
    main()