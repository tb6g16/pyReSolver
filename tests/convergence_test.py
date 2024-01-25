# This file contains a short test to show that an initial trajectory is
# converged to something that looks correct.

import numpy as np

from pyReSolver.FFTPlans import FFTPlans
from pyReSolver.my_min import my_min
from pyReSolver.resolvent_modes import resolvent, resolvent_modes
from pyReSolver.gen_rand_traj import gen_rand_traj
from pyReSolver.plot_traj import plot_traj
from pyReSolver.systems import lorenz

def main():
    # period = 3.1
    period = 50.0
    mean = np.array([[0, 0, 23.64]])
    init_traj = gen_rand_traj(3, 15*period)
    plans = FFTPlans([(init_traj.shape[0] - 1) << 1, init_traj.shape[1]], flag='FFTW_ESTIMATE')

    jac_at_mean = lorenz.jacobian(mean)
    B = np.array([[0, 0], [-1, 0], [0, 1]])
    resolvent_traj = resolvent((2*np.pi)/period, range(init_traj.shape[0]), jac_at_mean, B)
    psi, _, _ = resolvent_modes(resolvent_traj)

    opt_traj, _, _ = my_min(init_traj, (2*np.pi)/period, lorenz, mean, iter = 1000, method = 'L-BFGS-B', psi = psi, plans = plans)
    plot_traj(opt_traj, discs = [100000], means = [mean])
    # plot_traj(opt_traj, discs = [100000], means = [mean], proj = 'xz', save = 'test.pdf')

if __name__ == '__main__':
    main()
