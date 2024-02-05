# This file contains a short test to show that an initial trajectory is
# converged to something that looks correct.

import numpy as np

import pyReSolver

def main():
    period = 3.1
    mean = np.array([[0, 0, 23.64]])
    init_traj = pyReSolver.utils.generateRandomTrajectory(3, int(15*period))

    jac_at_mean = pyReSolver.systems.lorenz.jacobian(mean)
    B = np.array([[0, 0], [-1, 0], [0, 1]])
    resolvent_traj = pyReSolver.resolvent((2*np.pi)/period, range(init_traj.shape[0]), jac_at_mean, B)
    psi, _, _ = pyReSolver.resolvent_modes(resolvent_traj)

    opt_traj, _, _ = pyReSolver.minimiseResidual(init_traj, (2*np.pi)/period, pyReSolver.systems.lorenz, mean, iter=1000, method="L-BFGS-B", psi=psi)
    pyReSolver.plot_traj(opt_traj, discs = [100000], means = [mean])

if __name__ == '__main__':
    main()
