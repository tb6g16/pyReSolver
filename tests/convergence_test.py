# This file contains a short test to show that an initial trajectory is
# converged to something that looks correct.

import numpy as np

import pyReSolver

def main():
    period = 3.1
    mean = np.array([[0, 0, 23.64]])
    init_traj = pyReSolver.utils.generateRandomTrajectory(3, int(15*period))
    psi = pyReSolver.utils.initialiseModes(period, mean, pyReSolver.systems.lorenz, init_traj.shape[0])

    opt_traj, trace, _ = pyReSolver.minimiseResidual(init_traj, (2*np.pi)/period, pyReSolver.systems.lorenz, mean, method="L-BFGS-B", psi=psi, options={"maxiter": 1000, "disp": True})
    pyReSolver.plot_traj(opt_traj, discs = [100000], means = [mean])

if __name__ == '__main__':
    main()
