import cProfile
import subprocess

import numpy as np

import pyReSolver

T = 10
M = int(10*T)
init_traj = pyReSolver.utils.generateRandomTrajectory(3, M)
init_freq = (2*np.pi)/T
mean = np.array([[0, 0, 23.64]])

B = np.array([[0, 0], [-1, 0], [0, 1]])
jac_at_mean = pyReSolver.systems.lorenz.jacobian(mean)
resolvents = pyReSolver.resolvent(init_freq, range(M), jac_at_mean, B)
psi, _, _ = pyReSolver.resolvent_modes(resolvents)

def do_op():
    _, _, _ = pyReSolver.minimiseResidual(init_traj, init_freq, pyReSolver.systems.lorenz, mean, method="L-BFGS-B", iter=200, psi=psi)
    # plot_traj(op_traj, discs = [10000], means = [mean])

def main():
    cProfile.run('do_op()', './tests/restats.prof')
    subprocess.run(['snakeviz', './tests/restats.prof'])

if __name__ == '__main__':
    main()
