import cProfile
import subprocess

import numpy as np

from pyReSolver.FFTPlans import FFTPlans
from pyReSolver.my_min import my_min
from pyReSolver.utils import lorenz
from pyReSolver.gen_rand_traj import gen_rand_traj
from pyReSolver.plot_traj import plot_traj
from pyReSolver.resolvent_modes import resolvent, resolvent_modes

T = 10
M = int(10*T)
init_traj = gen_rand_traj(3, M)
init_freq = (2*np.pi)/T
plans = FFTPlans([(init_traj.shape[0] - 1) << 1, init_traj.shape[1]])
mean = np.array([[0, 0, 23.64]])

B = np.array([[0, 0], [-1, 0], [0, 1]])
jac_at_mean = lorenz.jacobian(mean)
resolvents = resolvent(init_freq, range(M), jac_at_mean, B)
psi, _, _ = resolvent_modes(resolvents)

def do_op():
    _, _, _ = my_min(init_traj, init_freq, lorenz, mean, method = 'CG', plans = plans, iter = 200, psi = psi)
    # plot_traj(op_traj, discs = [10000], means = [mean])

def main():
    cProfile.run('do_op()', './tests/restats.prof')
    subprocess.run(['snakeviz', './tests/restats.prof'])

if __name__ == '__main__':
    main()
