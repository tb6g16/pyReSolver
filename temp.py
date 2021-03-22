from Trajectory import Trajectory
from System import System
from systems import lorenz
from residual_functions import local_residual, global_residual, gr_traj_grad, gr_freq_grad
import numpy as np
from my_fft import my_rfft, my_irfft
from traj_util import list2array, array2list

traj = np.zeros([10, 3], dtype = complex)
traj = Trajectory(array2list(traj))
traj[1] = np.array([4+2j, 6+2j, 0])
traj[4] = np.array([1, -3, 4])
mean = [0, 0, 25]
freq = (2*np.pi)/2.1
sys = System(lorenz)

# print(np.shape(my_irfft(list2array(traj.mode_list))))

# print(local_residual(traj, sys, freq, mean)[2])
# print(global_residual(traj, sys, freq, mean))
# print(gr_traj_grad(traj, sys, freq, mean)[3])
print(gr_freq_grad(traj, sys, freq, mean))
