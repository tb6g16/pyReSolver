from Trajectory import Trajectory
from System import System
from systems import lorenz
from residual_functions import local_residual, global_residual, global_residual_grad
import numpy as np
from my_fft import my_rfft, my_irfft
from traj_util import list2array, array2list

traj = np.zeros([10, 3], dtype = complex)
traj = Trajectory(array2list(traj))
traj[1] = [4+2j, 6+2j, 0]
# traj[4] = [1j, -3, 4+7j]
mean = [0, 0, 25]
freq = (2*np.pi)/2.1
sys = System(lorenz)

# print(np.shape(my_irfft(list2array(traj.mode_list))))

print(local_residual(traj, sys, freq, mean)[1])
# print(global_residual(traj, sys, freq, mean))
# print(global_residual_grad(traj, sys, freq, mean)[0].modes[0])