import sys
sys.path.append(r"C:\Users\user\Desktop\PhD\Bruno Paper\Code\Approach A")
from Trajectory import Trajectory
from System import System
from test_cases import lorenz
from residual_functions import local_residual, global_residual, global_residual_grad
import numpy as np
from my_fft import my_fft, my_ifft

traj = np.zeros([3, 10], dtype = complex)
traj = Trajectory(traj)
traj[:, 1] = [4+2j, 6+2j, 0]
traj[:, 4] = [1, -3, 4]
mean = [0, 0, 25]
freq = (2*np.pi)/2.1
sys = System(lorenz)

# print(np.shape(my_ifft(traj.modes)))

# print(local_residual(traj, sys, freq, mean).modes[:, 2])
# print(global_residual(traj, sys, freq, mean))
# print(global_residual_grad(traj, sys, freq, mean)[0].modes[:, 3])
print(global_residual_grad(traj, sys, freq, mean)[1])
