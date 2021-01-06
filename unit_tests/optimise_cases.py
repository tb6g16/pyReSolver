# This file contains a number of test cases for the optimisation methods
# implemented, such as to validate the behaviour.

import sys
sys.path.append(r"C:\Users\user\Desktop\PhD\Bruno Paper\Code\RA Dynamical System")
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from Trajectory import Trajectory
from System import System
import optimise as my_opt
from traj2vec import traj2vec, vec2traj
from test_cases import unit_circle as uc
from test_cases import ellipse as elps
from test_cases import van_der_pol as vpd
from test_cases import viswanath as vis

trace = []

def callback(x):
    trace.append(res_func(x))

# Case 1: single displaced point
n = 1
i = 2
a = 0
m = 1.2
freq = 1
dim = 2
mean = np.zeros([2, 1])
circle = Trajectory(uc.x)
circle_almost = circle.curve_array
circle_almost[n, i] = m*circle.curve_array[n, i] + a
circle_almost = Trajectory(circle_almost)
sys = System(vpd)

res_func, jac_func = my_opt.init_opt_funcs(sys, dim)
constraints = my_opt.init_constraints(sys, dim, mean)

op_vec = opt.minimize(res_func, traj2vec(circle_almost, freq), jac = jac_func, method = 'L-BFGS-B', callback = callback)
print(op_vec.message)
print("Number of iterations: " + str(op_vec.nit))

op_traj, op_freq = vec2traj(op_vec.x, dim)

print(res_func(traj2vec(circle_almost, freq)))
print(res_func(traj2vec(op_traj, op_freq)))
print(op_freq)

# circle_almost.plot(gradient = True, gradient_density = 32/256)
# op_traj.plot(gradient = True, gradient_density = 32/256)

plt.figure()
plt.plot(trace)
plt.show()

# Case 2: non-unit circle with viswanath system

# Case 3: circle input to nonlinear vpd and result input to linear vpd

# Case 4: vpd result input to viswanath system to get circle back out
