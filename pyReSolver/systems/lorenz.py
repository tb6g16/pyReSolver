# This file holds the function definition for the Lorenz system in the
# frequency domain.

import numpy as np

# define parameters
parameters = {'rho': 28.0, 'beta': 8/3, 'sigma': 10.0}

def response(x, out, defaults = parameters):
    # assign response
    np.copyto(out[:, 0], defaults['sigma']*(x[:, 1] - x[:, 0]))
    np.copyto(out[:, 1], (defaults['rho']*x[:, 0]) - x[:, 1] - (x[:, 0]*x[:, 2]))
    np.copyto(out[:, 2], (x[:, 0]*x[:, 1]) - (defaults['beta']*x[:, 2]))

def jacobian(x, defaults = parameters):
    # initialise jacobian matrix
    jacobian = np.zeros([np.shape(x)[0], np.shape(x)[1], np.shape(x)[1]])

    # compute jacobian elements
    jacobian[:, 0, 0] = -defaults['sigma']
    jacobian[:, 0, 1] = defaults['sigma']
    jacobian[:, 1, 0] = defaults['rho'] - x[:, 2]
    jacobian[:, 1, 1] = -1
    jacobian[:, 1, 2] = -x[:, 0]
    jacobian[:, 2, 0] = x[:, 1]
    jacobian[:, 2, 1] = x[:, 0]
    jacobian[:, 2, 2] = -defaults['beta']

    return np.squeeze(jacobian)

def nl_factor(x, out, defaults = parameters):
    # assign values
    out[:, 0] = 0
    np.copyto(out[:, 1], -x[:, 0]*x[:, 2])
    np.copyto(out[:, 2], x[:, 0]*x[:, 1])

def jac_conv(x, r, out, defaults = parameters):
    # compute response
    np.copyto(out[:, 0], -defaults['sigma']*r[:, 0] + defaults['sigma']*r[:, 1])
    np.copyto(out[:, 1],(defaults['rho'] - x[:, 2])*r[:, 0] - r[:, 1] - x[:, 0]*r[:, 2])
    np.copyto(out[:, 2], x[:, 1]*r[:, 0] + x[:, 0]*r[:, 1] - defaults['beta']*r[:, 2])

def jac_conv_adj(x, r, out, defaults = parameters):
    # compute response
    np.copyto(out[:, 0], -defaults['sigma']*r[:, 0] + (defaults['rho']- x[:, 2])*r[:, 1] + x[:, 1]*r[:, 2])
    np.copyto(out[:, 1], defaults['sigma']*r[:, 0] - r[:, 1] + x[:, 0]*r[:, 2])
    np.copyto(out[:, 2], -x[:, 0]*r[:, 1] - defaults['beta']*r[:, 2])
