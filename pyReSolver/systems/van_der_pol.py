# This file contains the functions defining the dynamical system of the Van der
# Pol equations, the proposed solution curve, and the real solution (limit
# cycle).

import numpy as np

# define optional arguments
parameters = {'mu': 0.0}

def response(x, out, parameters = parameters):
    # assign response
    np.copyto(out[:, 0], x[:, 1])
    np.copyto(out[:, 1], (parameters['mu']*(1 - (x[:, 0]**2))*x[:, 1]) - x[:, 0])

def jacobian(x, parameters = parameters):
    #initialise jacobian matrix
    jacobian_matrix = np.zeros([np.shape(x)[0], np.shape(x)[1], np.shape(x)[1]])

    # compute jacobian elements
    jacobian_matrix[:, 0, 1] = 1
    jacobian_matrix[:, 1, 0] = -(2*parameters['mu']*x[:, 0]*x[:, 1]) - 1
    jacobian_matrix[:, 1, 1] = parameters['mu']*(1 - (x[:, 0]**2))

    return np.squeeze(jacobian_matrix)

def nl_factor(x, out, parameters = parameters):
    # assign values to vector
    out[:, 0] = 0
    np.copyto(out[:, 1], -parameters['mu']*(x[:, 0]**2)*x[:, 1])

def jac_conv(x, r, out, parameters = parameters):
    # compute response
    np.copyto(out[:, 0], r[:, 1])
    np.copyto(out[:, 1], (-(2*parameters['mu']*x[:, 0]*x[:, 1]) - 1)*r[:, 0] + (parameters['mu']*(1 - (x[:, 0]**2)))*r[:, 1])

def jac_conv_adj(x, r, out, parameters = parameters):
    # compute response
    np.copyto(out[:, 0], (-(2*parameters['mu']*x[:, 0]*x[:, 1]) - 1)*r[:, 1])
    np.copyto(out[:, 1], r[:, 0] + (parameters['mu']*(1 - (x[:, 0]**2)))*r[:, 1])
