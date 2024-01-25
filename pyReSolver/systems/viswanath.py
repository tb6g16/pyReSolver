# This file contains the system definition for the system used in Viswanath
# (2001)

import numpy as np

# define optional arguments
parameters = {'mu': 0, 'r': 1}

def response(x, out, parameters = parameters):
    # assign response
    np.copyto(out[:, 0], x[:, 1] + (parameters['mu']*x[:, 0])*(parameters['r'] - np.sqrt((x[:, 0]**2) + (x[:, 1]**2))))
    np.copyto(out[:, 1], -x[:, 0] + (parameters['mu']*x[:, 1])*(parameters['r'] - np.sqrt((x[:, 0]**2) + (x[:, 1]**2))))

def jacobian(x, parameters = parameters):
    #initialise jacobian matrix
    jacobian = np.zeros([np.shape(x)[0], np.shape(x)[0]])

    # compute jacobian elements
    r = np.sqrt((x[0]**2) + (x[1]**2))
    jacobian[:, 0, 0] = parameters['mu']*(parameters['r'] - (2*(x[:, 0]**2) + (x[:, 1]**2))/r)
    jacobian[:, 0, 1] = 1 - (parameters['mu']*x[:, 0]*x[:, 1])/r
    jacobian[:, 1, 0] = -1 - (parameters['mu']*x[:, 0]*x[:, 1])/r
    jacobian[:, 1, 1] = parameters['mu']*(parameters['r'] - ((x[:, 0]**2) + 2*(x[:, 1]**2))/r)

    return np.squeeze(jacobian)

def nl_factor(x, out, parameters = parameters):
    # assign values to vector
    r = np.sqrt((x[:, 0]**2)+(x[:, 1]**2))
    np.copyto(out[:, 0], -parameters['mu']*x[:, 0]*r)
    np.copyto(out[:, 1], parameters['mu']*x[:, 1]*r)

def jac_conv(x, r, out, parameters = parameters):
    # compute response
    r = np.sqrt((x[0]**2) + (x[1]**2))
    np.copyto(out[:, 0], (parameters['mu']*(parameters['r'] - (2*(x[:, 0]**2) + (x[:, 1]**2))/r))*r[:, 0] + (1 - (parameters['mu']*x[:, 0]*x[:, 1])/r)*r[:, 1])
    np.copyto(out[:, 1], (-1 - (parameters['mu']*x[:, 0]*x[:, 1])/r)*r[:, 0] + (parameters['mu']*(parameters['r'] - ((x[:, 0]**2) + 2*(x[:, 1]**2))/r))*r[:, 1])

def jac_conv_adj(x, r, parameters = parameters):
    # compute response
    r = np.sqrt((x[0]**2) + (x[1]**2))
    np.copyto(out[:, 0], (parameters['mu']*(parameters['r'] - (2*(x[:, 0]**2) + (x[:, 1]**2))/r))*r[:, 0] + (-1 - (parameters['mu']*x[:, 0]*x[:, 1])/r)*r[:, 0])
    np.copyto(out[:, 1], (1 - (parameters['mu']*x[:, 0]*x[:, 1])/r)*r[:, 1] + (parameters['mu']*(parameters['r'] - ((x[:, 0]**2) + 2*(x[:, 1]**2))/r))*r[:, 1])
