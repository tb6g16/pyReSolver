# This file holds the function definition for the Lorenz system in the
# frequency domain.

import numpy as np

# define parameters
parameters = {'rho': 28, 'beta': 8/3, 'sigma': 10}

def response(x, out, defaults = parameters):
    # unpack defaults
    # TODO: does unpacking slow this down?
    rho = defaults['rho']
    beta = defaults['beta']
    sigma = defaults['sigma']

    # assign response
    np.copyto(out[:, 0], sigma*(x[:, 1] - x[:, 0]))
    np.copyto(out[:, 1], (rho*x[:, 0]) - x[:, 1] - (x[:, 0]*x[:, 2]))
    np.copyto(out[:, 2], (x[:, 0]*x[:, 1]) - (beta*x[:, 2]))

def jacobian(x, defaults = parameters):
    # unpack defaults
    rho = defaults['rho']
    beta = defaults['beta']
    sigma = defaults['sigma']

    # initialise jacobian matrix
    jacobian = np.zeros([np.shape(x)[0], np.shape(x)[1], np.shape(x)[1]])

    # compute jacobian elements
    jacobian[:, 0, 0] = -sigma
    jacobian[:, 0, 1] = sigma
    jacobian[:, 1, 0] = rho - x[:, 2]
    jacobian[:, 1, 1] = -1
    jacobian[:, 1, 2] = -x[:, 0]
    jacobian[:, 2, 0] = x[:, 1]
    jacobian[:, 2, 1] = x[:, 0]
    jacobian[:, 2, 2] = -beta

    return np.squeeze(jacobian)

def nl_factor(x, out, defaults = parameters):
    # assign values
    out[:, 0] = 0
    np.copyto(out[:, 1], -x[:, 0]*x[:, 2])
    np.copyto(out[:, 2], x[:, 0]*x[:, 1])

def jac_conv(x, r, defaults = parameters):
    # unpack defaults
    rho = defaults['rho']
    beta = defaults['beta']
    sigma = defaults['sigma']

    # initialise response
    response = np.zeros_like(x)

    # compute response
    response[:, 0] = -sigma*r[:, 0] + sigma*r[:, 1]
    response[:, 1] = (rho - x[:, 2])*r[:, 0] - r[:, 1] - x[:, 0]*r[:, 2]
    response[:, 2] = x[:, 1]*r[:, 0] + x[:, 0]*r[:, 1] - beta*r[:, 2]

    return response

def jac_conv_adj(x, r, defaults = parameters):
    # unpack defaults
    rho = defaults['rho']
    beta = defaults['beta']
    sigma = defaults['sigma']

    # initialise response
    response = np.zeros_like(x)

    # compute response
    response[:, 0] = -sigma*r[:, 0] + (rho - x[:, 2])*r[:, 1] + x[:, 1]*r[:, 2]
    response[:, 1] = sigma*r[:, 0] - r[:, 1] + x[:, 0]*r[:, 2]
    response[:, 2] = -x[:, 0]*r[:, 1] - beta*r[:, 2]

    return response
