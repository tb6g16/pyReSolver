# This file contains the functions defining the dynamical system of the Van der
# Pol equations, the proposed solution curve, and the real solution (limit
# cycle).

import numpy as np

# define optional arguments
parameters = {'mu': 0}

def response(x, parameters = parameters):
    # unpack defaults
    mu = parameters['mu']

    # initialise response vector
    response = np.zeros(np.shape(x))

    # assign response
    response[:, 0] = x[:, 1]
    response[:, 1] = (mu*(1 - (x[:, 0]**2))*x[:, 1]) - x[:, 0]

    return response

def jacobian(x, parameters = parameters):
    # unpack defaults
    mu = parameters['mu']

    #initialise jacobian matrix
    jacobian_matrix = np.zeros([np.shape(x)[0], np.shape(x)[1], np.shape(x)[1]])

    # compute jacobian elements
    jacobian_matrix[:, 0, 1] = 1
    jacobian_matrix[:, 1, 0] = -(2*mu*x[:, 0]*x[:, 1]) - 1
    jacobian_matrix[:, 1, 1] = mu*(1 - (x[:, 0]**2))

    return np.squeeze(jacobian_matrix)

def nl_factor(x, parameters = parameters):
    # unpack defualts
    mu = parameters['mu']

    # initialise output vector
    nl_vector = np.zeros(np.shape(x))

    # assign values to vector
    nl_vector[:, 1] = -mu*(x[:, 0]**2)*x[:, 1]

    return nl_vector

def jac_conv(x, r, parameters = parameters):
    # unpack defaults
    mu = parameters['mu']

    # initialise response
    response = np.zeros_like(x)

    # compute response
    response[:, 0] = r[:, 1]
    response[:, 1] = (-(2*mu*x[:, 0]*x[:, 1]) - 1)*r[:, 0] + (mu*(1 - (x[:, 0]**2)))*r[:, 1]

    return response

def jac_conv_adj(x, r, parameters = parameters):
    # unpack defaults
    mu = parameters['mu']

    # initialise response
    response = np.zeros_like(x)

    # compute response
    response[:, 0] = (-(2*mu*x[:, 0]*x[:, 1]) - 1)*r[:, 1]
    response[:, 1] = r[:, 0] + (mu*(1 - (x[:, 0]**2)))*r[:, 1]

    return response
