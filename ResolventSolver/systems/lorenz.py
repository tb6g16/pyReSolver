# This file holds the function definition for the Lorenz system in the
# frequency domain.

import numpy as np

# define parameters
parameters = {'rho': 28, 'beta': 8/3, 'sigma': 10}

def response(x, defaults = parameters):
    # unpack defaults
    rho = defaults['rho']
    beta = defaults['beta']
    sigma = defaults['sigma']

    # intialise response vector
    response = np.zeros(np.shape(x))

    # assign response
    response[:, 0] = sigma*(x[:, 1] - x[:, 0])
    response[:, 1] = (rho*x[:, 0]) - x[:, 1] - (x[:, 0]*x[:, 2])
    response[:, 2] = (x[:, 0]*x[:, 1]) - (beta*x[:, 2])

    return response

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

def nl_factor(x, defaults = parameters):
    # initialise output vector
    nl_vector = np.zeros(np.shape(x))

    # assign values
    nl_vector[:, 1] = -x[:, 0]*x[:, 2]
    nl_vector[:, 2] = x[:, 0]*x[:, 1]

    return nl_vector

# def nl_factor2(x, defaults = parameters):
#     # initialise output vector
#     nl_vector = np.zeros(2)

#     # assign values
#     nl_vector[0] = x[0]*x[2]
#     nl_vector[1] = x[0]*x[1]

#     return nl_vector

# def nl_factor_derv(x, defaults = parameters):
#     # initialise output matrix
#     nl_derv_mat = np.zeros([2, 3])

#     # assign values
#     nl_derv_mat[0, 0] = x[2]
#     nl_derv_mat[0, 2] = x[0]
#     nl_derv_mat[1, 0] = x[1]
#     nl_derv_mat[1, 1] = x[0]

#     return nl_derv_mat
