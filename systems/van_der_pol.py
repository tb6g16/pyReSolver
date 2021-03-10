# This file contains the functions defining the dynamical system of the Van der
# Pol equations, the proposed solution curve, and the real solution (limit
# cycle).

import numpy as np

# define optional arguments
parameters = {'mu': 0}

def response(x: np.ndarray, parameters = parameters):
    # unpack defaults
    mu = parameters['mu']

    # initialise response vector
    response = np.zeros(np.shape(x))

    # assign response
    response[0] = x[1]
    response[1] = (mu*(1 - (x[0] ** 2))*x[1]) - x[0]

    return response

def jacobian(x: np.ndarray, parameters = parameters):
    # unpack defaults
    mu = parameters['mu']

    #initialise jacobian matrix
    jacobian_matrix = np.zeros([np.shape(x)[0], np.shape(x)[0]])

    # compute jacobian elements
    jacobian_matrix[0, 1] = 1
    jacobian_matrix[1, 0] = -(2*mu*x[0]*x[1]) - 1
    jacobian_matrix[1, 1] = mu*(1 - (x[0] ** 2))

    return jacobian_matrix

def nl_factor(x: np.ndarray, parameters = parameters):
    # unpack defualts
    mu = parameters['mu']

    # initialise output vector
    nl_vector = np.zeros(np.shape(x))

    # assign values to vector
    nl_vector[1] = -mu*(x[0]**2)*x[1]

    return nl_vector

# set dimension attributes for functions
setattr(response, 'dim', 2)
setattr(jacobian, 'dim', 2)
setattr(nl_factor, 'dim', 2)
