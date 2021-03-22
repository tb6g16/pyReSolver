# This file contains the system definition for the system used in Viswanath
# (2001)

import numpy as np

# define optional arguments
defaults = {'mu': 0, 'r': 1}

def response(x: np.ndarray, defaults = defaults):
    
    # unpack default arguments
    mu = defaults['mu']
    rlim = defaults['r']

    # initialise vectors
    response = np.zeros(np.shape(x))

    # assign response
    response[0] = x[1] + (mu*x[0])*(rlim - np.sqrt((x[0]**2) + (x[1]**2)))
    response[1] = -x[0] + (mu*x[1])*(rlim - np.sqrt((x[0]**2) + (x[1]**2)))

    return response

def jacobian(x: np.ndarray, defaults = defaults):
    
    # unpack defaults
    mu = defaults['mu']
    rlim = defaults['r']

    #initialise jacobian matrix
    jacobian = np.zeros([np.shape(x)[0], np.shape(x)[0]])

    # compute jacobian elements
    r = np.sqrt((x[0]**2) + (x[1]**2))
    jacobian[0, 0] = mu*(rlim - (2*(x[0]**2) + (x[1]**2))/r)
    jacobian[0, 1] = 1 - (mu*x[0]*x[1])/r
    jacobian[1, 0] = -1 - (mu*x[0]*x[1])/r
    jacobian[1, 1] = mu*(rlim - ((x[0]**2) + 2*(x[1]**2))/r)

    return jacobian

def nl_factor(x: np.ndarray, defaults = defaults):
    
    # unpack defualts
    mu = defaults['mu']

    # initialise output vector
    nl_vector = np.zeros(np.shape(x))

    # assign values to vector
    r = np.sqrt((x[0]**2) + (x[1]**2))
    nl_vector[0] = -mu*x[0]*r
    nl_vector[1] = mu*x[1]*r

    return nl_vector
