# This file contains the definition for a elliptic curve parameterised with
# respect to s

import numpy as np

# define trajectory
def x(s: float):

    # initialise vectors
    state = np.zeros([2])

    # define state at given s
    state[0] = 2*np.cos(s)
    state[1] = -np.sin(s)

    return state
