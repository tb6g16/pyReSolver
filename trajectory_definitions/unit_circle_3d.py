# This file will hold the functions that define dynamical system and solution
# curve for testing purposes.

import numpy as np

# define a test solution
def x(s: float):

    # initialise vectors
    state = np.zeros(3)

    # define function behaviour
    state[0] = np.cos(s)
    state[1] = -np.sin(s)

    return state
