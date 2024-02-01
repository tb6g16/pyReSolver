import numpy as np

def ellipse(s: float):

    # initialise vectors
    state = np.zeros([2])

    # define state at given s
    state[0] = 2*np.cos(s)
    state[1] = -np.sin(s)

    return state

def unit_circle_3d(s: float):

    # initialise vectors
    state = np.zeros(3)

    # define function behaviour
    state[0] = np.cos(s)
    state[1] = -np.sin(s)

    return state

def unit_circle(s: float):

    # initialise vectors
    state = np.zeros([2])

    # define function behaviour
    state[0] = np.cos(s)
    state[1] = -np.sin(s)

    return state
