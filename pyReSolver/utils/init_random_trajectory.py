# This file generates a random trajectory using a given distribution.

import numpy as np

from pyReSolver.Trajectory import Trajectory

def generateRandomTrajectory(dimensions, numberOfModes, distribution=np.random.standard_normal):
    return Trajectory(distribution([numberOfModes, dimensions]) + 1j*distribution([numberOfModes, dimensions]))
