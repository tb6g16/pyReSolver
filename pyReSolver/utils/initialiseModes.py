# This file contains a utility function to initialise a set of Resolvent modes
# to be used for an optimisation.

import numpy as np

from ..resolvent_modes import resolvent, resolvent_modes

def initialiseModes(period, mean, system, numberOfModes):
    jacobianAtMean = system.jacobian(mean)
    massMatrix = np.array([[0, 0], [-1, 0], [0, 1]])
    return resolvent_modes(resolvent((2*np.pi)/period, range(numberOfModes), jacobianAtMean, massMatrix))[0]
