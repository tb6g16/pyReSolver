# This file contains the class definition to allow simple implementation of
# trajectory plotting in the plot_traj file.

class TrajPlotObject():
    """
        Class for the simple plotting data of a trajectory.

        Attributes
        ----------
        traj : Trajectory
            Trajectory to be plotted.
        disc : positive int
            The discretisation of the trajectory to be plotted.
        mean : ndarray
            1D array containing data of float type for the mean state of the
            trajectory.
    """
    
    __slots__ = ['traj', 'disc', 'mean']

    def __init__(self, traj, disc = None, mean = None):
        """
            Intialisation of TrajPlotObject instance.

            Parameters
            ----------
            traj : Trajectory
            disc : positive int
            mean : ndarray
        """
        self.traj = traj
        self.disc = disc
        self.mean = mean
