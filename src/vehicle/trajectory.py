import numpy as np

class Trajectory:
    """
    Represents the trajectory of any object in ECI frame
    """

    positionECI: np.ndarray              # Position of the object in ECI frame in meters
    velocityECI: np.ndarray              # Velocity of the object in ECI frame in m/s
    time: np.ndarray                     # Time vector corresponding to the trajectory points

    def __init__(self, ECI_position_in: np.ndarray, ECI_velocity_in: np.ndarray, time_in: np.ndarray):
        """
        Constructor for Trajectory class
        """
        if len(time_in) < 2:
            raise ValueError("Trajectory must contain at least 2 points to support interpolation.")

        # Ensure time is monotonically increasing
        if not np.all(np.diff(time_in) > 0):
            raise ValueError("Time vector must be monotonically increasing.")

        self.positionECI = ECI_position_in
        self.velocityECI = ECI_velocity_in
        self.time = time_in

    def get_state_at_time(self, query_time: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the position and velocity of the object at a specific time.
        If the exact time is not in the trajectory, it linearly interpolates between the two nearest points.
        """
        if query_time < self.time[0] or query_time > self.time[-1]:
            raise ValueError("Query time is out of bounds of the trajectory.")

        # Find the index of the closest time point
        idx = np.searchsorted(self.time, query_time)

        if idx == 0:
            return self.positionECI[0], self.velocityECI[0]
        elif idx == len(self.time):
            return self.positionECI[-1], self.velocityECI[-1]
        else:
            # Interpolate between idx-1 and idx
            t0, t1 = self.time[idx - 1], self.time[idx]
            p0, p1 = self.positionECI[idx - 1], self.positionECI[idx]
            v0, v1 = self.velocityECI[idx - 1], self.velocityECI[idx]

            # Linear interpolation
            alpha = (query_time - t0) / (t1 - t0)
            position_interp = (1 - alpha) * p0 + alpha * p1
            velocity_interp = (1 - alpha) * v0 + alpha * v1

            return position_interp, velocity_interp
