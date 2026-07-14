import src.config as config
from src.vehicle.trajectory import Trajectory
from src.physics.orbit import propogate_orbit_rk4
from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.vehicle.satellite import Satellite

import numpy as np

class Controller:
    """
    Implements a distributed model predictive control algorithm to open a slot in a satellite constellation
    """

    N: int = config.MPC_HORIZON_LENGTH         # MPC horizon length
    dt: float = config.MPC_TIME_STEP           # Time step for the MPC in seconds
    saftey_distance: float = config.SAFTEY_DISTANCE  # Minimum distance to maintain from other satellites in meters

    computedControl: np.ndarray = np.zeros(3)         # Computed control inputs
    computedTrajectory: Trajectory = None                 # Computed trajectory of the satellite

    neighboringSatTrajectories: dict[int, Trajectory] = {}   # Dictionary to hold neighboring satellite trajectories for MPC
    forignTrajectory: Trajectory               # Trajectory of object ot be avoided


    def __init__(self):
        pass

    # --- Temp for testing before controller is implemented ---
    def compute_control(self):
        return np.zeros(3)
    
    def recieve_neighboring_trajectories(self, neighborID: int, neighborTraj: Trajectory):
        """
        Receives the predicted trajectory of a neighboring satellite
        """
        self.neighboringSatTrajectories[neighborID] = neighborTraj

    def recieve_forign_trajectory(self, forignTraj: Trajectory, saftey_distance: Optional[float] = None):
        """
        Receives the predicted trajectory of a forign object to be avoided
        Optional: saftey distance can be specified to override the default saftey distance for this object
        """
        self.forignTrajectory = forignTraj
        if saftey_distance is not None:
            self.saftey_distance = saftey_distance

    def propogate_neighboring_trajectories(self):
        """
        In case of lost conenction, propogates the neighboring satellite trajectories
        Assume neighboring sat has equal mass
        """
        for neighborID, neighborTraj in self.neighboringSatTrajectories.items():
            
            
            # Propogate the trajectory forward by one time step
            propogate_orbit_rk4()
            

    def solve_mpc(self, sat: "Satellite"):
        """
        Solves the MPC optimization problem to compute the optimal control inputs for the satellite
        """
        pass

    def get_control_inputs(self, sat: "Satellite") -> np.ndarray:
        """
        Returns the control inputs for the satellite based on the MPC solution
        """
        pass