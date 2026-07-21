from src.vehicle.controller import Controller
from src.vehicle.thruster import Thruster
import src.simulation.config as config
from src.physics.orbit import step_sat_orbit, generate_orbit                                                                                                                          

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.vehicle.crosslink import Crosslink
import numpy as np

class Satellite:
    """
    Class representing a satellite vehicle
    """

    id: int                                  # Unique identifier for the satellite

    # Each sat has a neighboring leading and trailing sat
    leadingSat: "Satellite"                  # Leading satellite in the constellation
    leadingConnection: bool = False          # Whether the satellite is connected to its leading satellite
    trailingSat: "Satellite"                 # Trailing satellite in the constellation
    trailingConnection: bool = False         # Whether the satellite is connected to its leading satellite
    leading_link: "Crosslink" = None         # Physical crosslink object to leading neighbor
    trailing_link: "Crosslink" = None        # Physical crosslink object to trailing neighbor


    # Vehicle state variables
    positionECI: np.ndarray                  # Position of the satellite in ECI frame in meters
    velocityECI: np.ndarray                  # Velocity of the satellite in ECI frame in m/s
    time: float                              # Current time in seconds
    
    
    # Spacecraft properties
    mass: float                              # Mass of the satellite in kg
    crosslinkGimbalRange: float              # Gimbal range of the crosslink in radians
    controller: Controller                   # Controller for the satellite
    thruster: Thruster                       # Thruster for the satellite
    applied_thrust: np.ndarray               # Cached thrust applied during the step


    def __init__(self, id: int, mass: float, altitude: float, longitude: float):
        """
        Constructor for a basic satellite with default properties:

        Inputs: 
        id: Unique identifier for the satellite
        altitude: Altitude of the satellite in meters above Earth's surface
        longitude: Initial longitude of the satellite in degrees
            
        Default properties:
        time = 0 s
        default crosslink gimbal range of 10 degrees
        """
        self.id = id
        self.mass = config.SATELLITE_MASS
        self.positionECI, self.velocityECI = generate_orbit(altitude, longitude)  
        self.time = 0.0                                 # Initial time
        self.crosslinkGimbalRange = np.radians(10)  # 10 degree gimbal range
        self.thruster = Thruster(max_thrust = config.MAX_THRUST)  # Default max thrust of 0.1 N
        self.controller = Controller(self.positionECI.copy(), self.velocityECI.copy())
        self.leading_link = None
        self.trailing_link = None
        self.applied_thrust = np.zeros(3)


    def get_control_inputs(self) -> np.ndarray:
        """
        Returns the last computed control inputs (cached) without re-solving MPC
        """
        return self.applied_thrust
    

    def step(self, dt: float):
        """
        Advances the state of the satellite by a time step dt
        """
        # Get control inputs
        self.applied_thrust = self.controller.solve_mpc(
            time=self.time,
            pos=self.positionECI,
            vel=self.velocityECI,
            mass=self.mass,
            max_thrust=self.thruster.max_thrust
        )

        # Propogate vehicle
        step_sat_orbit(self, self.applied_thrust, dt)
