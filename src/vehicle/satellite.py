from src.vehicle.controller import Controller
from src.vehicle.thruster import Thruster
from src.physics.orbit import propogate_orbit, generate_orbit                                                                                                                          

import numpy as np

class Satellite:
    """
    Class representing a satellite vehicle
    """

    id: int                                  # Unique identifier for the satellite

    # Vehicle state variables
    positionECI: np.ndarray                  # Position of the satellite in ECI frame in meters
    velocityECI: np.ndarray                  # Velocity of the satellite in ECI frame in m/s
    time: float                              # Current time in seconds
    
    
    # Spacecraft properties
    mass: float                              # Mass of the satellite in kg
    crosslinkGimbalRange: float              # Maximum gimbal range for front/rear crosslink laser in radians

    controller: Controller                   # Controller for the satellite
    thruster: Thruster                       # Thruster for the satellite



    def __init__(self):
        """
        Base constructor for Satellite class: 
        """
        pass


    def __init__(self, id: int, mass: float, altitude: float, longitude: float):
        """
        Constructor for a basic satellite with default properties:

        Inputs: 
        id: Unique identifier for the satellite
        altitude: Altitude of the satellite in meters above Earth's surface
        longitude: Initial longitude of the satellite in degrees
            
        Default properties:
        No controller,
        time = 0 s
        crosslink gimbal range = 0 degrees
        """
        self.id = id
        self.mass = mass
        self.positionECI, self.velocityECI = generate_orbit(altitude, longitude)  
        self.time = 0.0                                 # Initial time
        self.crosslinkGimbalRange = np.radians(0)       # Default gimbal range
        

    #TODO
    def get_control_inputs(self) -> np.ndarray:
        """
        Computes the control inputs for the satellite
        """
        return np.zeros(3)
    
    def step(self, dt: float):
        """
        Advances the state of the satellite by a time step dt
        """
        # Get control inputs
        controlForce = self.get_control_inputs()

        # Propogate vehicle
        propogate_orbit(self, controlForce, dt)

