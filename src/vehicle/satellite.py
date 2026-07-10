from src.actuators.base import Actuator
from src.vehicle.controller import Controller
from src.physics.orbit import propogate_orbit, generate_orbit                                                                                                                          
from src.physics.attitude import propogate_attitude 

import numpy as np

class Satellite:
    """
    Class representing a satellite vehicle
    """

    id: int                                  # Unique identifier for the satellite

    actuators: list[Actuator]                # List of actuators on the satellite
    controller: Controller                   # Controller for the satellite

    # Vehicle state variables
    attitude: np.ndarray                     # Quaternion representing the satellite's orientation
    rollRates: np.ndarray                    # Roll rates of the satellite in rad/s
    positionECI: np.ndarray                  # Position of the satellite in ECI frame in meters
    velocityECI: np.ndarray                  # Velocity of the satellite in ECI frame in m/s
    time: float                              # Current time in seconds
    
    # Spacecraft properties
    mass: float                              # Mass of the satellite in kg
    momentOfInertia: np.ndarray              # 3x3 Moment of inertia matrix of the satellite in kg*m^2
    crosslinkGimbalRange: float              # Maximum gimbal range for front/rear crosslink laser in radians


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
        mass: Mass of the satellite in kg
        altitude: Altitude of the satellite in meters above Earth's surface
        longitude: Initial longitude of the satellite in degrees
            
        Default properties:
        No actuators,
        No controller,
        roll rates = 0 rad/s
        time = 0 s
        moment of inertia = 10 kg*m^2 (diagonal)
        cossine gimbal range = 0 degrees
        """
        self.id = id
        self.mass = mass
        self.attitude = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        self.rollRates = np.zeros(3)                    # No initial roll rates
        self.positionECI, self.velocityECI = generate_orbit(altitude, longitude)  
        self.time = 0.0                                 # Initial time
        self.momentOfInertia = np.diag([10.0, 10.0, 10.0])  # Default moment of inertia
        self.crosslinkGimbalRange = np.radians(0)       # Default gimbal range
        

    #TODO
    def get_control_inputs(self):
        """
        Computes the control inputs for the satellite
        """
        return np.zeros(3), np.zeros(3)  # Placeholder for control force and torque

    def step(self, dt: float):
        """
        Advances the state of the satellite by a time step dt
        """
        # Get control inputs
        controlForce, controlTorque = self.get_control_inputs()

        # Propogate vehicle
        propogate_orbit(self, controlForce, dt)
        propogate_attitude(self, controlTorque, dt)

