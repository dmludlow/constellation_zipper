import numpy as np
from src.constellation import Constellation
from src.vehicle.satellite import Satellite

class Simulation:
    """
    Class representing a simulation environment for a satellite vehicle
    """

    dt: float                                  # Time step for the simulation in seconds
    duration: float                            # Total duration of the simulation in seconds
    timeVector: np.ndarray                     # Array of time steps for the simulation
    constellation: Constellation               # Constellation of satellites in the simulation
    #telemetry

    
    #TODO       example below from chat
    def __init__(self, constellation, duration_sec, dt):
        self.constellation = constellation
        self.duration = duration_sec
        self.dt = dt
        self.timeVector = np.arange(0, duration_sec, dt)
        
        # Unified Telemetry Database
        self.telemetry = {}
        for sat in self.constellation.satellites:
            self.telemetry[sat.id] = {
                'time': [],
                'r': [],       # Will hold Nx3 arrays
                'v': [],       # Will hold Nx3 arrays
                'q': [],       # Will hold Nx4 arrays
                'w': [],       # Will hold Nx3 arrays
                'u_wheel': [], # Actuator torque commands
                'f_thresh':[], # Actuator thrust forces
                'nadir_err': [],
                'gimbal_err': []
            }
    

    def step(self):
        """
        Advances simulation by one time step
        """
        for sat in self.constellation.satellites:
            # Step each satellite
            sat.step(self.dt)

            # Record telemetry
            controlForce, controlTorque = sat.get_control_inputs()
            self.telemetry[sat.id]['time'].append(sat.time)
            self.telemetry[sat.id]['r'].append(sat.positionECI.copy())
            self.telemetry[sat.id]['v'].append(sat.velocityECI.copy())
            self.telemetry[sat.id]['q'].append(sat.attitude.copy())
            self.telemetry[sat.id]['w'].append(sat.rollRates.copy())
            self.telemetry[sat.id]['u_wheel'].append(controlTorque.copy())
            self.telemetry[sat.id]['f_thresh'].append(controlForce.copy())
            self.telemetry[sat.id]['nadir_err'].append(0.0)
            self.telemetry[sat.id]['gimbal_err'].append(0.0)

    def run(self):
        """
        Runs the simulation for the specified duration
        """
        for t in self.timeVector:
            self.step()

        # Convert lists to numpy arrays for easier post-processing
        for sat_id in self.telemetry:
            for key in self.telemetry[sat_id]:
                self.telemetry[sat_id][key] = np.array(self.telemetry[sat_id][key])