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
    telemetry: dict                            # Dictionary to store telemetry data for each satellite

    
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
                'r': [],                # Will hold Nx3 arrays
                'v': [],                # Will hold Nx3 arrays
                'applied_thrust': [],   # Will hold Nx3 arrays
                'leading_link_active': [],
                'trailing_link_active': [],
                'solver_status': [],    # Records MPC status at each step
            }
    

    def log_telemetry(self):
        """
        Records the current physical state, connection status, and control inputs
        for all satellites in the constellation.
        """
        for sat in self.constellation.satellites:
            controlForce = sat.get_control_inputs()
            status = sat.controller.prob.status if (hasattr(sat.controller, 'prob') and sat.controller.prob is not None) else "None"
            self.telemetry[sat.id]['time'].append(sat.time)
            self.telemetry[sat.id]['r'].append(sat.positionECI.copy())
            self.telemetry[sat.id]['v'].append(sat.velocityECI.copy())
            self.telemetry[sat.id]['applied_thrust'].append(controlForce.copy())
            self.telemetry[sat.id]['leading_link_active'].append(sat.leadingConnection)
            self.telemetry[sat.id]['trailing_link_active'].append(sat.trailingConnection)
            self.telemetry[sat.id]['solver_status'].append(status)

    def step(self):
        """
        Advances simulation by one time step
        """
        self.constellation.communicate()

        for sat in self.constellation.satellites:
            # Step each satellite (advances physics and time)
            sat.step(self.dt)

        # Update link connections 
        self.constellation.check_crosslinks()
        
        # Record new states in the telemetry database
        self.log_telemetry()

    def run(self):
        """
        Runs the simulation for the specified duration
        """
        for t in self.timeVector:
            # Print simulation progress every hour
            if t > 0 and t % 100.0 == 0:
                print(f" * sim time: {int(t)} s")
            self.step()

        # Convert lists to numpy arrays for easier post-processing
        for sat_id in self.telemetry:
            for key in self.telemetry[sat_id]:
                self.telemetry[sat_id][key] = np.array(self.telemetry[sat_id][key])