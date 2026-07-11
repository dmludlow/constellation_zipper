import numpy as np                                                                                                                                                    
from typing import TYPE_CHECKING                                                                                                                                      
                                                                                                                                                                          
if TYPE_CHECKING:                                                                                                                                                     
    from src.vehicle.satellite import Satellite 

class Thruster:
    """
    This class defines an electric propulsion thurster for a satellite.
    """
    
    max_thrust: float  # Maximum thrust in Newtons

    def __init__(self, max_thrust: float):
        self.max_thrust = max_thrust

    # ---- Used for testing before controller is implemented ----
    def thrust_prograde(self, satellite: "Satellite") -> np.ndarray:
        velocity_unit_vector = satellite.velocityECI / np.linalg.norm(satellite.velocityECI)
        return self.max_thrust * velocity_unit_vector