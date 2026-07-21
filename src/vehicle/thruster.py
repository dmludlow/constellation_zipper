import numpy as np                                                                                                                                                    
from typing import TYPE_CHECKING                                                                                                                                      
                                                                                                                                                                          
if TYPE_CHECKING:                                                                                                                                                     
    from src.vehicle.satellite import Satellite 

class Thruster:
    """
    This class defines an electric propulsion thurster for a satellite.
    """
    
    MAX_THRUST_N: float  # Maximum thrust in Newtons

    def __init__(self, MAX_THRUST_N: float):
        self.MAX_THRUST_N = MAX_THRUST_N

    # ---- Used for testing before controller is implemented ----
    def thrust_prograde(self, satellite: "Satellite") -> np.ndarray:
        velocity_unit_vector = satellite.velocityECI / np.linalg.norm(satellite.velocityECI)
        return self.MAX_THRUST_N * velocity_unit_vector