"""
This class defines an electric propulsion thurster for a satellite.
"""

class Thruster:
    
    max_thrust: float  # Maximum thrust in Newtons

    def __init__(self, max_thrust: float):
        self.max_thrust = max_thrust