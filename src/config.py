"""
This module contains physical constants used in the simulation.
"""

# --- Physical Constants (Earth) ---
EARTH_GRAVITATIONAL_PARAMETER = 3.986004418e14       # m^3/s^2
EARTH_EQUATORIAL_RADIUS = 6378137.0                  # meters
EARTH_J2_COEFFICIENT = 1.08262668e-3                 # Dimensionless

# --- Satelite Properties ---
SATELLITE_MASS = 800                                 # kg
MAX_THRUST = 10 # 0.17                                    # N

# --- Simulation Properties ---
SIMULATION_TIME_STEP = 10.0                            # seconds
SIMULATION_DURATION = 3600.0 * 6                     # seconds
