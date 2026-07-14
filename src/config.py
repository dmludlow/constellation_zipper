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

# --- Crosslink Properties ---
CROSSLINK_GIMBAL_RANGE = 0.17453292519943295        # radians (10 degrees)

# --- Controller Properties ---
MPC_TIME_STEP = 1.0                               # seconds
MPC_HORIZON_LENGTH = 10                           # Number of steps in the MPC horizon
SAFTEY_DISTANCE = 50e3                            # Minimum distance to maintain from other satellites in meters
