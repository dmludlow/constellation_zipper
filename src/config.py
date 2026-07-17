import numpy as np

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
CROSSLINK_GIMBAL_RANGE = np.deg2rad(20)          # radians

# --- Controller Properties ---
MPC_TIME_STEP = 300.0                             # seconds (5 minutes)
MPC_HORIZON_LENGTH = 48                           # Number of steps in the MPC horizon (4 hours lookahead)
SAFETY_DISTANCE = 50e3                            # Minimum distance to maintain from other satellites in meters
MPC_Q_MATRIX = np.diag([1.0, 1.0, 10.0, 10.0])    # State errors
MPC_R_MATRIX = 1                                 # Fuel penalty
MPC_W_MATRIX = 0.7                               # Spacing wieghting for equal spacing goal
