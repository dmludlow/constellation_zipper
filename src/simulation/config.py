import numpy as np

"""
Physical constants used in the simulation.
"""

# --- Physical Constants (Earth) ---
EARTH_GRAVITATIONAL_PARAMETER = 3.986004418e14       # m^3/s^2
EARTH_EQUATORIAL_RADIUS_M = 6378137.0                  # meters
EARTH_J2_COEFFICIENT = 1.08262668e-3                 # Dimensionless

# --- Satelite Properties ---
SATELLITE_MASS_KG = 500                                 # kg
MAX_THRUST_N = 0.2                                     # N

# --- Simulation Properties ---
SIMULATION_TIME_STEP_S = 10.0                            # seconds
SIMULATION_DURATION_S = 3600.0 * 6                     # seconds

INTERSECT_TIME_S = 3600 * 6                       # hours into simulation

# --- Crosslink Properties ---
CROSSLINK_GIMBAL_RANGE_RAD = np.deg2rad(20)          # radians

# --- Controller Properties ---
MPC_TIME_STEP_S = 300.0                             # seconds (5 minutes)
MPC_HORIZON_LENGTH_S = 48                           # Number of steps in the MPC horizon (4 hours lookahead)
SAFETY_DISTANCE_M = 100e3                           # Minimum distance to maintain from other satellites in meters

# Scale factor to convert cost metrics from meters^2 to kilometers^2 (prevents OSQP ill-conditioning)
COST_METER_TO_KM_SCALE = 1e-6

MPC_Q_MATRIX = np.diag([0.0, 0.0, 0.0, 0.0]) * COST_METER_TO_KM_SCALE  # Disable slot-keeping (Option 2)
MPC_R_MATRIX = 1 * COST_METER_TO_KM_SCALE                              # Fuel penalty (scaled to km^2)
MPC_W_MATRIX = 0.7 * COST_METER_TO_KM_SCALE                             # Spacing weighting (scaled to km^2)
MPC_SOFT_CONSTRAINT_PENALTY = 1e3 * COST_METER_TO_KM_SCALE             # Penalty for soft constraints (scaled to km^2)

