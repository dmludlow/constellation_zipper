from typing import TYPE_CHECKING
import numpy as np
import src.simulation.config as config

# what is going on here, there usually is a better way to do this.
# Why not just import the class directly?
# Oh I see circular import things.  Usually this means you are passing too much info around.
if TYPE_CHECKING:
    from src.vehicle.satellite import Satellite

def generate_orbit(altitude_m: float, degrees_long: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates initial ECI position and velocity for a satellite in a circular equatorial orbit at a given altitude_m and initial longitude.
    """
    # Convert degrees to radians
    theta_rad = np.radians(degrees_long)

    # Calculate radius from Earth's center
    r = config.EARTH_EQUATORIAL_RADIUS_M + altitude_m

    # Position in ECI frame (x, y, z)
    pos_eci = np.array([
        r * np.cos(theta_rad),  # x
        r * np.sin(theta_rad),  # y
        0.0                 # z (equatorial orbit)
    ])

    # Circular orbit velocity magnitude
    v_mag = np.sqrt(config.EARTH_GRAVITATIONAL_PARAMETER / r)

    # Velocity in ECI frame (perpendicular to position vector for circular orbit)
    vel_eci = np.array([
        -v_mag * np.sin(theta_rad),  # vx
         v_mag * np.cos(theta_rad),  # vy
         0.0                     # vz (equatorial orbit)
    ])

    return pos_eci, vel_eci



def orbit_derivatives(pos: np.ndarray, vel: np.ndarray, mass: float, commandedForce: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes position and velocity derivatives (dr/dt, dv/dt) under 2-body and J2 gravity.
    """
    r_norm = np.linalg.norm(pos)
    if r_norm == 0:
        return np.zeros(3), np.zeros(3)

    # Two-body gravity acceleration
    accel_gravity = - (config.EARTH_GRAVITATIONAL_PARAMETER * pos) / (r_norm**3)

    # J2 
    z = pos[2]
    j2_factor = (1.5 * config.EARTH_J2_COEFFICIENT * config.EARTH_GRAVITATIONAL_PARAMETER * config.EARTH_EQUATORIAL_RADIUS_M**2) / (r_norm**5)
    accel_j2 = np.array([
        j2_factor * pos[0] * (5.0 * (z**2) / (r_norm**2) - 1.0),
        j2_factor * pos[1] * (5.0 * (z**2) / (r_norm**2) - 1.0),
        j2_factor * z * (5.0 * (z**2) / (r_norm**2) - 3.0)
    ])

    # 3. Control acceleration from thrusters (stubbed for now)
    accel_control = commandedForce / mass

    accel_total = accel_gravity + accel_j2 + accel_control
    return vel, accel_total


def propogate_orbit_rk4(pos: np.ndarray, vel: np.ndarray, mass: float, commandedForce: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs a single RK4 step to propagate a position and velocity vector under J2 and control thrust.
    Returns the updated (position, velocity) tuple.
    """
    k1_pos, k1_vel = orbit_derivatives(pos, vel, mass, commandedForce)
    k2_pos, k2_vel = orbit_derivatives(pos + k1_pos * dt / 2.0, vel + k1_vel * dt / 2.0, mass, commandedForce)
    k3_pos, k3_vel = orbit_derivatives(pos + k2_pos * dt / 2.0, vel + k2_vel * dt / 2.0, mass, commandedForce)
    k4_pos, k4_vel = orbit_derivatives(pos + k3_pos * dt, vel + k3_vel * dt, mass, commandedForce)

    pos_next = pos + (dt / 6.0) * (k1_pos + 2.0 * k2_pos + 2.0 * k3_pos + k4_pos)
    vel_next = vel + (dt / 6.0) * (k1_vel + 2.0 * k2_vel + 2.0 * k3_vel + k4_vel)
    return pos_next, vel_next

# I would change this to "step_orbit" and and pass in the satellites position, vel, and mass separately.
# This would make it more general and not require the satellite class to be imported here, can avoid circular imports.
# It would also make it easier to test.
def step_sat_orbit(satellite: "Satellite", commandedForce: np.ndarray, dt: float):
    """
    Propagates the physical orbit of a Satellite object.
    """
    pos_next, vel_next = propogate_orbit_rk4(satellite.positionECI, satellite.velocityECI, satellite.mass, commandedForce, dt)
    satellite.positionECI = pos_next
    satellite.velocityECI = vel_next
    satellite.time += dt

# ----- CW Equations: Linearized relative motion for controller ------

from scipy.linalg import expm

def linearize_orbit(pos_ref: np.ndarray, vel_ref: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Linearizes the relative motion equations around a reference orbit.
    Returns the discrete Ad and Bd matrices for the linearized system.
    """
    A_cont, B_cont = eci_to_cw_matrix(pos_ref, vel_ref)
    return discretize_CW(A_cont, B_cont, dt)

def eci_to_cw_matrix(pos_ref: np.ndarray, vel_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts ECI reference position and velocity into the continuous CW matrices.
    x = [x_radial, y_along_track, x_dot, y_dot]^T
    """
    # Compute osculating mean motion n
    r_mag = np.linalg.norm(pos_ref)
    n = np.sqrt(config.EARTH_GRAVITATIONAL_PARAMETER / (r_mag**3))
    
    # A continous
    # x_dot = vx, y_dot = vy
    # vx_dot = 3*n^2*x + 2*n*vy
    # vy_dot = -2*n*vx
    A = np.array([
        [0.0,      0.0, 1.0,      0.0],
        [0.0,      0.0, 0.0,      1.0],
        [3.0*n**2, 0.0, 0.0,      2.0*n],
        [0.0,      0.0, -2.0*n,   0.0]
    ])
    
    # Continous B matrix
    # Along-track force converts to acceleration in the y-direction (index 3)
    B = np.array([
        [0.0],
        [0.0],
        [0.0],
        [1.0 / config.SATELLITE_MASS_KG]
    ])
    
    return A, B

def discretize_CW(A: np.ndarray, B: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Discretizes using block matrix exponential (ZOH).
    """
    n_states = A.shape[0]
    n_inputs = B.shape[1] if B.ndim > 1 else 1
    
    B_col = B.reshape(-1, 1) if B.ndim == 1 else B
    
    # Construct block matrix M = [[A, B], [0, 0]]
    M = np.zeros((n_states + n_inputs, n_states + n_inputs))
    M[:n_states, :n_states] = A
    M[:n_states, n_states:] = B_col
    
    # exp(M * dt)
    ExpM = expm(M * dt)
    
    Ad = ExpM[:n_states, :n_states]
    Bd = ExpM[:n_states, n_states:]
    
    if B.ndim == 1:
        Bd = Bd.flatten()
        
    return Ad, Bd

def eci_to_lvlh(pos_ref: np.ndarray, vel_ref: np.ndarray, pos_query: np.ndarray, vel_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforms position and velocity vectors from the ECI frame to the local LVLH frame
    """
    # LVLH unit vectors
    r_unit = pos_ref / np.linalg.norm(pos_ref)
    h_vec = np.cross(pos_ref, vel_ref)
    h_unit = h_vec / np.linalg.norm(h_vec)
    theta_rad_unit = np.cross(h_unit, r_unit)  # Along-track direction
    
    # Rotation matrix from ECI to LVLH
    R_eci_to_lvlh = np.vstack([r_unit, theta_rad_unit, h_unit])
    
    # Relative Position
    dp_eci = pos_query - pos_ref
    pos_lvlh = R_eci_to_lvlh @ dp_eci
    
    # Relative Velocity (accounting for rotating coordinate frame)
    r_mag = np.linalg.norm(pos_ref)
    omega = h_unit * np.sqrt(config.EARTH_GRAVITATIONAL_PARAMETER / (r_mag**3)) # angular velocity vector
    
    dv_eci = vel_query - vel_ref
    vel_lvlh = R_eci_to_lvlh @ (dv_eci - np.cross(omega, dp_eci))
    
    return pos_lvlh, vel_lvlh

def lvlh_to_eci_force(pos_ref: np.ndarray, vel_ref: np.ndarray, force_lvlh: np.ndarray) -> np.ndarray:
    """
    Rotates a force vector from the LVLH frame back to the absolute ECI frame.
    """
    r_unit = pos_ref / np.linalg.norm(pos_ref)
    h_vec = np.cross(pos_ref, vel_ref)
    h_unit = h_vec / np.linalg.norm(h_vec)
    theta_rad_unit = np.cross(h_unit, r_unit)
    
    # R_eci_to_lvlh transposed is the inverse rotation (LVLH to ECI)
    R_lvlh_to_eci = np.vstack([r_unit, theta_rad_unit, h_unit]).T
    return R_lvlh_to_eci @ force_lvlh

