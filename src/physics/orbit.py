from typing import TYPE_CHECKING
import numpy as np
import src.config as config

if TYPE_CHECKING:
    from src.vehicle.satellite import Satellite

def generate_orbit(altitude: float, degrees_long: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates initial ECI position and velocity for a satellite in a circular equatorial orbit at a given altitude and initial longitude.
    """
    # Convert degrees to radians
    theta = np.radians(degrees_long)

    # Calculate radius from Earth's center
    r = config.EARTH_EQUATORIAL_RADIUS + altitude

    # Position in ECI frame (x, y, z)
    pos_eci = np.array([
        r * np.cos(theta),  # x
        r * np.sin(theta),  # y
        0.0                 # z (equatorial orbit)
    ])

    # Circular orbit velocity magnitude
    v_mag = np.sqrt(config.EARTH_GRAVITATIONAL_PARAMETER / r)

    # Velocity in ECI frame (perpendicular to position vector for circular orbit)
    vel_eci = np.array([
        -v_mag * np.sin(theta),  # vx
         v_mag * np.cos(theta),  # vy
         0.0                     # vz (equatorial orbit)
    ])

    return pos_eci, vel_eci


# TODO verify this works
def get_b_field(position_eci: np.ndarray) -> np.ndarray:
    """
    Returns Earth's magnetic field vector B (in Teslas) at a given position
    using a simplified dipole model.
    """
    r_norm = np.linalg.norm(position_eci)
    if r_norm == 0:
        return np.zeros(3)

    dipole_axis = np.array([0.0, 0.0, 1.0])  # Simplified dipole axis
    m_vector = dipole_axis * config.EARTH_MAGNETIC_DIPOLE_MOMENT

    mu0_over_4pi = 1e-7
    dot_product = np.dot(m_vector, position_eci)

    B = mu0_over_4pi * (3.0 * dot_product * position_eci / (r_norm**5) - m_vector / (r_norm**3))
    return B


def orbit_derivatives(pos: np.ndarray, vel: np.ndarray, mass: float) -> tuple[np.ndarray, np.ndarray]:
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
    j2_factor = (1.5 * config.EARTH_J2_COEFFICIENT * config.EARTH_GRAVITATIONAL_PARAMETER * config.EARTH_EQUATORIAL_RADIUS**2) / (r_norm**5)
    accel_j2 = np.array([
        j2_factor * pos[0] * (5.0 * (z**2) / (r_norm**2) - 1.0),
        j2_factor * pos[1] * (5.0 * (z**2) / (r_norm**2) - 1.0),
        j2_factor * z * (5.0 * (z**2) / (r_norm**2) - 3.0)
    ])

    # 3. Control acceleration from thrusters (stubbed for now)
    # TODO: Add thruster capability here
    # When active, you would rotate force from body to ECI frame and compute:
    # accel_control = force_eci / mass
    # would need to add mass decrease too 
    accel_control = np.zeros(3)

    accel_total = accel_gravity + accel_j2 + accel_control
    return vel, accel_total


def propogate_orbit(satellite: "Satellite", force: np.ndarray, dt: float):
    """
    Propagates the orbit of the satellite using its current state and inputs,
    including J2 orbital perturbations.
    """
    r = satellite.positionECI
    v = satellite.velocityECI
    m = satellite.mass

    # RK4 Integration
    k1_pos, k1_vel = orbit_derivatives(r, v, m)
    k2_pos, k2_vel = orbit_derivatives(r + k1_pos * dt / 2.0, v + k1_vel * dt / 2.0, m)
    k3_pos, k3_vel = orbit_derivatives(r + k2_pos * dt / 2.0, v + k2_vel * dt / 2.0, m)
    k4_pos, k4_vel = orbit_derivatives(r + k3_pos * dt, v + k3_vel * dt, m)

    # Update satellite state variables
    satellite.positionECI = r + (dt / 6.0) * (k1_pos + 2.0 * k2_pos + 2.0 * k3_pos + k4_pos)
    satellite.velocityECI = v + (dt / 6.0) * (k1_vel + 2.0 * k2_vel + 2.0 * k3_vel + k4_vel)
    satellite.time += dt