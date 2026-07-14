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
    j2_factor = (1.5 * config.EARTH_J2_COEFFICIENT * config.EARTH_GRAVITATIONAL_PARAMETER * config.EARTH_EQUATORIAL_RADIUS**2) / (r_norm**5)
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


def step_sat_orbit(satellite: "Satellite", commandedForce: np.ndarray, dt: float):
    """
    Propagates the physical orbit of a Satellite object.
    """
    pos_next, vel_next = propogate_orbit_rk4(satellite.positionECI, satellite.velocityECI, satellite.mass, commandedForce, dt)
    satellite.positionECI = pos_next
    satellite.velocityECI = vel_next
    satellite.time += dt