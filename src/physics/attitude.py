from typing import TYPE_CHECKING
import numpy as np
import src.config as config
from src.physics.quaternion import rotate_vector, quaternion_normalize

if TYPE_CHECKING:
    from src.vehicle.satellite import Satellite

def attitude_derivatives(
    q: np.ndarray, 
    omega: np.ndarray, 
    inertia: np.ndarray, 
    pos_eci: np.ndarray, 
    control_torque: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes attitude quaternion and angular velocity derivatives (dq/dt, domega/dt).
    """
    # Normalize quat 
    q_normalized = quaternion_normalize(q)
    qw, qx, qy, qz = q_normalized
    wx, wy, wz = omega

    # Kinematics: dq/dt = 0.5 * q * omega
    dq_dt = 0.5 * np.array([
        -qx * wx - qy * wy - qz * wz,
         qw * wx + qy * wz - qz * wy,
         qw * wy + qz * wx - qx * wz,
         qw * wz + qx * wy - qy * wx
    ])

    # Gravity gradient disturbance torque
    r_norm = np.linalg.norm(pos_eci)
    if r_norm > 0:
        # Rotate Earth-to-satellite vector into body frame
        r_body = rotate_vector(q_normalized, pos_eci)
        # Tau_gg = 3 * mu / r^5 * (r_body x (J * r_body))
        tau_gg = (3.0 * config.EARTH_GRAVITATIONAL_PARAMETER / (r_norm**5)) * np.cross(r_body, inertia @ r_body)
    else:
        tau_gg = np.zeros(3)

    # Total external torque in body frame (excluding reaction wheel internal torque dynamics for now)
    # TODO: Add other disturbances like magnetic torque, drag, etc.
    tau_total = control_torque + tau_gg

    # Dynamics: J * domega/dt = tau_total - omega x (J * omega)
    # Solving for domega/dt using np.linalg.solve for speed/stability
    domega_dt = np.linalg.solve(inertia, tau_total - np.cross(omega, inertia @ omega))

    return dq_dt, domega_dt

def propogate_attitude(satellite: "Satellite", controlTorque: np.ndarray, dt: float):
    """
    Propagates the attitude quaternion and roll rates of the satellite using RK4 integration,
    including gravity-gradient torque perturbations.
    """
    q = satellite.attitude
    omega = satellite.rollRates
    inertia = satellite.momentOfInertia
    pos_eci = satellite.positionECI

    # RK4
    k1_q, k1_omega = attitude_derivatives(q, omega, inertia, pos_eci, controlTorque)
    
    k2_q, k2_omega = attitude_derivatives(
        q + k1_q * dt / 2.0, 
        omega + k1_omega * dt / 2.0, 
        inertia, 
        pos_eci, 
        controlTorque
    )
    
    k3_q, k3_omega = attitude_derivatives(
        q + k2_q * dt / 2.0, 
        omega + k2_omega * dt / 2.0, 
        inertia, 
        pos_eci, 
        controlTorque
    )
    
    k4_q, k4_omega = attitude_derivatives(
        q + k3_q * dt, 
        omega + k3_omega * dt, 
        inertia, 
        pos_eci, 
        controlTorque
    )

    # Update state variables
    new_q = q + (dt / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
    new_omega = omega + (dt / 6.0) * (k1_omega + 2.0 * k2_omega + 2.0 * k3_omega + k4_omega)

    # Save normalized states back to the satellite
    satellite.attitude = quaternion_normalize(new_q)
    satellite.rollRates = new_omega