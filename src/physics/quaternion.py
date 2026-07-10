import numpy as np

def quaternion_multiply(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Multiplies two quaternions q and p using the Hamilton convention.
    Both q and p are expected as [qw, qx, qy, qz].
    """
    qw, qx, qy, qz = q
    pw, px, py, pz = p
    
    return np.array([
        qw*pw - qx*px - qy*py - qz*pz,
        qw*px + qx*pw + qy*pz - qz*py,
        qw*py - qx*pz + qy*pw + qz*px,
        qw*pz + qx*py - qy*px + qz*pw
    ])

def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Returns the conjugate of a quaternion q = [qw, qx, qy, qz].
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])

def rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotates a 3D vector v using quaternion q = [qw, qx, qy, qz].
    Assumes q represents the rotation from frame A to frame B (e.g. ECI to body),
    yielding v_B = rotate_vector(q, v_A).
    Formula: [0, v_B] = q * [0, v_A] * q*
    """
    q_v = np.array([0.0, v[0], v[1], v[2]])
    q_conj = quaternion_conjugate(q)
    rotated_q = quaternion_multiply(quaternion_multiply(q, q_v), q_conj)
    return rotated_q[1:]

def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """
    Normalizes a quaternion to be a unit quaternion.
    """
    norm = np.linalg.norm(q)
    if norm == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm
