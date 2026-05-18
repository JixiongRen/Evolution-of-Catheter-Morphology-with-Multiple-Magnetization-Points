"""Basic quaternion and linear-algebra helpers."""

import numpy as np



GL3_C = np.array([
    0.5 - np.sqrt(15)/10,
    0.5,
    0.5 + np.sqrt(15)/10,
    ])

GL3_B = np.array([
    5/18,
    8/18,
    5/18,
    ])

GL3_A = np.array([
    [5/36,  2/9 - np.sqrt(15)/15,  5/36 - np.sqrt(15)/30],
    [5/36 + np.sqrt(15)/24,  2/9,  5/36 - np.sqrt(15)/24],
    [5/36 + np.sqrt(15)/30,  2/9 + np.sqrt(15)/15,  5/36],
])




def skew(v: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric matrix of a 3D vector."""
    vx, vy, vz = v
    return np.array([[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]])


def quat_to_rotmat(Q: np.ndarray) -> np.ndarray:
    """Convert a quaternion in ``wxyz`` order to a rotation matrix."""
    qr = Q[0]
    q = Q[1:].reshape(3)
    I = np.eye(3)
    q_sq = qr**2 - np.dot(q, q)
    return q_sq * I + 2 *qr *skew(q) + 2 * np.outer(q, q)


def quat_derivative(Q: np.ndarray, Omega: np.ndarray) -> np.ndarray:
    """Compute the quaternion derivative for a given angular velocity."""
    qr = Q[0]
    q = Q[1:].reshape(3)
    upper = -q.reshape(1, 3)
    lower = qr * np.eye(3) - skew(q)
    K = np.vstack((upper, lower))
    return 0.5 * K @ Omega
