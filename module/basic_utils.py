import numpy as np

# -------- Gauss-Legendre 6 点积分参数 --------
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

# 3×3 a_ij 矩阵
GL3_A = np.array([
    [5/36,  2/9 - np.sqrt(15)/15,  5/36 - np.sqrt(15)/30],
    [5/36 + np.sqrt(15)/24,  2/9,  5/36 - np.sqrt(15)/24],
    [5/36 + np.sqrt(15)/30,  2/9 + np.sqrt(15)/15,  5/36],
])

# -------- 基础工具函数 --------

def skew(v: np.ndarray) -> np.ndarray:
    """
    计算向量 v 的反对称矩阵

    参数
    ----------
    v : numpy.array of shape (3,)
        向量
    """
    vx, vy, vz = v
    return np.array([[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]])

def quat_to_rotmat(Q: np.ndarray) -> np.ndarray:
    """
    计算四元数 Q 对应的旋转矩阵

    参数
    ----------
    Q : numpy.array of shape (4,)
        四元数
    """
    qr = Q[0]
    q = Q[1:].reshape(3)
    I = np.eye(3)
    q_sq = qr**2 - np.dot(q, q)
    return q_sq * I + 2 *qr *skew(q) + 2 * np.outer(q, q)


def quat_derivative(Q: np.ndarray, Omega: np.ndarray) -> np.ndarray:
    """
    计算四元数 Q 对应的旋转矩阵的导数

    参数
    ----------
    Q : numpy.array of shape (4,)
        四元数
    Omega : numpy.array of shape (3,)
        角速度
    """
    qr = Q[0]
    q = Q[1:].reshape(3)
    upper = -q.reshape(1, 3)
    lower = qr * np.eye(3) - skew(q)
    K = np.vstack((upper, lower))
    return 0.5 * K @ Omega
