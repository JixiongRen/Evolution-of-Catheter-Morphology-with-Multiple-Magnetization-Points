# basics_nondim.py
from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

def skew(v: jnp.ndarray) -> jnp.ndarray:
    v = jnp.asarray(v).reshape(3,)
    vx, vy, vz = v[0], v[1], v[2]
    return jnp.array([[0.0, -vz, vy],
                      [vz, 0.0, -vx],
                      [-vy, vx, 0.0]], dtype=v.dtype)

def quat_normalize(q: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    q = jnp.asarray(q).reshape(4,)
    n = jnp.linalg.norm(q)
    return q / jnp.maximum(n, eps)

def quat_to_rotmat(q: jnp.ndarray) -> jnp.ndarray:
    """Quaternion q = [w,x,y,z] -> R_world_from_body."""
    q = quat_normalize(q)
    w, x, y, z = q
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    return jnp.array([
        [ww + xx - yy - zz, 2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),       ww - xx + yy - zz, 2*(yz - wx)],
        [2*(xz - wy),       2*(yz + wx),       ww - xx - yy + zz],
    ], dtype=q.dtype)

def quat_derivative(q: jnp.ndarray, omega_world: jnp.ndarray) -> jnp.ndarray:
    """q_dot = 0.5 * Omega(omega) * q, omega in WORLD frame."""
    q = quat_normalize(q)
    wx, wy, wz = jnp.asarray(omega_world).reshape(3,)
    w, x, y, z = q
    return 0.5 * jnp.array([
        -x*wx - y*wy - z*wz,
         w*wx + y*wz - z*wy,
         w*wy - x*wz + z*wx,
         w*wz + x*wy - y*wx,
    ], dtype=q.dtype)

# 3-stage Gauss-Legendre (order 6) tableau on [0,1]
sqrt15 = jnp.sqrt(15.0)
GL3_C = jnp.array([0.5 - sqrt15/10.0, 0.5, 0.5 + sqrt15/10.0], dtype=jnp.float64)
GL3_B = jnp.array([5.0/18.0, 4.0/9.0, 5.0/18.0], dtype=jnp.float64)
GL3_A = jnp.array([
    [5.0/36.0,               2.0/9.0 - sqrt15/15.0, 5.0/36.0 - sqrt15/30.0],
    [5.0/36.0 + sqrt15/24.0, 2.0/9.0,               5.0/36.0 - sqrt15/24.0],
    [5.0/36.0 + sqrt15/30.0, 2.0/9.0 + sqrt15/15.0, 5.0/36.0],
], dtype=jnp.float64)
