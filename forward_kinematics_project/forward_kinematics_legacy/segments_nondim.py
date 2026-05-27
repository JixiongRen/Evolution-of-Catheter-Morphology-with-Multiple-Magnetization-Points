# segments_nondim.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable

import jax
import jax.numpy as jnp

from basics_nondim import skew, quat_to_rotmat, quat_derivative, quat_normalize, GL3_A, GL3_B, GL3_C

jax.config.update("jax_enable_x64", True)
Array = jnp.ndarray

@dataclass(frozen=True)
class FlexibleParams:
    length: float
    Kse_inv: Array
    Kbt_inv: Array
    v_star: Array
    u_star: Array

@dataclass(frozen=True)
class RigidParams:
    length: float
    v_star: Array

def cosserat_rhs_dim(
    x_dim: Array,
    sigma_dim: float,
    *,
    Kse_inv: Array,
    Kbt_inv: Array,
    v_star: Array,
    u_star: Array,
    fext_density: Optional[Callable[[Array, float], Array]] = None,
    tauext_density: Optional[Callable[[Array, float], Array]] = None,
) -> Array:
    Q = quat_normalize(x_dim[3:7])
    f = x_dim[7:10]
    tau = x_dim[10:13]
    R = quat_to_rotmat(Q)

    f_bar = jnp.zeros((3,), dtype=x_dim.dtype) if fext_density is None else fext_density(x_dim, sigma_dim)
    tau_bar = jnp.zeros((3,), dtype=x_dim.dtype) if tauext_density is None else tauext_density(x_dim, sigma_dim)

    v = Kse_inv @ (R.T @ f) + v_star
    u = Kbt_inv @ (R.T @ tau) + u_star

    dp = R @ v
    omega_world = R @ u
    dQ = quat_derivative(Q, omega_world)

    df = -f_bar
    dtau = skew(f) @ dp - tau_bar

    dx = jnp.zeros((13,), dtype=x_dim.dtype)
    dx = dx.at[0:3].set(dp)
    dx = dx.at[3:7].set(dQ)
    dx = dx.at[7:10].set(df)
    dx = dx.at[10:13].set(dtau)
    return dx

def interval_residual_gl6_bar(
    x_n_bar: Array,
    k_n_bar: Array,
    x_np1_bar: Array,
    sbar_n: float,
    hbar: float,
    *,
    cs_len: int,
    C_S_fun: Callable[[Array, Array], Array],
    C_BV_fun: Callable[[Array, Array], Array],
    rhs_bar_fun: Callable[[Array, float], Array],
) -> Array:
    C_S = C_S_fun(x_n_bar, x_np1_bar)[:cs_len]
    res_state = x_np1_bar - x_n_bar - hbar * (GL3_B[0]*k_n_bar[0] + GL3_B[1]*k_n_bar[1] + GL3_B[2]*k_n_bar[2])

    res_ks = []
    for i in range(3):
        x_stage = x_n_bar + hbar * (GL3_A[i,0]*k_n_bar[0] + GL3_A[i,1]*k_n_bar[1] + GL3_A[i,2]*k_n_bar[2])
        sbar_stage = sbar_n + GL3_C[i]*hbar
        g_i = rhs_bar_fun(x_stage, sbar_stage)
        res_ks.append(k_n_bar[i] - g_i)
    res_ks = jnp.concatenate(res_ks, axis=0)

    C_BV = C_BV_fun(x_n_bar, x_np1_bar)
    return jnp.concatenate([C_S, res_state, res_ks, C_BV], axis=0)

def rigid_state_along_dim(
    x_prox_dim: Array,
    sigma: float,
    *,
    rigid: RigidParams,
    f_ext_total_dim: Array,
    tau_ext_total_dim: Array,
) -> Array:
    L = float(rigid.length)
    sigma = jnp.clip(jnp.asarray(sigma, dtype=x_prox_dim.dtype), 0.0, L)

    pp = x_prox_dim[0:3]
    Qp = quat_normalize(x_prox_dim[3:7])
    fp = x_prox_dim[7:10]
    tau_p = x_prox_dim[10:13]
    R_p = quat_to_rotmat(Qp)

    p_sigma = pp + sigma * (R_p @ rigid.v_star)
    Q_sigma = Qp

    f_sigma = fp - (sigma / L) * f_ext_total_dim

    r = p_sigma - pp
    term1 = tau_p - (sigma / L) * tau_ext_total_dim
    term2 = -skew(r) @ fp
    term3 = (sigma / (2.0 * L)) * (skew(r) @ f_ext_total_dim)
    tau_sigma = term1 + term2 + term3

    x = jnp.zeros((13,), dtype=x_prox_dim.dtype)
    x = x.at[0:3].set(p_sigma)
    x = x.at[3:7].set(Q_sigma)
    x = x.at[7:10].set(f_sigma)
    x = x.at[10:13].set(tau_sigma)
    return x
