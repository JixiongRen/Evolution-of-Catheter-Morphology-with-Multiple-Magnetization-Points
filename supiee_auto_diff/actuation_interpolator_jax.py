# -*- coding: utf-8 -*-
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Any, Dict, Optional

import numpy as np
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ActuationTable:
    """
    JAX-friendly container.
    A_table: (nx,ny,nz,8,8)
    xs,ys,zs: 1D axes
    """
    A_table: jnp.ndarray
    xs: jnp.ndarray
    ys: jnp.ndarray
    zs: jnp.ndarray
    meta: Dict[str, Any]


def load_actuation_table(pkl_path: str | Path, *, dtype=jnp.float32) -> ActuationTable:
    p = Path(pkl_path).resolve()
    with p.open("rb") as f:
        payload = pickle.load(f)

    if "A_table" not in payload:
        raise KeyError("Missing 'A_table' in payload. Run build_actuation_table.py first.")

    A_np = np.asarray(payload["A_table"])
    xs_np = np.asarray(payload["x"])
    ys_np = np.asarray(payload["y"])
    zs_np = np.asarray(payload["z"])

    # Device arrays
    A = jnp.asarray(A_np, dtype=dtype)
    xs = jnp.asarray(xs_np, dtype=dtype)
    ys = jnp.asarray(ys_np, dtype=dtype)
    zs = jnp.asarray(zs_np, dtype=dtype)

    meta = payload.get("meta", {})
    return ActuationTable(A_table=A, xs=xs, ys=ys, zs=zs, meta=meta)


def _cell_index_and_t(axis: jnp.ndarray, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    axis: (n,)
    x: scalar
    returns:
      i: int index in [0, n-2]
      t in [0,1] such that x = (1-t)*axis[i] + t*axis[i+1]
    """
    n = axis.shape[0]
    # side='right' so that x==axis[-1] falls into last cell with t=1
    i = jnp.searchsorted(axis, x, side="right") - 1
    i = jnp.clip(i, 0, n - 2)

    x0 = axis[i]
    x1 = axis[i + 1]
    t = (x - x0) / (x1 - x0)
    # numerical safety
    t = jnp.clip(t, 0.0, 1.0)
    return i, t


def interpolate_A_jax(A_table: jnp.ndarray, xs: jnp.ndarray, ys: jnp.ndarray, zs: jnp.ndarray, P: jnp.ndarray) -> jnp.ndarray:
    """
    Pure JAX trilinear interpolation.
    Inputs:
      A_table: (nx,ny,nz,8,8)
      P: (3,) -> [x,y,z]
    Returns:
      A(P): (8,8)
    """
    x, y, z = P[0], P[1], P[2]
    ix, tx = _cell_index_and_t(xs, x)
    iy, ty = _cell_index_and_t(ys, y)
    iz, tz = _cell_index_and_t(zs, z)

    wx0, wx1 = (1.0 - tx), tx
    wy0, wy1 = (1.0 - ty), ty
    wz0, wz1 = (1.0 - tz), tz

    # 8 corners, each is (8,8)
    A000 = A_table[ix,     iy,     iz    ]
    A100 = A_table[ix + 1, iy,     iz    ]
    A010 = A_table[ix,     iy + 1, iz    ]
    A110 = A_table[ix + 1, iy + 1, iz    ]
    A001 = A_table[ix,     iy,     iz + 1]
    A101 = A_table[ix + 1, iy,     iz + 1]
    A011 = A_table[ix,     iy + 1, iz + 1]
    A111 = A_table[ix + 1, iy + 1, iz + 1]

    w000 = wx0 * wy0 * wz0
    w100 = wx1 * wy0 * wz0
    w010 = wx0 * wy1 * wz0
    w110 = wx1 * wy1 * wz0
    w001 = wx0 * wy0 * wz1
    w101 = wx1 * wy0 * wz1
    w011 = wx0 * wy1 * wz1
    w111 = wx1 * wy1 * wz1

    A = (
        w000 * A000 + w100 * A100 + w010 * A010 + w110 * A110 +
        w001 * A001 + w101 * A101 + w011 * A011 + w111 * A111
    )
    return A


def apply_jax(A_table: jnp.ndarray, xs: jnp.ndarray, ys: jnp.ndarray, zs: jnp.ndarray, P: jnp.ndarray, i: jnp.ndarray) -> jnp.ndarray:
    """
    Pure JAX: y = A(P) @ i
    P: (3,)
    i: (8,)
    y: (8,)
    """
    A = interpolate_A_jax(A_table, xs, ys, zs, P)  # (8,8), inside JAX graph
    return A @ i


# JIT-able kernels (still pure JAX)
interpolate_A_jit = jax.jit(interpolate_A_jax)
apply_jit = jax.jit(apply_jax)


def _check_in_bounds(axis: np.ndarray, val: float, name: str) -> None:
    lo = float(axis[0])
    hi = float(axis[-1])
    if not (lo <= val <= hi):
        raise ValueError(f"{name}={val} out of bounds [{lo}, {hi}] (no extrapolation allowed)")


def apply_checked(table: ActuationTable, P: np.ndarray, i: np.ndarray) -> np.ndarray:
    """
    Python-level boundary check + calls JAX kernel.
    P: (3,) numpy
    i: (8,) numpy
    Returns numpy y: (8,)
    """
    P = np.asarray(P, dtype=np.float64).reshape(3,)
    i = np.asarray(i, dtype=np.float64).reshape(8,)

    # boundary check in python (raise)
    _check_in_bounds(np.asarray(table.xs), float(P[0]), "x")
    _check_in_bounds(np.asarray(table.ys), float(P[1]), "y")
    _check_in_bounds(np.asarray(table.zs), float(P[2]), "z")

    y = apply_jit(table.A_table, table.xs, table.ys, table.zs, jnp.asarray(P, table.xs.dtype), jnp.asarray(i, table.xs.dtype))
    return np.asarray(y)


def interpolate_A_checked(table: ActuationTable, P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64).reshape(3,)
    _check_in_bounds(np.asarray(table.xs), float(P[0]), "x")
    _check_in_bounds(np.asarray(table.ys), float(P[1]), "y")
    _check_in_bounds(np.asarray(table.zs), float(P[2]), "z")
    A = interpolate_A_jit(table.A_table, table.xs, table.ys, table.zs, jnp.asarray(P, table.xs.dtype))
    return np.asarray(A)


# Batch helpers (JAX)
apply_vmap = jax.vmap(apply_jax, in_axes=(None, None, None, None, 0, 0))  # batch over P and i
interpolate_A_vmap = jax.vmap(interpolate_A_jax, in_axes=(None, None, None, None, 0))  # batch over P


def demo_grad_wrt_P(table: ActuationTable, P: jnp.ndarray, i: jnp.ndarray) -> jnp.ndarray:
    """
    Example: gradient of a scalar objective wrt P.
    This function is pure JAX and demonstrates the interpolation is in the diff graph.
    """
    def objective(P_):
        y = apply_jax(table.A_table, table.xs, table.ys, table.zs, P_, i)
        return jnp.sum(y * y)  # simple scalar
    return jax.grad(objective)(P)
