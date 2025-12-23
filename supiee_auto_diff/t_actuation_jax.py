# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

from actuation_interpolator_jax import (
    load_actuation_table,
    interpolate_A_checked,
    apply_checked,
    interpolate_A_jax,
    apply_jax,
    apply_jit,
    interpolate_A_jit,
    apply_vmap,
    interpolate_A_vmap,
)


def _numpy_cell_index_and_t(axis: np.ndarray, x: float):
    # Mimic jnp.searchsorted(axis, x, side="right") - 1 and clamp
    i = int(np.searchsorted(axis, x, side="right") - 1)
    i = max(0, min(i, axis.size - 2))
    x0, x1 = axis[i], axis[i + 1]
    t = (x - x0) / (x1 - x0)
    t = float(np.clip(t, 0.0, 1.0))
    return i, t


def _numpy_trilerp_A(A_table: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, P: np.ndarray) -> np.ndarray:
    x, y, z = float(P[0]), float(P[1]), float(P[2])
    ix, tx = _numpy_cell_index_and_t(xs, x)
    iy, ty = _numpy_cell_index_and_t(ys, y)
    iz, tz = _numpy_cell_index_and_t(zs, z)

    wx0, wx1 = (1.0 - tx), tx
    wy0, wy1 = (1.0 - ty), ty
    wz0, wz1 = (1.0 - tz), tz

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


def _rand_in_bounds(rng: np.random.Generator, lo: float, hi: float) -> float:
    # avoid exact boundary to reduce “cell edge” corner cases in random testing
    eps = 1e-12 * max(1.0, abs(hi - lo))
    return float(rng.uniform(lo + eps, hi - eps))


def main():
    root = Path(__file__).resolve().parent
    pkl_path = root / "offline_interpolation_data" / "actuation_tables" / "actuation_table.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing {pkl_path}. Run build_actuation_table.py first.")

    table = load_actuation_table(pkl_path, dtype=jnp.float32)

    # --- 1) Basic shape checks ---
    A_table = np.asarray(table.A_table)  # host copy for test convenience
    xs = np.asarray(table.xs)
    ys = np.asarray(table.ys)
    zs = np.asarray(table.zs)

    assert A_table.ndim == 5 and A_table.shape[-2:] == (8, 8), f"Bad A_table shape: {A_table.shape}"
    assert xs.ndim == ys.ndim == zs.ndim == 1
    nx, ny, nz = xs.size, ys.size, zs.size
    assert A_table.shape[0] == nx and A_table.shape[1] == ny and A_table.shape[2] == nz
    print(f"[OK] Loaded A_table: {A_table.shape}, axes: ({nx},{ny},{nz})")

    # --- 2) Consistency: y == A(P) @ i ---
    rng = np.random.default_rng(0)
    P = np.array([
        _rand_in_bounds(rng, xs[0], xs[-1]),
        _rand_in_bounds(rng, ys[0], ys[-1]),
        _rand_in_bounds(rng, zs[0], zs[-1]),
    ], dtype=np.float64)

    i = rng.normal(size=(8,)).astype(np.float64)

    A_chk = interpolate_A_checked(table, P)       # numpy output
    y_chk = apply_checked(table, P, i)            # numpy output
    y_ref = A_chk @ i

    err = np.max(np.abs(y_chk - y_ref))
    print(f"[OK] y == A@i max abs err: {err:e}")
    assert err < 1e-6, f"Mismatch y vs A@i: {err}"

    # --- 3) JAX kernel vs numpy trilerp (same weights) ---
    A_np_interp = _numpy_trilerp_A(A_table, xs, ys, zs, P)
    A_jax_interp = np.asarray(
        interpolate_A_jit(table.A_table, table.xs, table.ys, table.zs, jnp.asarray(P, table.xs.dtype))
    )
    errA = np.max(np.abs(A_np_interp - A_jax_interp))
    print(f"[OK] A_jax vs A_numpy_trilerp max abs err: {errA:e}")
    assert errA < 1e-5, f"Mismatch A interpolation: {errA}"

    # --- 4) Boundary: out-of-bounds must raise ---
    P_bad = P.copy()
    P_bad[0] = float(xs[-1] + (xs[-1] - xs[0]) * 0.01)
    try:
        _ = apply_checked(table, P_bad, i)
        raise AssertionError("Expected ValueError for out-of-bounds P, but no exception was raised.")
    except ValueError:
        print("[OK] Out-of-bounds check raises ValueError as expected.")

    # --- 5) Gradient tests: ensure interpolation is in JAX diff graph ---
    Pj = jnp.asarray(P, dtype=table.xs.dtype)
    ij = jnp.asarray(i, dtype=table.xs.dtype)

    def objective_P(P_):
        y_ = apply_jax(table.A_table, table.xs, table.ys, table.zs, P_, ij)
        return jnp.sum(y_ * y_)

    def objective_i(i_):
        y_ = apply_jax(table.A_table, table.xs, table.ys, table.zs, Pj, i_)
        return jnp.sum(y_ * y_)

    gP = jax.grad(objective_P)(Pj)
    gi = jax.grad(objective_i)(ij)

    gP_np = np.asarray(gP)
    gi_np = np.asarray(gi)
    assert np.all(np.isfinite(gP_np)), f"Non-finite grad wrt P: {gP_np}"
    assert np.all(np.isfinite(gi_np)), f"Non-finite grad wrt i: {gi_np}"
    print(f"[OK] grad wrt P finite, norm={np.linalg.norm(gP_np):.6e}")
    print(f"[OK] grad wrt i finite, norm={np.linalg.norm(gi_np):.6e}")

    # --- 6) vmap batch test ---
    N = 16
    P_batch = np.stack([
        np.array([
            _rand_in_bounds(rng, xs[0], xs[-1]),
            _rand_in_bounds(rng, ys[0], ys[-1]),
            _rand_in_bounds(rng, zs[0], zs[-1]),
        ], dtype=np.float64)
        for _ in range(N)
    ], axis=0)

    I_batch = rng.normal(size=(N, 8)).astype(np.float64)

    y_b = apply_vmap(
        table.A_table, table.xs, table.ys, table.zs,
        jnp.asarray(P_batch, table.xs.dtype),
        jnp.asarray(I_batch, table.xs.dtype),
    )
    y_b_np = np.asarray(y_b)
    assert y_b_np.shape == (N, 8)
    assert np.all(np.isfinite(y_b_np))
    print("[OK] vmap batch apply returns finite outputs with shape (N,8).")

    print("\nALL TESTS PASSED.")


if __name__ == "__main__":
    main()
