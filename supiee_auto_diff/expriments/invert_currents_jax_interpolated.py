# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import sys

import numpy as np
import jax
import jax.numpy as jnp


def _parse_vec(tokens, *, n: int, name: str) -> np.ndarray:
    if isinstance(tokens, (list, tuple)):
        if len(tokens) == 1:
            s = str(tokens[0])
            parts = [p.strip() for p in s.split(",") if p.strip()]
            if len(parts) != n:
                raise ValueError(f"--{name} must be either {n} numbers or 1 CSV string with {n} numbers")
            return np.asarray([float(p) for p in parts], dtype=np.float64)
        if len(tokens) == n:
            return np.asarray([float(p) for p in tokens], dtype=np.float64)
        raise ValueError(f"--{name} must be either {n} numbers or 1 CSV string with {n} numbers")

    s = str(tokens)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != n:
        raise ValueError(f"--{name} must be either {n} numbers or 1 CSV string with {n} numbers")
    return np.asarray([float(p) for p in parts], dtype=np.float64)


def _check_in_bounds(axis: np.ndarray, val: float, name: str) -> None:
    lo = float(axis[0])
    hi = float(axis[-1])
    if not (lo <= val <= hi):
        raise ValueError(f"{name}={val} out of bounds [{lo}, {hi}] (no extrapolation allowed)")


def _stats(x: np.ndarray) -> str:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return "[]"
    qs = np.quantile(x, [0.0, 0.5, 0.9, 0.95, 0.99, 1.0])
    return str([float(v) for v in qs])


def _clip_box(i: jnp.ndarray, imax: float) -> jnp.ndarray:
    return jnp.clip(i, -imax, imax)


def _svd_smax(A: jnp.ndarray) -> jnp.ndarray:
    s = jnp.linalg.svd(A, compute_uv=False)
    return s[0]


def _svd_smax_smin_cond(A: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    s = jnp.linalg.svd(A, compute_uv=False)
    smax = s[0]
    smin = s[-1]
    cond = smax / (smin + 1e-12)
    return smax, smin, cond


def _ridge_one(A: jnp.ndarray, y: jnp.ndarray, lam: float, w: float) -> jnp.ndarray:
    At = A.T
    Q = w * (At @ A) + (lam + 1e-12) * jnp.eye(8, dtype=A.dtype)
    b = w * (At @ y)
    return jnp.linalg.solve(Q, b)


@jax.jit
def _ridge_batch(A: jnp.ndarray, y: jnp.ndarray, lam: float, w: float) -> jnp.ndarray:
    return jax.vmap(lambda A1, y1: _ridge_one(A1, y1, lam, w))(A, y)


def _pgd_one(A: jnp.ndarray, y: jnp.ndarray, lam: float, w: float, imax: float, iters: int, i0: jnp.ndarray) -> jnp.ndarray:
    At = A.T
    smax = _svd_smax(A)
    L = w * (smax * smax) + lam + 1e-12
    alpha = 1.0 / L

    def body(_, i):
        grad = w * (At @ (A @ i - y)) + lam * i
        i2 = i - alpha * grad
        return _clip_box(i2, imax)

    i0 = jnp.asarray(i0, dtype=A.dtype).reshape(8,)
    i0 = _clip_box(i0, imax)
    return jax.lax.fori_loop(0, iters, body, i0)


@partial(jax.jit, static_argnames=("iters",))
def _pgd_batch(A: jnp.ndarray, y: jnp.ndarray, lam: float, w: float, imax: float, iters: int) -> jnp.ndarray:
    z0 = jnp.zeros((8,), dtype=A.dtype)
    return jax.vmap(lambda A1, y1: _pgd_one(A1, y1, lam, w, imax, iters, z0))(A, y)


@partial(jax.jit, static_argnames=("iters",))
def _pgd_batch_init(A: jnp.ndarray, y: jnp.ndarray, i0: jnp.ndarray, lam: float, w: float, imax: float, iters: int) -> jnp.ndarray:
    return jax.vmap(lambda A1, y1, i1: _pgd_one(A1, y1, lam, w, imax, iters, i1))(A, y, i0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actuation_table_pkl",
        type=str,
        default=str(
            (
                Path(__file__).resolve().parents[1]
                / "offline_interpolation_data/actuation_tables/actuation_table.pkl"
            ).resolve()
        ),
    )

    parser.add_argument("--in_npz", type=str, default="")
    parser.add_argument("--P", nargs="+", default=[])
    parser.add_argument("--y8", nargs="+", default=[])

    parser.add_argument("--out", type=str, default="supiee_auto_diff/invert_currents_result/invert_currents_jax_interpolated_out.npz")
    parser.add_argument("--batch", type=int, default=4096)

    parser.add_argument("--imax", type=float, default=50.0)
    parser.add_argument("--lam", type=float, default=1e-6)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--method", type=str, default="ridge_pgd", choices=["ridge", "pgd", "ridge_pgd"])
    parser.add_argument("--check_bounds", action="store_true")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--enable_x64", action="store_true")

    args = parser.parse_args()

    if args.enable_x64:
        jax.config.update("jax_enable_x64", True)

    here = Path(__file__).resolve().parent
    repo_root = here.parents[2]
    supiee_dir = here.parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(supiee_dir))

    from supiee_auto_diff.actuation_interpolator_jax import load_actuation_table, interpolate_A_vmap

    pkl_path = Path(args.actuation_table_pkl).resolve()
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing {pkl_path}. Build it with supiee_auto_diff/build_actuation_table.py")

    dtype = jnp.float32 if args.dtype == "float32" else jnp.float64
    table = load_actuation_table(pkl_path, dtype=dtype)

    if args.in_npz:
        in_path = Path(args.in_npz).resolve()
        if not in_path.exists():
            raise FileNotFoundError(f"Missing {in_path}")

        payload = np.load(in_path, allow_pickle=True)
        if "P" not in payload:
            raise KeyError("input npz must contain key 'P' with shape (N,3)")

        P = np.asarray(payload["P"], dtype=np.float64).reshape(-1, 3)

        if "y8" in payload:
            y8 = np.asarray(payload["y8"], dtype=np.float64).reshape(-1, 8)
        elif "B" in payload and "G5" in payload:
            B = np.asarray(payload["B"], dtype=np.float64).reshape(-1, 3)
            G5 = np.asarray(payload["G5"], dtype=np.float64).reshape(-1, 5)
            y8 = np.concatenate([B, G5], axis=1)
        else:
            raise KeyError("input npz must contain 'y8' or both 'B' and 'G5'")

        if P.shape[0] != y8.shape[0]:
            raise ValueError(f"P and y8 batch size mismatch: P={P.shape}, y8={y8.shape}")

        in_npz_str = str(in_path)
    else:
        if not args.P or not args.y8:
            raise ValueError("Provide either --in_npz, or both --P and --y8")
        P = _parse_vec(args.P, n=3, name="P").reshape(1, 3)
        y8 = _parse_vec(args.y8, n=8, name="y8").reshape(1, 8)
        in_npz_str = ""

    if args.check_bounds:
        xs = np.asarray(table.xs)
        ys = np.asarray(table.ys)
        zs = np.asarray(table.zs)
        for k in range(P.shape[0]):
            _check_in_bounds(xs, float(P[k, 0]), "x")
            _check_in_bounds(ys, float(P[k, 1]), "y")
            _check_in_bounds(zs, float(P[k, 2]), "z")

    N = int(P.shape[0])
    bs = int(args.batch)

    i_hat = np.empty((N, 8), dtype=np.float64)
    y_pred = np.empty((N, 8), dtype=np.float64)
    res_norm = np.empty((N,), dtype=np.float64)
    res_B_norm = np.empty((N,), dtype=np.float64)
    res_G_norm = np.empty((N,), dtype=np.float64)
    fit_term = np.empty((N,), dtype=np.float64)
    reg_term = np.empty((N,), dtype=np.float64)
    obj_term = np.empty((N,), dtype=np.float64)
    i_norm = np.empty((N,), dtype=np.float64)
    smax_arr = np.empty((N,), dtype=np.float64)
    smin_arr = np.empty((N,), dtype=np.float64)
    cond_arr = np.empty((N,), dtype=np.float64)

    imax = float(args.imax)
    lam = float(args.lam)
    w = float(args.w)
    iters = int(args.iters)
    method = str(args.method)

    for s in range(0, N, bs):
        e = min(N, s + bs)

        A_blk = interpolate_A_vmap(
            table.A_table,
            table.xs,
            table.ys,
            table.zs,
            jnp.asarray(P[s:e], dtype=table.xs.dtype),
        )
        y_blk = jnp.asarray(y8[s:e], dtype=table.xs.dtype)

        sv = jax.vmap(_svd_smax_smin_cond)(A_blk)
        smax_blk, smin_blk, cond_blk = sv[0], sv[1], sv[2]
        smax_arr[s:e] = np.asarray(smax_blk, dtype=np.float64)
        smin_arr[s:e] = np.asarray(smin_blk, dtype=np.float64)
        cond_arr[s:e] = np.asarray(cond_blk, dtype=np.float64)

        if method == "pgd":
            i_blk = _pgd_batch(A_blk, y_blk, lam=lam, w=w, imax=imax, iters=iters)
        else:
            i_ridge = _ridge_batch(A_blk, y_blk, lam=lam, w=w)
            i_ridge = jnp.clip(i_ridge, -imax, imax)
            if method == "ridge":
                i_blk = i_ridge
            else:
                i_blk = _pgd_batch_init(A_blk, y_blk, i_ridge, lam=lam, w=w, imax=imax, iters=iters)

        y_hat_blk = jnp.einsum("nij,nj->ni", A_blk, i_blk)

        i_hat[s:e] = np.asarray(i_blk, dtype=np.float64)
        y_pred[s:e] = np.asarray(y_hat_blk, dtype=np.float64)

        r = np.asarray(y_hat_blk - y_blk, dtype=np.float64)
        res_norm[s:e] = np.linalg.norm(r, axis=1)
        res_B_norm[s:e] = np.linalg.norm(r[:, 0:3], axis=1)
        res_G_norm[s:e] = np.linalg.norm(r[:, 3:8], axis=1)
        i_norm[s:e] = np.linalg.norm(i_hat[s:e], axis=1)

        fit_term[s:e] = np.sum(r * r, axis=1)
        reg_term[s:e] = lam * np.sum(i_hat[s:e] * i_hat[s:e], axis=1)
        obj_term[s:e] = w * fit_term[s:e] + reg_term[s:e]

    sat_frac = float(np.mean(np.any(np.abs(i_hat) >= (imax - 1e-9), axis=1)))

    out_path = Path(args.out).resolve()
    np.savez(
        out_path,
        P=P,
        y8=y8,
        i_hat=i_hat,
        y_pred=y_pred,
        res_norm=res_norm,
        res_B_norm=res_B_norm,
        res_G_norm=res_G_norm,
        fit_term=fit_term,
        reg_term=reg_term,
        obj_term=obj_term,
        i_norm=i_norm,
        sat_frac=sat_frac,
        smax=smax_arr,
        smin=smin_arr,
        cond=cond_arr,
        lam=lam,
        w=w,
        imax=imax,
        iters=iters,
        method=method,
        dtype=str(args.dtype),
        in_npz=in_npz_str,
        actuation_table_pkl=str(pkl_path),
    )

    print("saved:", out_path)
    print("N:", N)
    print("res_norm min/median/p90/p95/p99/max:", _stats(res_norm))
    print("res_B   min/median/p90/p95/p99/max:", _stats(res_B_norm))
    print("res_G   min/median/p90/p95/p99/max:", _stats(res_G_norm))
    print("fit_term(||Ai-y||^2) min/median/p90/p95/p99/max:", _stats(fit_term))
    print("reg_term(lam||i||^2) min/median/p90/p95/p99/max:", _stats(reg_term))
    print("obj_term(w*fit+reg)  min/median/p90/p95/p99/max:", _stats(obj_term))
    print("i_norm   min/median/p90/p95/p99/max:", _stats(i_norm))
    print("sat_frac (any |i_k| close to imax):", sat_frac)
    print("cond(A)  min/median/p90/p95/p99/max:", _stats(cond_arr))

    if N == 1:
        print("P:", P[0])
        print("y8:", y8[0])
        print("i_hat:", i_hat[0])
        print("y_pred:", y_pred[0])
        print("y_err:", (y_pred[0] - y8[0]))
        print("smax/smin/cond:", float(smax_arr[0]), float(smin_arr[0]), float(cond_arr[0]))
        print("fit_term, reg_term, obj_term:", float(fit_term[0]), float(reg_term[0]), float(obj_term[0]))


if __name__ == "__main__":
    main()
