# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def _metrics(A_ref: np.ndarray, A_hat: np.ndarray, eps: float = 1e-12) -> dict:
    D = A_hat - A_ref
    abs_err = np.abs(D)
    rel_err = abs_err / (np.abs(A_ref) + eps)

    ref_fro = float(np.linalg.norm(A_ref, ord="fro"))
    diff_fro = float(np.linalg.norm(D, ord="fro"))
    return {
        "max_abs": float(abs_err.max()),
        "mean_abs": float(abs_err.mean()),
        "max_rel": float(rel_err.max()),
        "mean_rel": float(rel_err.mean()),
        "fro": diff_fro,
        "fro_rel": float(diff_fro / (ref_fro + eps)),
    }


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have same size")
    xm = x - x.mean()
    ym = y - y.mean()
    denom = float(np.linalg.norm(xm) * np.linalg.norm(ym))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(xm, ym) / denom)


def _summarize_dist(name: str, v: np.ndarray) -> None:
    v = np.asarray(v, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        print(name, ": no finite values")
        return
    qs = np.percentile(v, [0, 50, 90, 95, 99, 100])
    print(name, "min/median/p90/p95/p99/max:", qs.tolist())


def _bin_by_quantiles(x: np.ndarray, y: np.ndarray, qs: list[float]) -> None:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size == 0:
        return
    edges = np.quantile(x, qs)
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i == len(edges) - 2:
            sel = (x >= lo) & (x <= hi)
        else:
            sel = (x >= lo) & (x < hi)
        if not np.any(sel):
            continue
        yy = y[sel]
        print(
            f"cond quantile bin [{qs[i]:.2f},{qs[i+1]:.2f}]",
            "count:",
            int(sel.sum()),
            "y_mean:",
            float(np.mean(yy)),
            "y_max:",
            float(np.max(yy)),
        )


def _solve_inverse(A: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64).reshape(8, 8)
    y = np.asarray(y, dtype=np.float64).reshape(8,)
    lam = float(lam)
    if lam <= 0.0:
        i, *_ = np.linalg.lstsq(A, y, rcond=None)
        return np.asarray(i, dtype=np.float64).reshape(8,)
    M = A.T @ A + lam * np.eye(8, dtype=np.float64)
    b = A.T @ y
    return np.linalg.solve(M, b).reshape(8,)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mag", type=str, default="supiee_auto_diff/mag_manip_jax_interpolator_compare_result/actuation_mag_manip.npz")
    parser.add_argument("--jax", type=str, default="supiee_auto_diff/mag_manip_jax_interpolator_compare_result/ctuation_jax_interpolated.npz")
    parser.add_argument("--out", type=str, default="supiee_auto_diff/mag_manip_jax_interpolator_compare_result/actuation_diff.npz")
    parser.add_argument("--lambda", dest="lam", type=float, default=1e-6)
    parser.add_argument("--inv_seed", type=int, default=0)
    args = parser.parse_args()

    mag_path = Path(args.mag).resolve()
    jax_path = Path(args.jax).resolve()
    if not mag_path.exists():
        raise FileNotFoundError(mag_path)
    if not jax_path.exists():
        raise FileNotFoundError(jax_path)

    mag = np.load(mag_path, allow_pickle=True)
    jax = np.load(jax_path, allow_pickle=True)

    Pm = np.asarray(mag["P"], dtype=np.float64)
    Pj = np.asarray(jax["P"], dtype=np.float64)
    if Pm.shape != Pj.shape or not np.allclose(Pm, Pj, rtol=0.0, atol=0.0):
        raise ValueError("P mismatch between mag and jax files. Ensure jax export uses mag NPZ as input.")

    A_mag = np.asarray(mag["A_mag"], dtype=np.float64)
    A_jax = np.asarray(jax["A_jax"], dtype=np.float64)
    if A_mag.shape != A_jax.shape:
        raise ValueError(f"A shape mismatch: {A_mag.shape} vs {A_jax.shape}")

    N = A_mag.shape[0]
    stats_full = [_metrics(A_mag[k], A_jax[k]) for k in range(N)]
    stats_B = [_metrics(A_mag[k][0:3, :], A_jax[k][0:3, :]) for k in range(N)]
    stats_G = [_metrics(A_mag[k][3:8, :], A_jax[k][3:8, :]) for k in range(N)]

    def _agg(key: str):
        vals = np.array([s[key] for s in stats_full], dtype=np.float64)
        return float(vals.mean()), float(vals.max())

    def _agg_part(stats: list[dict], key: str):
        vals = np.array([s[key] for s in stats], dtype=np.float64)
        return float(vals.mean()), float(vals.max())

    for key in ("max_abs", "mean_abs", "max_rel", "mean_rel", "fro", "fro_rel"):
        m, mx = _agg(key)
        print("full", key, "mean:", m, "max:", mx)

    for key in ("max_abs", "mean_abs", "max_rel", "mean_rel", "fro", "fro_rel"):
        mB, mxB = _agg_part(stats_B, key)
        mG, mxG = _agg_part(stats_G, key)
        print("B_rows", key, "mean:", mB, "max:", mxB)
        print("G_rows", key, "mean:", mG, "max:", mxG)

    cond_mag = np.array([float(np.linalg.cond(A_mag[k])) for k in range(N)], dtype=np.float64)
    cond_jax = np.array([float(np.linalg.cond(A_jax[k])) for k in range(N)], dtype=np.float64)

    _summarize_dist("cond_mag", cond_mag)
    _summarize_dist("cond_jax", cond_jax)

    fro_rel_full = np.array([s["fro_rel"] for s in stats_full], dtype=np.float64)
    fro_rel_B = np.array([s["fro_rel"] for s in stats_B], dtype=np.float64)
    fro_rel_G = np.array([s["fro_rel"] for s in stats_G], dtype=np.float64)

    m_full = np.isfinite(cond_mag) & np.isfinite(fro_rel_full) & (cond_mag > 0)
    if np.any(m_full):
        r = _pearsonr(np.log10(cond_mag[m_full]), fro_rel_full[m_full])
        print("corr(log10(cond_mag), fro_rel_full):", r)

    mB = np.isfinite(cond_mag) & np.isfinite(fro_rel_B) & (cond_mag > 0)
    if np.any(mB):
        r = _pearsonr(np.log10(cond_mag[mB]), fro_rel_B[mB])
        print("corr(log10(cond_mag), fro_rel_B_rows):", r)

    mG = np.isfinite(cond_mag) & np.isfinite(fro_rel_G) & (cond_mag > 0)
    if np.any(mG):
        r = _pearsonr(np.log10(cond_mag[mG]), fro_rel_G[mG])
        print("corr(log10(cond_mag), fro_rel_G_rows):", r)

    print("bins using cond_mag vs fro_rel_full")
    _bin_by_quantiles(np.log10(cond_mag), fro_rel_full, [0.0, 0.5, 0.9, 0.99, 1.0])

    rng = np.random.default_rng(int(args.inv_seed))
    i_true = rng.normal(size=(N, 8)).astype(np.float64)
    y = np.einsum("nij,nj->ni", A_mag, i_true)

    i_hat_mag = np.empty((N, 8), dtype=np.float64)
    i_hat_jax = np.empty((N, 8), dtype=np.float64)
    for k in range(N):
        i_hat_mag[k] = _solve_inverse(A_mag[k], y[k], args.lam)
        i_hat_jax[k] = _solve_inverse(A_jax[k], y[k], args.lam)

    def _rel_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        da = a - b
        num = np.linalg.norm(da, axis=-1)
        den = np.linalg.norm(b, axis=-1)
        return num / (den + eps)

    i_err_mag_vs_true = _rel_l2(i_hat_mag, i_true)
    i_err_jax_vs_true = _rel_l2(i_hat_jax, i_true)
    i_err_jax_vs_mag = _rel_l2(i_hat_jax, i_hat_mag)

    y_hat_mag = np.einsum("nij,nj->ni", A_mag, i_hat_mag)
    y_hat_jax_on_mag = np.einsum("nij,nj->ni", A_mag, i_hat_jax)

    y_res_mag = _rel_l2(y_hat_mag, y)
    y_res_jax_on_mag = _rel_l2(y_hat_jax_on_mag, y)

    def _summ(name: str, v: np.ndarray) -> None:
        v = np.asarray(v, dtype=np.float64).ravel()
        v = v[np.isfinite(v)]
        if v.size == 0:
            print(name, ": no finite values")
            return
        qs = np.percentile(v, [0, 50, 90, 95, 99, 100])
        print(name, "min/median/p90/p95/p99/max:", qs.tolist())

    print("inverse settings: lambda=", float(args.lam), "inv_seed=", int(args.inv_seed))
    _summ("i_err_mag_vs_true", i_err_mag_vs_true)
    _summ("i_err_jax_vs_true", i_err_jax_vs_true)
    _summ("i_err_jax_vs_mag", i_err_jax_vs_mag)
    _summ("y_res_mag", y_res_mag)
    _summ("y_res_jax_on_mag", y_res_jax_on_mag)

    m_inv = np.isfinite(cond_mag) & (cond_mag > 0) & np.isfinite(i_err_jax_vs_mag)
    if np.any(m_inv):
        r = _pearsonr(np.log10(cond_mag[m_inv]), i_err_jax_vs_mag[m_inv])
        print("corr(log10(cond_mag), i_err_jax_vs_mag):", r)
        print("bins using cond_mag vs i_err_jax_vs_mag")
        _bin_by_quantiles(np.log10(cond_mag), i_err_jax_vs_mag, [0.0, 0.5, 0.9, 0.99, 1.0])

    diff = A_jax - A_mag

    out_path = Path(args.out).resolve()
    np.savez(
        out_path,
        P=Pm,
        A_mag=A_mag,
        A_jax=A_jax,
        diff=diff,
        cond_mag=cond_mag,
        cond_jax=cond_jax,
        fro_rel_full=fro_rel_full,
        fro_rel_B=fro_rel_B,
        fro_rel_G=fro_rel_G,
        lam=float(args.lam),
        inv_seed=int(args.inv_seed),
        i_true=i_true,
        y=y,
        i_hat_mag=i_hat_mag,
        i_hat_jax=i_hat_jax,
        i_err_mag_vs_true=i_err_mag_vs_true,
        i_err_jax_vs_true=i_err_jax_vs_true,
        i_err_jax_vs_mag=i_err_jax_vs_mag,
        y_res_mag=y_res_mag,
        y_res_jax_on_mag=y_res_jax_on_mag,
        mag_file=str(mag_path),
        jax_file=str(jax_path),
    )
    print("saved:", out_path)


if __name__ == "__main__":
    main()
