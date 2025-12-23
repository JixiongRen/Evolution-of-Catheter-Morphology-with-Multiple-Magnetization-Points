"""
此脚本用于验证 actuation_interpolator 的准确性
给定一组电流向量
1. 在 x in [-0.2, 0.2]m, y in [-0.2, 0.2]m, z in [-0.2, 0.2]m 的范围内
    1.1 使用mag_manip和我们的插值函数分别计算驱动矩阵
    1.2 分析驱动矩阵的误差
"""

from mag_manip import mag_manip
from mag_manip.mag_manip import ForwardModelMPEM
from pathlib import Path
import sys
import argparse
import numpy as np
import jax.numpy as jnp


def _rand_in_bounds(rng: np.random.Generator, lo: float, hi: float) -> float:
    eps = 1e-12 * max(1.0, abs(hi - lo))
    return float(rng.uniform(lo + eps, hi - eps))


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="actuation_interpolator_accuracy_results.npz")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    repo_root = here.parents[2]
    sys.path.insert(0, str(repo_root))

    from supiee_auto_diff.actuation_interpolator_jax import load_actuation_table, interpolate_A_vmap

    supiee_mag_manip = ForwardModelMPEM()
    calib_path = (Path(__file__).resolve().parents[2] / "calib/mpem_calibration_file_sp=40_order=1.yaml").resolve()
    supiee_mag_manip.setCalibrationFile(str(calib_path))

    pkl_path = (repo_root / "supiee_auto_diff" / "offline_interpolation_data" / "actuation_tables" / "actuation_table.pkl").resolve()
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing {pkl_path}. Run supiee_auto_diff/build_actuation_table.py first.")
    table = load_actuation_table(pkl_path)

    xs = np.asarray(table.xs)
    ys = np.asarray(table.ys)
    zs = np.asarray(table.zs)

    rng = np.random.default_rng(args.seed)

    P0 = np.array([
        _rand_in_bounds(rng, xs[0], xs[-1]),
        _rand_in_bounds(rng, ys[0], ys[-1]),
        _rand_in_bounds(rng, zs[0], zs[-1]),
    ], dtype=np.float64)

    A_mag0 = np.asarray(supiee_mag_manip.getActuationMatrix(position=P0), dtype=np.float64)
    A_jax0 = np.asarray(
        interpolate_A_vmap(table.A_table, table.xs, table.ys, table.zs, jnp.asarray([P0], dtype=table.xs.dtype))
    )[0].astype(np.float64)

    print("single point check")
    print("P0:", P0)
    print("A_mag0 shape:", A_mag0.shape, "A_jax0 shape:", A_jax0.shape)

    I_unit = np.eye(8, dtype=np.float64)
    B_cols_ok = True
    G_cols_ok = True
    for c in range(8):
        e = I_unit[c]
        B = np.asarray(supiee_mag_manip.computeFieldFromCurrents(position=P0, currents=e), dtype=np.float64).reshape(3,)
        if np.max(np.abs(B - A_mag0[0:3, c])) > 1e-9:
            B_cols_ok = False

        G5 = np.asarray(supiee_mag_manip.computeGradient5FromCurrents(position=P0, currents=e), dtype=np.float64).reshape(5,)
        if np.max(np.abs(G5 - A_mag0[3:8, c])) > 1e-9:
            G_cols_ok = False

    print("A_mag0: first 3 rows equal unit-current B columns:", B_cols_ok)
    print("A_mag0: last 5 rows equal unit-current Gradient5 columns:", G_cols_ok)

    N = int(args.N)
    P_batch = np.stack([
        np.array([
            _rand_in_bounds(rng, xs[0], xs[-1]),
            _rand_in_bounds(rng, ys[0], ys[-1]),
            _rand_in_bounds(rng, zs[0], zs[-1]),
        ], dtype=np.float64)
        for _ in range(N)
    ], axis=0)

    A_jax = np.asarray(
        interpolate_A_vmap(table.A_table, table.xs, table.ys, table.zs, jnp.asarray(P_batch, dtype=table.xs.dtype))
    ).astype(np.float64)

    A_mag = np.empty((N, 8, 8), dtype=np.float64)
    for k in range(N):
        A_mag[k] = np.asarray(supiee_mag_manip.getActuationMatrix(position=P_batch[k]), dtype=np.float64)

    stats = []
    for k in range(N):
        stats.append(_metrics(A_mag[k], A_jax[k]))

    def _agg(key: str):
        vals = np.array([s[key] for s in stats], dtype=np.float64)
        return float(vals.mean()), float(vals.max())

    for key in ("max_abs", "mean_abs", "max_rel", "mean_rel", "fro", "fro_rel"):
        m, mx = _agg(key)
        print(key, "mean:", m, "max:", mx)

    out_path = (here / args.out).resolve()
    np.savez(
        out_path,
        P=P_batch,
        A_mag=A_mag,
        A_jax=A_jax,
        diff=A_jax - A_mag,
    )

    print("saved:", out_path)
