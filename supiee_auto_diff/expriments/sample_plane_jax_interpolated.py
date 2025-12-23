# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import jax.numpy as jnp


def _parse_currents(tokens) -> np.ndarray:
    if isinstance(tokens, (list, tuple)):
        if len(tokens) == 1:
            s = str(tokens[0])
            parts = [p.strip() for p in s.split(",") if p.strip()]
            if len(parts) != 8:
                raise ValueError("--currents must be either 8 numbers or 1 CSV string with 8 numbers")
            return np.asarray([float(p) for p in parts], dtype=np.float64)
        if len(tokens) == 8:
            return np.asarray([float(p) for p in tokens], dtype=np.float64)
        raise ValueError("--currents must be either 8 numbers or 1 CSV string with 8 numbers")

    s = str(tokens)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 8:
        raise ValueError("--currents must be either 8 numbers or 1 CSV string with 8 numbers")
    return np.asarray([float(p) for p in parts], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_mag_npz", type=str, default="plane_mag_manip.npz")
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
    parser.add_argument("--currents", nargs="+", default=[])
    parser.add_argument("--out", type=str, default="plane_jax_interpolated.npz")
    parser.add_argument("--batch", type=int, default=4096)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    repo_root = here.parents[2]
    supiee_dir = here.parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(supiee_dir))

    from supiee_auto_diff.actuation_interpolator_jax import load_actuation_table, interpolate_A_vmap

    in_path = Path(args.in_mag_npz).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Missing {in_path}. Run sample_plane_mag_manip.py first.")

    mag = np.load(in_path, allow_pickle=True)
    ys = np.asarray(mag["ys"], dtype=np.float64)
    zs = np.asarray(mag["zs"], dtype=np.float64)
    P = np.asarray(mag["P"], dtype=np.float64)

    if args.currents:
        currents = _parse_currents(args.currents)
    else:
        currents = np.asarray(mag["currents"], dtype=np.float64).reshape(8,)

    pkl_path = Path(args.actuation_table_pkl).resolve()
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing {pkl_path}. Build it with supiee_auto_diff/build_actuation_table.py")

    table = load_actuation_table(pkl_path)

    N = P.shape[0]
    bs = int(args.batch)
    y8 = np.empty((N, 8), dtype=np.float64)

    i_j = jnp.asarray(currents, dtype=table.xs.dtype)

    for s in range(0, N, bs):
        e = min(N, s + bs)
        A_blk = interpolate_A_vmap(
            table.A_table,
            table.xs,
            table.ys,
            table.zs,
            jnp.asarray(P[s:e], dtype=table.xs.dtype),
        )
        y_blk = jnp.einsum("nij,j->ni", A_blk, i_j)
        y8[s:e] = np.asarray(y_blk, dtype=np.float64)

    B = y8[:, 0:3]
    G5 = y8[:, 3:8]
    B_norm = np.linalg.norm(B, axis=1)

    out_path = Path(args.out).resolve()
    np.savez(
        out_path,
        plane=str(mag["plane"]),
        x0=float(mag["x0"]),
        y0=float(mag["y0"]),
        ys=ys,
        zs=zs,
        P=P,
        currents=currents,
        B=B,
        G5=G5,
        y8=y8,
        B_norm=B_norm,
        in_mag_npz=str(in_path),
        actuation_table_pkl=str(pkl_path),
    )
    print("saved:", out_path)
    print("P:", P.shape, "y8:", y8.shape)


if __name__ == "__main__":
    main()
