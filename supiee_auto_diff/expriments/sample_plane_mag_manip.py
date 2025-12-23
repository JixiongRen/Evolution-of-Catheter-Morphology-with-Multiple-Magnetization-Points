# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from mag_manip.mag_manip import ForwardModelMPEM


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


def _grid_plane_x(
    *, x0: float, y_lim: Tuple[float, float], z_lim: Tuple[float, float], ny: int, nz: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys = np.linspace(float(y_lim[0]), float(y_lim[1]), int(ny), dtype=np.float64)
    zs = np.linspace(float(z_lim[0]), float(z_lim[1]), int(nz), dtype=np.float64)
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    X = np.full_like(Y, float(x0))
    P = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    return ys, zs, P


def _compute_batch_field(model: ForwardModelMPEM, P: np.ndarray, currents: np.ndarray) -> np.ndarray:
    try:
        B = model.computeFieldFromCurrents(position=P, currents=currents)
        B = np.asarray(B, dtype=np.float64)
        if B.ndim == 2 and B.shape[1] == 3:
            return B
    except Exception:
        pass

    out = np.empty((P.shape[0], 3), dtype=np.float64)
    for k in range(P.shape[0]):
        out[k] = np.asarray(model.computeFieldFromCurrents(position=P[k], currents=currents), dtype=np.float64).reshape(3,)
    return out


def _compute_batch_grad5(model: ForwardModelMPEM, P: np.ndarray, currents: np.ndarray) -> np.ndarray:
    try:
        G = model.computeGradient5FromCurrents(position=P, currents=currents)
        G = np.asarray(G, dtype=np.float64)
        if G.ndim == 2 and G.shape[1] == 5:
            return G
    except Exception:
        pass

    out = np.empty((P.shape[0], 5), dtype=np.float64)
    for k in range(P.shape[0]):
        out[k] = np.asarray(model.computeGradient5FromCurrents(position=P[k], currents=currents), dtype=np.float64).reshape(5,)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plane", type=str, default="x", choices=["x"])
    parser.add_argument("--x0", type=float, default=0.0)
    parser.add_argument("--y0", type=float, default=0.0)
    parser.add_argument("--ymin", type=float, default=-0.3)
    parser.add_argument("--ymax", type=float, default=0.3)
    parser.add_argument("--zmin", type=float, default=-0.3)
    parser.add_argument("--zmax", type=float, default=0.3)
    parser.add_argument("--ny", type=int, default=100)
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--currents", nargs="+", default=["1", "0", "0", "0", "0", "0", "0", "0"])
    parser.add_argument(
        "--calib",
        type=str,
        default=str((Path(__file__).resolve().parents[2] / "calib/mpem_calibration_file_sp=40_order=1.yaml").resolve()),
    )
    parser.add_argument("--out", type=str, default="plane_mag_manip.npz")
    args = parser.parse_args()

    currents = _parse_currents(args.currents)

    if args.plane != "x":
        raise ValueError("only --plane x is implemented")

    ys, zs, P = _grid_plane_x(
        x0=float(args.x0),
        y_lim=(float(args.ymin), float(args.ymax)),
        z_lim=(float(args.zmin), float(args.zmax)),
        ny=int(args.ny),
        nz=int(args.nz),
    )

    model = ForwardModelMPEM()
    model.setCalibrationFile(str(Path(args.calib).resolve()))

    B = _compute_batch_field(model, P, currents)
    G5 = _compute_batch_grad5(model, P, currents)

    y8 = np.concatenate([B, G5], axis=1)
    B_norm = np.linalg.norm(B, axis=1)

    out_path = Path(args.out).resolve()
    np.savez(
        out_path,
        plane=str(args.plane),
        x0=float(args.x0),
        y0=float(args.y0),
        ys=ys,
        zs=zs,
        P=P,
        currents=currents,
        B=B,
        G5=G5,
        y8=y8,
        B_norm=B_norm,
        calib=str(Path(args.calib).resolve()),
    )
    print("saved:", out_path)
    print("P:", P.shape, "B:", B.shape, "G5:", G5.shape)


if __name__ == "__main__":
    main()
