# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import numpy as np

from mag_manip.mag_manip import ForwardModelMPEM


def _rand_in_bounds(rng: np.random.Generator, lo: float, hi: float) -> float:
    eps = 1e-12 * max(1.0, abs(hi - lo))
    return float(rng.uniform(lo + eps, hi - eps))


def _load_axes_from_actuation_table(pkl_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with pkl_path.open("rb") as f:
        payload = pickle.load(f)
    xs = np.asarray(payload["x"], dtype=np.float64)
    ys = np.asarray(payload["y"], dtype=np.float64)
    zs = np.asarray(payload["z"], dtype=np.float64)
    return xs, ys, zs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--calib",
        type=str,
        default=str(
            (Path(__file__).resolve().parents[2] / "calib/mpem_calibration_file_sp=40_order=1.yaml").resolve()
        ),
    )
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
    parser.add_argument("--out", type=str, default="actuation_mag_manip.npz")
    args = parser.parse_args()

    pkl_path = Path(args.actuation_table_pkl).resolve()
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing {pkl_path}. Build it with supiee_auto_diff/build_actuation_table.py")

    xs, ys, zs = _load_axes_from_actuation_table(pkl_path)

    rng = np.random.default_rng(args.seed)
    P = np.stack(
        [
            np.array(
                [
                    _rand_in_bounds(rng, xs[0], xs[-1]),
                    _rand_in_bounds(rng, ys[0], ys[-1]),
                    _rand_in_bounds(rng, zs[0], zs[-1]),
                ],
                dtype=np.float64,
            )
            for _ in range(int(args.N))
        ],
        axis=0,
    )

    model = ForwardModelMPEM()
    model.setCalibrationFile(str(Path(args.calib).resolve()))

    A_mag = np.empty((P.shape[0], 8, 8), dtype=np.float64)
    for k in range(P.shape[0]):
        A_mag[k] = np.asarray(model.getActuationMatrix(position=P[k]), dtype=np.float64)

    out_path = Path(args.out).resolve()
    np.savez(
        out_path,
        P=P,
        A_mag=A_mag,
        seed=int(args.seed),
        calib=str(Path(args.calib).resolve()),
        actuation_table_pkl=str(pkl_path),
    )
    print("saved:", out_path)
    print("P:", P.shape, "A_mag:", A_mag.shape)


if __name__ == "__main__":
    main()
