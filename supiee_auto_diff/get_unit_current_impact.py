# -*- coding: utf-8 -*-
"""Sample unit-current magnetic field responses on a cubic grid."""

from __future__ import annotations

import pickle
from pathlib import Path
import numpy as np

from mag_manip import mag_manip
from mag_manip.mag_manip import ForwardModelMPEM


def generate_unit_currents(n_coils: int = 8, dtype=np.float64) -> np.ndarray:
    """Generate one unit-current vector per coil."""
    return np.eye(n_coils, dtype=dtype)


def generate_grid(side: float = 0.2, num: int = 10, dtype=np.float64):
    """Generate a cubic sampling grid centered at the origin."""
    half = side / 2.0
    xs = np.linspace(-half, half, num=num, dtype=dtype)
    ys = np.linspace(-half, half, num=num, dtype=dtype)
    zs = np.linspace(-half, half, num=num, dtype=dtype)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_xyz = np.stack([X, Y, Z], axis=-1)
    return grid_xyz, xs, ys, zs


def compute_batch_B(supiee, positions: np.ndarray, currents: np.ndarray) -> np.ndarray:
    """
    positions: (N,3)
    currents:  (8,)
    return:    (N,3)
    """
    try:
        B = supiee.computeFieldFromCurrents(position=positions, currents=currents)
        B = np.asarray(B, dtype=np.float64)
        if B.ndim == 2 and B.shape[1] == 3:
            return B
    except Exception:
        print("not support batch position")
        pass

    return np.stack(
        [
            np.asarray(
                supiee.computeFieldFromCurrents(position=p, currents=currents),
                dtype=np.float64,
            ).reshape(3,)
            for p in positions
        ],
        axis=0,
    )


def main():
    supiee = ForwardModelMPEM()
    calib_path = (Path(__file__).resolve().parent / "../calib/mpem_calibration_file_sp=40_order=1.yaml").resolve()
    supiee.setCalibrationFile(str(calib_path))

    # b_field = supiee.computeFieldFromCurrents(position=position, currents=coils_currents).flatten()

    currents = generate_unit_currents(n_coils=8, dtype=np.float64)  # (8,8)

    grid_xyz, xs, ys, zs = generate_grid(side=0.6, num=30, dtype=np.float64)
    nx, ny, nz, _ = grid_xyz.shape
    C = currents.shape[0]  # 8

    B_tensor = np.empty((nx, ny, nz, C, 3), dtype=np.float64)

    pts_flat = grid_xyz.reshape(-1, 3)  # (N,3)
    for c in range(C):
        coils_currents = currents[c]  # (8,)
        B_flat = compute_batch_B(supiee, pts_flat, coils_currents)  # (N,3)
        B_tensor[:, :, :, c, :] = B_flat.reshape(nx, ny, nz, 3)

    out_dir = (Path(__file__).resolve().parent / "offline_interpolation_data/unit_current_b_data").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (out_dir / "unit_current_impact.pkl").resolve()
    payload = {
        "B": B_tensor,               #
        "x": xs, "y": ys, "z": zs,
        "currents": currents,
        "meta": {
            "units": {"length": "m", "current": "A", "B": "T"},
            "side": 0.6,
            "num_per_edge": 30,
            "shape": list(B_tensor.shape),
            "description": "Unit-current magnetic field impact tensor over a cubic grid centered at origin.",
        }
    }

    with out_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved to: {out_path}")
    print(f"B tensor shape: {B_tensor.shape} (expected [{nx}, {ny}, {nz}, 8, 3])")


if __name__ == "__main__":
    main()

    # out_path = (Path(__file__).resolve().parent / "unit_current_b_data/unit_current_impact.pkl").resolve()
    # with open(out_path, "rb") as f:
    #     payload = pickle.load(f)
    # B = payload["B"]
    #
    # assert B.shape == (10, 10, 10, 8, 3)
