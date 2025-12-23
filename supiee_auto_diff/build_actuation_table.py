# -*- coding: utf-8 -*-
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np


G5_ORDER = [
    "dBx_dx",
    "dBx_dy",
    "dBx_dz",
    "dBy_dy",
    "dBy_dz",
]


def _resolve_paths(
    input_pkl: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Tuple[Path, Path]:
    here = Path(__file__).resolve().parent
    in_path = (
        Path(input_pkl).resolve()
        if input_pkl
        else (here / "offline_interpolation_data/unit_current_b_data/unit_current_impact.pkl").resolve()
    )

    out_dir = Path(output_dir).resolve() if output_dir else (here / "offline_interpolation_data/actuation_tables").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = (out_dir / "actuation_table.pkl").resolve()
    return in_path, out_path


def _validate_payload(payload: Dict[str, Any]) -> None:
    for k in ("B", "x", "y", "z"):
        if k not in payload:
            raise KeyError(f"payload missing key '{k}'")

    B = payload["B"]
    if not (isinstance(B, np.ndarray) and B.ndim == 5 and B.shape[-2:] == (8, 3)):
        raise ValueError(f"payload['B'] must be ndarray with shape (nx,ny,nz,8,3), got {getattr(B,'shape',None)}")

    xs, ys, zs = payload["x"], payload["y"], payload["z"]
    for name, arr in (("x", xs), ("y", ys), ("z", zs)):
        if not isinstance(arr, np.ndarray) or arr.ndim != 1:
            raise ValueError(f"payload['{name}'] must be 1D ndarray, got {type(arr)} with shape {getattr(arr,'shape',None)}")
        if arr.size < 2:
            raise ValueError(f"payload['{name}'] must have at least 2 points")
        if not np.all(np.diff(arr) > 0):
            raise ValueError(f"payload['{name}'] must be strictly increasing")


def build_actuation_table(
    payload: Dict[str, Any],
    *,
    edge_order: int = 2,
    out_dtype: np.dtype = np.float64,
) -> Dict[str, Any]:
    """
    Input:
      payload['B']: (nx, ny, nz, 8, 3)  unit-current B field, T/A
      payload['x','y','z']: axis coordinates, m

    Output payload includes:
      - 'A_table': (nx, ny, nz, 8, 8)
      - 'G5':      (nx, ny, nz, 8, 5)
      - plus original 'B','x','y','z','currents','meta'
    """
    _validate_payload(payload)

    B_tensor = np.asarray(payload["B"], dtype=out_dtype)  # (nx,ny,nz,8,3)
    xs = np.asarray(payload["x"], dtype=out_dtype)
    ys = np.asarray(payload["y"], dtype=out_dtype)
    zs = np.asarray(payload["z"], dtype=out_dtype)

    nx, ny, nz, C, _ = B_tensor.shape
    if C != 8:
        raise ValueError(f"Expected 8 coils, got {C}")

    # dB/dx, dB/dy, dB/dz: (nx,ny,nz,8,3)
    dB_dx = np.empty((nx, ny, nz, C, 3), dtype=out_dtype)
    dB_dy = np.empty((nx, ny, nz, C, 3), dtype=out_dtype)
    dB_dz = np.empty((nx, ny, nz, C, 3), dtype=out_dtype)

    # Compute gradients per coil and component
    for c in range(C):
        for comp in range(3):
            S = B_tensor[:, :, :, c, comp]  # (nx,ny,nz)
            dS_dx, dS_dy, dS_dz = np.gradient(S, xs, ys, zs, edge_order=edge_order)
            dB_dx[:, :, :, c, comp] = dS_dx
            dB_dy[:, :, :, c, comp] = dS_dy
            dB_dz[:, :, :, c, comp] = dS_dz

    # Build G5: (nx,ny,nz,8,5)
    G5 = np.empty((nx, ny, nz, C, 5), dtype=out_dtype)
    # g = [dBx/dx, dBx/dy, dBx/dz, dBy/dy, dBy/dz]
    G5[:, :, :, :, 0] = dB_dx[:, :, :, :, 0]
    G5[:, :, :, :, 1] = dB_dy[:, :, :, :, 0]
    G5[:, :, :, :, 2] = dB_dz[:, :, :, :, 0]
    G5[:, :, :, :, 3] = dB_dy[:, :, :, :, 1]
    G5[:, :, :, :, 4] = dB_dz[:, :, :, :, 1]

    # Build A_table: (nx,ny,nz,8,8), rows are outputs, cols are coils
    A_table = np.empty((nx, ny, nz, 8, 8), dtype=out_dtype)
    # First 3 rows are Bx,By,Bz
    # B_tensor[..., c, :] => column c in the first 3 rows
    # Shape alignment: fill A_table[..., 0:3, c] with B_tensor[..., c, 0:3]
    for c in range(C):
        A_table[:, :, :, 0:3, c] = B_tensor[:, :, :, c, 0:3]
        A_table[:, :, :, 3:8, c] = G5[:, :, :, c, 0:5]

    meta_in = payload.get("meta", {}) if isinstance(payload.get("meta", {}), dict) else {}
    meta_out = {
        **meta_in,
        "actuation_table": {
            "version": "v1",
            "edge_order": int(edge_order),
            "g5_order": list(G5_ORDER),
            "A_rows": ["Bx", "By", "Bz"] + list(G5_ORDER),
            "A_cols": [f"coil_{i}" for i in range(8)],
            "units": {
                "B": "T/A",
                "grad": "T/(A*m)",
                "length": "m",
                "current": "A",
            },
            "shapes": {
                "B_tensor": list(B_tensor.shape),
                "G5_tensor": list(G5.shape),
                "A_table": list(A_table.shape),
            },
        },
    }

    out = dict(payload)
    out["B"] = B_tensor
    out["G5"] = G5
    out["A_table"] = A_table
    out["x"], out["y"], out["z"] = xs, ys, zs
    out["meta"] = meta_out
    return out


def main():
    in_path, out_path = _resolve_paths()
    with in_path.open("rb") as f:
        payload = pickle.load(f)

    out_payload = build_actuation_table(payload, edge_order=2, out_dtype=np.float64)

    with out_path.open("wb") as f:
        pickle.dump(out_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    A = out_payload["A_table"]
    print(f"[OK] wrote: {out_path}")
    print(f"  A_table shape: {A.shape}  (nx,ny,nz,8,8)")
    print(f"  g5 order: {out_payload['meta']['actuation_table']['g5_order']}")


if __name__ == "__main__":
    main()
