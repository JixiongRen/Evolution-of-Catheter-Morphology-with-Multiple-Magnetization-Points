
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from forward_api import ForwardModel


@dataclass
class SampleResult:
    I: np.ndarray          # (8,)
    p: np.ndarray          # (3,)
    q: np.ndarray          # (4,)
    success: bool
    n_iter: int
    final_E_raw: float
    final_E_scaled: float


def latin_hypercube(n: int, d: int, *, seed: int = 0) -> np.ndarray:
    """Minimal Latin Hypercube Sampling in [0,1]^d."""
    rng = np.random.default_rng(seed)
    u = np.empty((n, d), dtype=float)
    for j in range(d):
        perm = rng.permutation(n)
        u[:, j] = (perm + rng.random(n)) / n
    return u


def voxelize(
    points: np.ndarray,
    *,
    voxel: float,
    p_min: Optional[np.ndarray] = None,
    p_max: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxelize 3D points into an occupancy grid."""
    pts = np.asarray(points, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        return np.zeros((1, 1, 1), dtype=bool), np.zeros((3,), dtype=float), np.array([1, 1, 1], dtype=int)

    if p_min is None:
        p_min = pts.min(axis=0)
    if p_max is None:
        p_max = pts.max(axis=0)

    p_min = np.asarray(p_min, dtype=float)
    p_max = np.asarray(p_max, dtype=float)

    pad = 0.5 * float(voxel)
    origin = p_min - pad
    maxi = p_max + pad

    shape = np.ceil((maxi - origin) / float(voxel)).astype(int)
    shape = np.maximum(shape, 1)

    idx = np.floor((pts - origin) / float(voxel)).astype(int)
    idx = np.clip(idx, 0, shape - 1)

    occ = np.zeros(tuple(shape.tolist()), dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occ, origin, shape


class WorkspaceAnalyzer:
    """
    Build a first workspace range estimate via forward sampling.

    Outputs:
      - samples.npz: arrays for I, p, q, success, diagnostics
      - workspace_voxels.npz: occupancy grid + metadata (optional)
    """

    def __init__(self, forward: ForwardModel):
        self.fwd = forward

    def sample_forward(
        self,
        *,
        n_samples: int,
        I_max: float,
        method: str = "lhs",
        seed: int = 0,
        save_npz: Optional[str] = "samples.npz",
        voxel: Optional[float] = None,
        save_voxels_npz: Optional[str] = "workspace_voxels.npz",
    ) -> Dict[str, Any]:
        d = 8
        m = method.lower()
        if m == "lhs":
            u = latin_hypercube(int(n_samples), d, seed=int(seed))
        elif m in ("uniform", "rand", "random"):
            rng = np.random.default_rng(int(seed))
            u = rng.random((int(n_samples), d))
        else:
            raise ValueError(f"Unknown method: {method}. Use 'lhs' or 'uniform'.")

        I_samples = (2.0 * u - 1.0) * float(I_max)

        p_list, q_list = [], []
        success_list, n_iter_list, Eraw_list, Escaled_list = [], [], [], []

        z0 = None
        for k in range(int(n_samples)):
            print(f"Sample {k+1}/{n_samples}")
            I = I_samples[k]
            tip, z_star, meta = self.fwd.solve(I, z0_bar=z0)
            if meta.success and np.isfinite(tip.p).all():
                z0 = z_star

            p_list.append(tip.p)
            q_list.append(tip.q)
            success_list.append(bool(meta.success))
            n_iter_list.append(int(meta.n_iter))
            Eraw_list.append(float(meta.final_E_raw))
            Escaled_list.append(float(meta.final_E_scaled))

        P = np.asarray(p_list, dtype=float)
        Q = np.asarray(q_list, dtype=float)
        S = np.asarray(success_list, dtype=bool)
        Nit = np.asarray(n_iter_list, dtype=int)
        Eraw = np.asarray(Eraw_list, dtype=float)
        Esc = np.asarray(Escaled_list, dtype=float)

        out: Dict[str, Any] = {
            "I": I_samples,
            "p": P,
            "q": Q,
            "success": S,
            "n_iter": Nit,
            "E_raw": Eraw,
            "E_scaled": Esc,
        }

        if save_npz:
            np.savez(save_npz, **out)

        vox_meta = None
        if voxel is not None and float(voxel) > 0:
            P_ok = P[S & np.isfinite(P).all(axis=1)]
            occ, origin, shape = voxelize(P_ok, voxel=float(voxel))
            vox_meta = {"voxel": float(voxel), "origin": origin, "shape": shape, "occ": occ}
            if save_voxels_npz:
                np.savez(save_voxels_npz, voxel=float(voxel), origin=origin, shape=shape, occ=occ)

        P_ok = P[S & np.isfinite(P).all(axis=1)]
        summary = {
            "n_samples": int(n_samples),
            "n_success": int(P_ok.shape[0]),
            "success_rate": float(P_ok.shape[0] / max(1, int(n_samples))),
            "p_min": P_ok.min(axis=0) if P_ok.shape[0] else np.full((3,), np.nan),
            "p_max": P_ok.max(axis=0) if P_ok.shape[0] else np.full((3,), np.nan),
            "p_mean": P_ok.mean(axis=0) if P_ok.shape[0] else np.full((3,), np.nan),
            "voxels": vox_meta,
        }
        out["summary"] = summary
        return out
