"""IK artifact saving & plotting utilities.

This module is intentionally *side-effect free* except for explicit save/plot
functions. It is used by ik_position_gn.py to:

- Extract a full catheter centerline (in SI units) from a packed state z_bar.
- Save accepted IK centerlines and key metrics to disk.
- Generate summary figures after IK finishes.

All plotting is headless (matplotlib Agg) so it works on servers.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Matplotlib: headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from nondim import x_bar_to_dim
from utils_nondim import unpack_z_bar_jax

Array = jnp.ndarray


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_numpy(a: Any) -> np.ndarray:
    """Device -> host conversion with dtype preservation."""
    return np.asarray(jax.device_get(a))


def extract_centerline_dim(z_bar: Array, *, M_list: Sequence[int], scales) -> np.ndarray:
    """Extract a full catheter centerline as an (N,3) numpy array in SI units.

    The centerline is assembled as:
      [flex nodes of seg0] + [rigid distal of seg0] + ... + [flex nodes of last] + [rigid distal of last]

    Notes
    -----
    - We include rigid distal points because the physical tip is the last rigid distal.
    - There may be duplicated points at boundaries; for visualization this is fine.
    """
    x_nodes_list_bar, _k_list_bar, x_rigid_list_bar = unpack_z_bar_jax(z_bar, M_list=M_list)

    pts: List[np.ndarray] = []

    # helper: vectorized x_bar_to_dim
    def _x_to_p_dim(x_bar: Array) -> Array:
        x_dim = x_bar_to_dim(x_bar, scales)
        return x_dim[0:3]

    vmap_p = jax.vmap(_x_to_p_dim)

    for seg_idx, x_nodes_bar in enumerate(x_nodes_list_bar):
        p_nodes = vmap_p(x_nodes_bar)  # (Mi+1,3)
        pts.append(_to_numpy(p_nodes))

        # rigid distal state for this segment
        xR_bar = x_rigid_list_bar[seg_idx]
        pR = _x_to_p_dim(xR_bar)
        pts.append(_to_numpy(pR).reshape(1, 3))

    if not pts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(pts, axis=0)


def save_centerline(centerline_xyz: np.ndarray, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, centerline_xyz)
    # Defensive: ensure file really exists on disk (helpful when running in remote IDEs)
    if not out_path.exists():
        # np.save may append .npy automatically
        alt = out_path.with_suffix(out_path.suffix + ".npy")
        if alt.exists():
            return
        raise RuntimeError(f"save_centerline: file was not created: {str(out_path)}")


def save_history_json(history: Dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            # Some FS (e.g., certain network mounts) do not support fsync.
            pass
    if not out_path.exists():
        raise RuntimeError(f"save_history_json: file was not created: {str(out_path)}")


def plot_centerlines_3d(
    centerlines: Sequence[np.ndarray],
    *,
    p_tip_init: Optional[np.ndarray],
    p_tip_final: Optional[np.ndarray],
    p_des: Optional[np.ndarray],
    out_path: str | Path,
    title: str = "IK accepted centerlines",
) -> None:
    """Plot accepted centerlines and mark initial/final tip points."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # draw all accepted centerlines
    for i, cl in enumerate(centerlines):
        if cl.size == 0:
            continue
        ax.plot(cl[:, 0], cl[:, 1], cl[:, 2], linewidth=1.0, alpha=0.35)

    if p_tip_init is not None and np.all(np.isfinite(p_tip_init)):
        ax.scatter([p_tip_init[0]], [p_tip_init[1]], [p_tip_init[2]], c="r", s=35, label="tip init")

    if p_tip_final is not None and np.all(np.isfinite(p_tip_final)):
        ax.scatter([p_tip_final[0]], [p_tip_final[1]], [p_tip_final[2]], c="g", s=35, label="tip final")

    if p_des is not None and np.all(np.isfinite(p_des)):
        ax.scatter([p_des[0]], [p_des[1]], [p_des[2]], c="b", s=35, marker="*", label="p_des")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(title)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    if not out_path.exists():
        raise RuntimeError(f"plot_centerlines_3d: image was not created: {str(out_path)}")


def plot_metrics(history: Dict[str, Any], *, out_dir: str | Path) -> None:
    """Plot a few key metrics from IK history.

    We plot both (1) per-outer-iter traces and (2) per-accepted-step traces.
    """

    out_dir = ensure_dir(out_dir)

    outer = history.get("outer", [])
    if not outer:
        return

    # Build arrays
    k = np.asarray([d.get("iter", i) for i, d in enumerate(outer)], dtype=np.int64)
    err_mm = np.asarray([d.get("err_mm", np.nan) for d in outer], dtype=np.float64)
    normE = np.asarray([d.get("normE", np.nan) for d in outer], dtype=np.float64)
    mu = np.asarray([d.get("mu", np.nan) for d in outer], dtype=np.float64)
    lm_iter = np.asarray([d.get("lm_iter", np.nan) for d in outer], dtype=np.float64)

    # 1) err_mm vs k
    fig = plt.figure(figsize=(7.5, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(k, err_mm)
    ax.set_xlabel("outer iter")
    ax.set_ylabel("||e_p|| (mm)")
    ax.set_title("Tip position error")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "curve_err_mm.png", dpi=200)
    plt.close(fig)
    if not (out_dir / "curve_err_mm.png").exists():
        raise RuntimeError(f"plot_metrics: file not created: {str(out_dir / 'curve_err_mm.png')}")

    # 2) normE vs k (log)
    fig = plt.figure(figsize=(7.5, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(k, normE)
    ax.set_yscale("log")
    ax.set_xlabel("outer iter")
    ax.set_ylabel("||E||")
    ax.set_title("Inner equilibrium residual norm")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "curve_normE.png", dpi=200)
    plt.close(fig)
    if not (out_dir / "curve_normE.png").exists():
        raise RuntimeError(f"plot_metrics: file not created: {str(out_dir / 'curve_normE.png')}")

    # 3) lm_iter vs k
    fig = plt.figure(figsize=(7.5, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(k, lm_iter)
    ax.set_xlabel("outer iter")
    ax.set_ylabel("LM iterations")
    ax.set_title("Inner LM iteration count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "curve_lm_iter.png", dpi=200)
    plt.close(fig)
    if not (out_dir / "curve_lm_iter.png").exists():
        raise RuntimeError(f"plot_metrics: file not created: {str(out_dir / 'curve_lm_iter.png')}")

    # 4) mu vs k (log)
    fig = plt.figure(figsize=(7.5, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(k, mu)
    ax.set_yscale("log")
    ax.set_xlabel("outer iter")
    ax.set_ylabel("mu")
    ax.set_title("Outer damping (mu)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "curve_mu.png", dpi=200)
    plt.close(fig)
    if not (out_dir / "curve_mu.png").exists():
        raise RuntimeError(f"plot_metrics: file not created: {str(out_dir / 'curve_mu.png')}")
