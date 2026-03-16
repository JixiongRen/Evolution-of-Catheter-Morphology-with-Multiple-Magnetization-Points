# plot_workspace.py
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt


def _load_npz(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)


def _finite_rows(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    return np.isfinite(P).all(axis=1)


def plot_point_cloud(samples_npz: str, out_dir: str):
    data = _load_npz(samples_npz)
    P = data["p"]          # (N,3)
    S = data["success"]    # (N,)
    Nit = data.get("n_iter", None)
    Eraw = data.get("E_raw", None)

    ok = (S.astype(bool)) & _finite_rows(P)
    bad = (~S.astype(bool)) & _finite_rows(P)

    P_ok = P[ok]
    P_bad = P[bad]

    print(f"[plot] total={len(P)}, ok={len(P_ok)}, bad={len(P_bad)}, success_rate={len(P_ok)/max(1,len(P)):.3f}")
    if len(P_ok) > 0:
        print("  p_min (m):", P_ok.min(axis=0))
        print("  p_max (m):", P_ok.max(axis=0))

    # --- 3D scatter ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if len(P_bad) > 0:
        ax.scatter(P_bad[:, 0], P_bad[:, 1], P_bad[:, 2], s=2, alpha=0.2, label="fail")
    if len(P_ok) > 0:
        ax.scatter(P_ok[:, 0], P_ok[:, 1], P_ok[:, 2], s=3, alpha=0.6, label="success")
    ax.set_title("Tip position point cloud (3D)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "point_cloud_3d.png"), dpi=200)

    # --- XY / XZ / YZ projections ---
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    def scatter2(ax, a, b, title):
        if len(P_bad) > 0:
            ax.scatter(P_bad[:, a], P_bad[:, b], s=2, alpha=0.2)
        if len(P_ok) > 0:
            ax.scatter(P_ok[:, a], P_ok[:, b], s=3, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel(["x","y","z"][a] + " (m)")
        ax.set_ylabel(["x","y","z"][b] + " (m)")
        ax.axis("equal")

    scatter2(axs[0], 0, 1, "XY projection")
    scatter2(axs[1], 0, 2, "XZ projection")
    scatter2(axs[2], 1, 2, "YZ projection")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "point_cloud_projections.png"), dpi=200)

    # --- diagnostics histograms (optional) ---
    if Nit is not None:
        Nit = np.asarray(Nit)
        fig = plt.figure()
        plt.hist(Nit[ok], bins=30)
        plt.title("LM iterations (success only)")
        plt.xlabel("n_iter")
        plt.ylabel("count")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "hist_n_iter.png"), dpi=200)

    if Eraw is not None:
        Eraw = np.asarray(Eraw, dtype=float)
        Eraw_ok = Eraw[ok]
        Eraw_ok = Eraw_ok[np.isfinite(Eraw_ok)]
        if len(Eraw_ok) > 0:
            fig = plt.figure()
            plt.hist(np.log10(Eraw_ok + 1e-30), bins=40)
            plt.title("log10(||E_raw||) (success only)")
            plt.xlabel("log10(||E_raw||)")
            plt.ylabel("count")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "hist_logEraw.png"), dpi=200)

    plt.close("all")


def plot_voxels(voxels_npz: str, out_dir: str, max_points: int = 200_000):
    """
    Visualize occupancy voxels as a 3D scatter of voxel centers.
    Note: if the grid is large, this can be heavy; max_points caps the scatter size.
    """
    data = _load_npz(voxels_npz)
    occ = data["occ"].astype(bool)   # (nx,ny,nz)
    origin = np.asarray(data["origin"], dtype=float).reshape(3,)
    voxel = float(data["voxel"])
    shape = np.asarray(data["shape"], dtype=int).reshape(3,)

    idx = np.argwhere(occ)
    n = idx.shape[0]
    print(f"[plot] voxels occupied={n}, shape={tuple(shape.tolist())}, voxel={voxel} m")

    if n == 0:
        return

    # Downsample if too many voxels
    if n > max_points:
        sel = np.random.default_rng(0).choice(n, size=max_points, replace=False)
        idx = idx[sel]
        n = idx.shape[0]
        print(f"[plot] downsampled voxels to {n}")

    centers = origin + (idx + 0.5) * voxel  # (n,3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=2, alpha=0.5)
    ax.set_title("Workspace occupancy voxels (centers)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "voxels_3d.png"), dpi=200)
    plt.close(fig)


def main():
    out_dir = "workspace_out"
    samples_npz = os.path.join(out_dir, "samples.npz")
    voxels_npz = os.path.join(out_dir, "workspace_voxels.npz")

    os.makedirs(out_dir, exist_ok=True)

    plot_point_cloud(samples_npz, out_dir)

    if os.path.exists(voxels_npz):
        plot_voxels(voxels_npz, out_dir)
    else:
        print("[plot] workspace_voxels.npz not found; skip voxel plot.")


if __name__ == "__main__":
    main()
