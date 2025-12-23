# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


G5_NAMES = ["dBx_dx", "dBx_dy", "dBx_dz", "dBy_dy", "dBy_dz"]


def _reshape_field(v: np.ndarray, ny: int, nz: int) -> np.ndarray:
    return np.asarray(v, dtype=np.float64).reshape(ny, nz)


def _abs_stats(name: str, a: np.ndarray, b: np.ndarray) -> None:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    print(name, "abs diff mean:", float(np.mean(np.abs(d))), "max:", float(np.max(np.abs(d))))


def _plot_grid(fig, axes, fields_top, fields_bottom, titles, extent, cmap="viridis"):
    for j, title in enumerate(titles):
        a0 = fields_top[j]
        a1 = fields_bottom[j]
        vmin = float(min(np.min(a0), np.min(a1)))
        vmax = float(max(np.max(a0), np.max(a1)))
        im0 = axes[0, j].imshow(a0, origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        im1 = axes[1, j].imshow(a1, origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        axes[0, j].set_title("mag " + title)
        axes[1, j].set_title("jax " + title)
        fig.colorbar(im0, ax=axes[:, j], fraction=0.046, pad=0.04)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mag", type=str, default="plane_mag_manip.npz")
    parser.add_argument("--jax", type=str, default="plane_jax_interpolated.npz")
    parser.add_argument("--out_prefix", type=str, default="plane_fields")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    mag_path = Path(args.mag).resolve()
    jax_path = Path(args.jax).resolve()
    if not mag_path.exists():
        raise FileNotFoundError(mag_path)
    if not jax_path.exists():
        raise FileNotFoundError(jax_path)

    mag = np.load(mag_path, allow_pickle=True)
    jax = np.load(jax_path, allow_pickle=True)

    ys = np.asarray(mag["ys"], dtype=np.float64)
    zs = np.asarray(mag["zs"], dtype=np.float64)
    ny, nz = ys.size, zs.size

    Pm = np.asarray(mag["P"], dtype=np.float64)
    Pj = np.asarray(jax["P"], dtype=np.float64)
    if Pm.shape != Pj.shape or not np.allclose(Pm, Pj, rtol=0.0, atol=0.0):
        raise ValueError("P mismatch. Ensure jax sampling reads the mag npz.")

    B_mag = np.asarray(mag["B"], dtype=np.float64)
    G_mag = np.asarray(mag["G5"], dtype=np.float64)
    Bn_mag = np.asarray(mag["B_norm"], dtype=np.float64)

    B_jax = np.asarray(jax["B"], dtype=np.float64)
    G_jax = np.asarray(jax["G5"], dtype=np.float64)
    Bn_jax = np.asarray(jax["B_norm"], dtype=np.float64)

    for i, name in enumerate(["Bx", "By", "Bz"]):
        _abs_stats(name, B_mag[:, i], B_jax[:, i])
    _abs_stats("|B|", Bn_mag, Bn_jax)
    for i, name in enumerate(G5_NAMES):
        _abs_stats(name, G_mag[:, i], G_jax[:, i])

    extent = [float(zs[0]), float(zs[-1]), float(ys[0]), float(ys[-1])]

    fig1, axes1 = plt.subplots(2, 4, figsize=(16, 6), constrained_layout=True)
    titles1 = ["Bx", "By", "Bz", "|B|"]
    fields_mag_1 = [
        _reshape_field(B_mag[:, 0], ny, nz),
        _reshape_field(B_mag[:, 1], ny, nz),
        _reshape_field(B_mag[:, 2], ny, nz),
        _reshape_field(Bn_mag, ny, nz),
    ]
    fields_jax_1 = [
        _reshape_field(B_jax[:, 0], ny, nz),
        _reshape_field(B_jax[:, 1], ny, nz),
        _reshape_field(B_jax[:, 2], ny, nz),
        _reshape_field(Bn_jax, ny, nz),
    ]
    _plot_grid(fig1, axes1, fields_mag_1, fields_jax_1, titles1, extent)
    axes1[0, 0].set_ylabel("y")
    axes1[1, 0].set_ylabel("y")
    for j in range(4):
        axes1[1, j].set_xlabel("z")

    out1 = (Path(args.out_prefix).resolve().parent / (Path(args.out_prefix).name + "_B.png")).resolve()
    fig1.savefig(out1, dpi=int(args.dpi))
    plt.close(fig1)

    fig2, axes2 = plt.subplots(2, 5, figsize=(20, 6), constrained_layout=True)
    fields_mag_2 = [_reshape_field(G_mag[:, i], ny, nz) for i in range(5)]
    fields_jax_2 = [_reshape_field(G_jax[:, i], ny, nz) for i in range(5)]
    _plot_grid(fig2, axes2, fields_mag_2, fields_jax_2, G5_NAMES, extent)
    axes2[0, 0].set_ylabel("y")
    axes2[1, 0].set_ylabel("y")
    for j in range(5):
        axes2[1, j].set_xlabel("z")

    out2 = (Path(args.out_prefix).resolve().parent / (Path(args.out_prefix).name + "_G5.png")).resolve()
    fig2.savefig(out2, dpi=int(args.dpi))
    plt.close(fig2)

    print("saved:", out1)
    print("saved:", out2)


if __name__ == "__main__":
    main()
