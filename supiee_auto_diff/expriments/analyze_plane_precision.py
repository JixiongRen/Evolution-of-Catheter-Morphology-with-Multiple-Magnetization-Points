# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


COMP_NAMES: List[str] = [
    "Bx",
    "By",
    "Bz",
    "dBx_dx",
    "dBx_dy",
    "dBx_dz",
    "dBy_dy",
    "dBy_dz",
]


def _reshape(v: np.ndarray, ny: int, nz: int) -> np.ndarray:
    return np.asarray(v, dtype=np.float64).reshape(ny, nz)


def _safe_rel_err(abs_err: np.ndarray, ref: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return abs_err / (np.abs(ref) + eps)


def _summary_stats(ref: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    ref = np.asarray(ref, dtype=np.float64).ravel()
    pred = np.asarray(pred, dtype=np.float64).ravel()
    d = pred - ref
    abs_err = np.abs(d)
    rel_err = _safe_rel_err(abs_err, ref)
    rmse = float(np.sqrt(np.mean(d * d)))
    # Pearson correlation (guard zeros)
    xr = ref - ref.mean() if ref.size else ref
    yp = pred - pred.mean() if pred.size else pred
    denom = float(np.linalg.norm(xr) * np.linalg.norm(yp))
    corr = float(np.dot(xr, yp) / denom) if denom > 0 else float("nan")
    return {
        "mean_abs": float(np.mean(abs_err)),
        "max_abs": float(np.max(abs_err)),
        "rmse": rmse,
        "mean_rel": float(np.mean(rel_err)),
        "max_rel": float(np.max(rel_err)),
        "corr": corr,
    }


def _add_colorbars(fig, axes, ims):
    # Add a colorbar next to each subplot heatmap
    for ax, im in zip(axes.ravel(), ims):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mag", type=str, default="plane_mag_manip.npz")
    parser.add_argument("--jax", type=str, default="plane_jax_interpolated.npz")
    parser.add_argument("--out_prefix", type=str, default="plane_precision")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--clip_rel_p99", type=float, default=99.0, help="Percentile to clip relative error colormap")
    parser.add_argument("--rel_alpha", type=float, default=1e-2, help="Alpha for adaptive floor: tau_k = alpha * p99(|ref_k|)")
    parser.add_argument("--weighted_stats", type=str, default="B_norm", choices=["none", "B_norm"], help="Weighting for mean stats")
    parser.add_argument("--angle_for_B", action="store_true", default=True, help="Enable B direction angle error outputs")
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
        raise ValueError("P 不匹配，请确保 jax 采样读取的是 mag 的 npz 文件")

    # 参考与预测的 8 分量
    Bm = np.asarray(mag["B"], dtype=np.float64)
    Gm = np.asarray(mag["G5"], dtype=np.float64)
    Bp = np.asarray(jax["B"], dtype=np.float64)
    Gp = np.asarray(jax["G5"], dtype=np.float64)
    # B magnitude from reference for weighting
    Bn_mag = np.asarray(mag["B_norm"], dtype=np.float64) if "B_norm" in mag else np.linalg.norm(Bm, axis=1)

    ref8 = np.concatenate([Bm, Gm], axis=1)  # (N,8)
    pred8 = np.concatenate([Bp, Gp], axis=1)

    # Adaptive floor-relative setup: tau_k = alpha * p99(|ref_k|)
    abs_ref8 = np.abs(ref8)
    taus = np.zeros((8,), dtype=np.float64)
    for k in range(8):
        rk = abs_ref8[:, k]
        rk = rk[np.isfinite(rk)]
        if rk.size == 0:
            taus[k] = 0.0
        else:
            p99 = float(np.percentile(rk, 99.0))
            taus[k] = float(args.rel_alpha) * p99

    abs_err8 = np.abs(pred8 - ref8)
    # floor-relative error per component
    rel_err8 = np.empty_like(abs_err8)
    for k in range(8):
        denom = np.maximum(np.abs(ref8[:, k]), taus[k])
        rel_err8[:, k] = abs_err8[:, k] / (denom + 1e-12)

    # 逐分量统计（mean_abs/max_abs/rmse/corr 基于绝对误差；mean_rel/max_rel 基于 floor-relative）
    comp_stats: Dict[str, Dict[str, float]] = {}
    for k, name in enumerate(COMP_NAMES):
        s = _summary_stats(ref8[:, k], pred8[:, k])
        # override rel stats by floor-relative results
        s["mean_rel"] = float(np.mean(rel_err8[:, k]))
        s["max_rel"] = float(np.max(rel_err8[:, k]))
        comp_stats[name] = s

    # Weighted stats using B_norm if requested
    weights = None
    if args.weighted_stats == "B_norm":
        w = np.asarray(Bn_mag, dtype=np.float64)
        w = np.clip(w, 0.0, np.inf)
        if np.all(~np.isfinite(w)) or float(np.nansum(w)) == 0.0:
            weights = None
        else:
            w[np.isnan(w)] = 0.0
            weights = w / (float(np.sum(w)) + 1e-12)

    w_mean_abs = []
    w_mean_rel = []
    for k in range(8):
        if weights is None:
            w_mean_abs.append(float(np.mean(abs_err8[:, k])))
            w_mean_rel.append(float(np.mean(rel_err8[:, k])))
        else:
            w_mean_abs.append(float(np.sum(weights * abs_err8[:, k])))
            w_mean_rel.append(float(np.sum(weights * rel_err8[:, k])))

    # 汇总柱状图数据
    mean_abs = [comp_stats[n]["mean_abs"] for n in COMP_NAMES]
    max_abs = [comp_stats[n]["max_abs"] for n in COMP_NAMES]
    mean_rel = [comp_stats[n]["mean_rel"] for n in COMP_NAMES]
    max_rel = [comp_stats[n]["max_rel"] for n in COMP_NAMES]

    # 可视化 1：绝对误差热力图（8 分量，2x4）
    extent = [float(zs[0]), float(zs[-1]), float(ys[0]), float(ys[-1])]
    fig1, axes1 = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    ims1 = []
    for k, name in enumerate(COMP_NAMES):
        r = _reshape(ref8[:, k], ny, nz)
        p = _reshape(pred8[:, k], ny, nz)
        abs_err = np.abs(p - r)
        vmin, vmax = float(np.min(abs_err)), float(np.max(abs_err))
        im = axes1[k // 4, k % 4].imshow(abs_err, origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap="magma", aspect="auto")
        axes1[k // 4, k % 4].set_title(f"|Δ| {name}")
        ims1.append(im)
    # add colorbars for absolute error heatmaps
    _add_colorbars(fig1, axes1, ims1)
    for j in range(4):
        axes1[1, j].set_xlabel("z")
    axes1[0, 0].set_ylabel("y")
    axes1[1, 0].set_ylabel("y")
    out1 = (Path(args.out_prefix).resolve().parent / (Path(args.out_prefix).name + "_abs_err.png")).resolve()
    fig1.savefig(out1, dpi=int(args.dpi))
    plt.close(fig1)

    # 可视化 2：相对误差热力图（按 p99 截断）
    fig2, axes2 = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    ims2 = []
    # 先计算全量 floor-relative 误差用于统一色阶
    rel_all = rel_err8.reshape(-1)
    finite_rel = rel_all[np.isfinite(rel_all)]
    if finite_rel.size == 0:
        vclip = 1.0
    else:
        q = float(np.percentile(finite_rel, float(args.clip_rel_p99)))
        vclip = q if q > 0 else float(np.max(finite_rel)) if finite_rel.size else 1.0
        if not np.isfinite(vclip) or vclip <= 0:
            vclip = 1.0

    for k, name in enumerate(COMP_NAMES):
        rel_err = _reshape(rel_err8[:, k], ny, nz)
        rel_err = np.clip(rel_err, 0.0, vclip)
        im = axes2[k // 4, k % 4].imshow(rel_err, origin="lower", extent=extent, vmin=0.0, vmax=vclip, cmap="viridis", aspect="auto")
        axes2[k // 4, k % 4].set_title(f"floor-rel Δ {name}")
        ims2.append(im)
    # add colorbars for relative error heatmaps
    _add_colorbars(fig2, axes2, ims2)
    for j in range(4):
        axes2[1, j].set_xlabel("z")
    axes2[0, 0].set_ylabel("y")
    axes2[1, 0].set_ylabel("y")
    out2 = (Path(args.out_prefix).resolve().parent / (Path(args.out_prefix).name + "_rel_err.png")).resolve()
    fig2.savefig(out2, dpi=int(args.dpi))
    plt.close(fig2)

    # 可视化 3：误差汇总柱状图（绝对/相对）
    x = np.arange(len(COMP_NAMES))
    width = 0.28

    fig3, ax3 = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
    ax3.bar(x - width / 2, w_mean_abs, width, label=("w_mean_abs (B_norm)" if args.weighted_stats == "B_norm" else "mean_abs"))
    ax3.bar(x + width / 2, max_abs, width, label="max_abs")
    ax3.set_xticks(x)
    ax3.set_xticklabels(COMP_NAMES, rotation=15, fontsize=9)
    ax3.set_title("Absolute error statistics")
    ax3.set_ylabel("abs error", fontsize=10)
    ax3.margins(x=0.02)
    ax3.legend(fontsize=9, ncol=2, loc="best")
    out3 = (Path(args.out_prefix).resolve().parent / (Path(args.out_prefix).name + "_abs_bars.png")).resolve()
    fig3.savefig(out3, dpi=int(args.dpi))
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
    ax4.bar(x - width / 2, w_mean_rel if args.weighted_stats != "none" else mean_rel, width, label=("w_mean_rel (B_norm)" if args.weighted_stats == "B_norm" else "mean_rel"))
    ax4.bar(x + width / 2, max_rel, width, label="max_rel")
    ax4.set_xticks(x)
    ax4.set_xticklabels(COMP_NAMES, rotation=15, fontsize=9)
    ax4.set_title("Relative error statistics")
    ax4.set_ylabel("relative error", fontsize=10)
    ax4.margins(x=0.02)
    ax4.legend(fontsize=9, ncol=2, loc="best")
    out4 = (Path(args.out_prefix).resolve().parent / (Path(args.out_prefix).name + "_rel_bars.png")).resolve()
    fig4.savefig(out4, dpi=int(args.dpi))
    plt.close(fig4)

    # Optional: B direction angle error (in degrees)
    out_angle = None
    angle_stats = None
    if args.angle_for_B:
        Bm_norm = np.linalg.norm(Bm, axis=1)
        Bp_norm = np.linalg.norm(Bp, axis=1)
        denom = (Bm_norm * Bp_norm) + 1e-12
        cosang = np.sum(Bm * Bp, axis=1) / denom
        cosang = np.clip(cosang, -1.0, 1.0)
        ang = np.degrees(np.arccos(cosang))  # (N,)
        # heatmap
        figA, axA = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)
        ang_map = _reshape(ang, ny, nz)
        imA = axA.imshow(ang_map, origin="lower", extent=extent, cmap="inferno", aspect="auto")
        axA.set_title("B direction angle error (deg)")
        axA.set_xlabel("z")
        axA.set_ylabel("y")
        figA.colorbar(imA, ax=axA, fraction=0.046, pad=0.04)
        out_angle = (Path(args.out_prefix).resolve().parent / (Path(args.out_prefix).name + "_B_angle.png")).resolve()
        figA.savefig(out_angle, dpi=int(args.dpi))
        plt.close(figA)
        # stats
        finite_ang = ang[np.isfinite(ang)]
        if finite_ang.size > 0:
            qs = np.percentile(finite_ang, [50, 90, 95, 99])
            angle_stats = {"p50": float(qs[0]), "p90": float(qs[1]), "p95": float(qs[2]), "p99": float(qs[3])}
        else:
            angle_stats = {"p50": float("nan"), "p90": float("nan"), "p95": float("nan"), "p99": float("nan")}

    # 保存数值指标（json + csv）
    out_json = (Path(args.out_prefix).resolve().parent / (Path(args.out_prefix).name + "_metrics.json")).resolve()
    with open(out_json, "w", encoding="utf-8") as f:
        payload = {
            "comp_stats": comp_stats,
            "taus": {name: float(taus[i]) for i, name in enumerate(COMP_NAMES)},
            "weights": ("B_norm" if args.weighted_stats == "B_norm" else "none"),
        }
        if angle_stats is not None:
            payload["B_angle_stats_deg"] = angle_stats
        json.dump(payload, f, ensure_ascii=False, indent=2)

    out_csv = (Path(args.out_prefix).resolve().parent / (Path(args.out_prefix).name + "_metrics.csv")).resolve()
    headers = ["component", "mean_abs", "max_abs", "rmse", "mean_rel", "max_rel", "corr"]
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for name in COMP_NAMES:
            s = comp_stats[name]
            row = [
                name,
                f"{s['mean_abs']:.10g}",
                f"{s['max_abs']:.10g}",
                f"{s['rmse']:.10g}",
                f"{s['mean_rel']:.10g}",
                f"{s['max_rel']:.10g}",
                f"{s['corr']:.10g}",
            ]
            f.write(",".join(row) + "\n")

    # 也保存 npz 便于后续复用
    out_npz = (Path(args.out_prefix).resolve().parent / (Path(args.out_prefix).name + "_errors.npz")).resolve()
    abs_err8 = np.abs(pred8 - ref8)
    rel_err8 = _safe_rel_err(abs_err8, ref8)
    np.savez(
        out_npz,
        ys=ys,
        zs=zs,
        P=Pm,
        ref8=ref8,
        pred8=pred8,
        abs_err8=abs_err8,
        rel_err8=rel_err8,
        taus=taus,
        comp_names=np.array(COMP_NAMES, dtype=object),
        mag_file=str(mag_path),
        jax_file=str(jax_path),
    )

    print("saved:", out1)
    print("saved:", out2)
    print("saved:", out3)
    print("saved:", out4)
    print("saved:", out_json)
    print("saved:", out_csv)
    print("saved:", out_npz)
    if out_angle is not None:
        print("saved:", out_angle)


if __name__ == "__main__":
    main()
