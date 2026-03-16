from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_get(d: Dict[str, Any], keys: Sequence[str], default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _read_trace_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _stack(trace: List[Dict[str, Any]], key: str, *, default=np.nan) -> np.ndarray:
    out: List[float] = []
    for d in trace:
        v = d.get(key, default)
        try:
            out.append(float(v))
        except Exception:
            out.append(float(default))
    return np.asarray(out, dtype=np.float64)


def _stack_vec(trace: List[Dict[str, Any]], key: str, n: int) -> np.ndarray:
    arr = np.full((len(trace), n), np.nan, dtype=np.float64)
    for i, d in enumerate(trace):
        v = d.get(key, None)
        if v is None:
            continue
        try:
            xs = np.asarray(v, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        m = min(n, xs.size)
        arr[i, :m] = xs[:m]
    return arr


def plot_3d_convergence(run_dir: Path, *, out_path: Path, max_lines: int = 200) -> None:
    cfg = _read_json(run_dir / "run_config.json")

    p_des = _maybe_get(cfg, ["target", "p_des_m"], _maybe_get(cfg, ["p_des"], None))
    p_des_np = None if p_des is None else np.asarray(p_des, dtype=np.float64).reshape(3,)

    init_rel = _maybe_get(cfg, ["init", "centerline_npz"], None)
    fin_rel = _maybe_get(cfg, ["final", "centerline_npz"], None)

    init_cl = None
    if init_rel is not None and (run_dir / init_rel).exists():
        init_cl = np.load(run_dir / init_rel)["centerline"]

    fin_cl = None
    if fin_rel is not None and (run_dir / fin_rel).exists():
        fin_cl = np.load(run_dir / fin_rel)["centerline"]

    poses_dir = run_dir / "poses"
    cls = sorted(poses_dir.glob("iter_*.npz"))
    if max_lines > 0 and len(cls) > max_lines:
        idx = np.linspace(0, len(cls) - 1, max_lines).astype(int)
        cls = [cls[i] for i in idx]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    for p in cls:
        try:
            cl = np.load(p)["centerline"]
        except Exception:
            continue
        if cl.size == 0:
            continue
        ax.plot(cl[:, 0], cl[:, 1], cl[:, 2], linewidth=1.0, alpha=0.18, c="k")

    if init_cl is not None and init_cl.size != 0:
        ax.plot(init_cl[:, 0], init_cl[:, 1], init_cl[:, 2], linewidth=2.2, alpha=0.9, c="r", label="init")
        ax.scatter([init_cl[-1, 0]], [init_cl[-1, 1]], [init_cl[-1, 2]], c="r", s=35)

    if fin_cl is not None and fin_cl.size != 0:
        ax.plot(fin_cl[:, 0], fin_cl[:, 1], fin_cl[:, 2], linewidth=2.2, alpha=0.9, c="g", label="final")
        ax.scatter([fin_cl[-1, 0]], [fin_cl[-1, 1]], [fin_cl[-1, 2]], c="g", s=35)

    if p_des_np is not None and np.all(np.isfinite(p_des_np)):
        ax.scatter([p_des_np[0]], [p_des_np[1]], [p_des_np[2]], c="b", s=70, marker="*", label="target")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Single-layer IK centerline convergence")
    ax.legend(loc="best")
    ax.set_xlim([-0.10, 0.10])
    ax.set_ylim([-0.10, 0.10])
    ax.set_zlim([-0.10, 0.10])

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.show()
    # plt.close(fig)


def plot_timeseries(run_dir: Path, *, out_dir: Path) -> None:
    cfg = _read_json(run_dir / "run_config.json")
    trace_rel = _maybe_get(cfg, ["trace", "path"], "trace.jsonl")
    trace = _read_trace_jsonl(run_dir / trace_rel)
    if not trace:
        return

    k = np.asarray([int(d.get("iter", i)) for i, d in enumerate(trace)], dtype=np.int64)

    dist_mm = _stack(trace, "dist_to_target_mm")
    normR = _stack(trace, "normR")
    cost = _stack(trace, "cost")

    x_m = _stack(trace, "x_m")
    dx_m = _stack(trace, "dx_m")

    lam = _stack(trace, "lam")
    rho = _stack(trace, "rho")

    I_A = _stack_vec(trace, "I_A", 8)
    dI_A = _stack_vec(trace, "dI_A", 8)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) distance + normR + cost
    fig, ax1 = plt.subplots(figsize=(9.0, 4.8))
    ax1.plot(k, dist_mm, label="||p_tip - p_des|| (mm)")
    ax1.set_xlabel("IK iter")
    ax1.set_ylabel("distance (mm)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(k, normR, color="tab:orange", alpha=0.75, label="||R||")
    ax2.plot(k, cost, color="tab:green", alpha=0.55, label="cost")
    ax2.set_ylabel("norm/cost")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")

    fig.tight_layout()
    fig.savefig(out_dir / "ts_dist_normR_cost.png", dpi=220)
    plt.close(fig)

    # 2) insertion depth and increment
    fig, ax1 = plt.subplots(figsize=(9.0, 4.8))
    ax1.plot(k, x_m * 1e3, label="x (mm)")
    ax1.set_xlabel("IK iter")
    ax1.set_ylabel("x (mm)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(k, dx_m * 1e3, color="tab:purple", alpha=0.8, label="Δx (mm)")
    ax2.set_ylabel("Δx (mm)")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")

    fig.tight_layout()
    fig.savefig(out_dir / "ts_x_dx.png", dpi=220)
    plt.close(fig)

    # 3) currents
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    for i in range(I_A.shape[1]):
        ax.plot(k, I_A[:, i], linewidth=1.2, alpha=0.85, label=f"I{i+1}")
    ax.set_xlabel("IK iter")
    ax.set_ylabel("I (A)")
    ax.set_title("coil currents")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "ts_I.png", dpi=220)
    plt.close(fig)

    # 4) delta currents
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    for i in range(dI_A.shape[1]):
        ax.plot(k, dI_A[:, i], linewidth=1.2, alpha=0.85, label=f"ΔI{i+1}")
    ax.set_xlabel("IK iter")
    ax.set_ylabel("ΔI (A)")
    ax.set_title("coil current increments")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "ts_dI.png", dpi=220)
    plt.close(fig)

    # 5) LM: lambda (log) and rho
    fig, ax1 = plt.subplots(figsize=(9.0, 4.8))
    ax1.plot(k, lam, label="lambda")
    ax1.set_yscale("log")
    ax1.set_xlabel("IK iter")
    ax1.set_ylabel("lambda (log)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(k, rho, color="tab:red", alpha=0.8, label="rho")
    ax2.set_ylabel("rho")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")

    fig.tight_layout()
    fig.savefig(out_dir / "ts_lam_rho.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot single-layer IK run outputs")
    ap.add_argument("--run_dir", type=str, required=True, help="ik_out/run_*/ directory")
    ap.add_argument("--out_dir", type=str, default=None, help="output dir for plots (default: <run_dir>/plots)")
    ap.add_argument("--max_lines", type=int, default=200, help="max centerlines drawn in 3D plot")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else (run_dir / "plots")

    plot_3d_convergence(run_dir, out_path=out_dir / "centerline_3d.png", max_lines=int(args.max_lines))
    plot_timeseries(run_dir, out_dir=out_dir)


if __name__ == "__main__":
    main()
