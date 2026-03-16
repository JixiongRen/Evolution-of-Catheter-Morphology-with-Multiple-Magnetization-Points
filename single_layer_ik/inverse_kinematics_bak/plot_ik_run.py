from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
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


def _load_centerline(run_dir: Path, rel: Optional[str]) -> Optional[np.ndarray]:
    if rel is None:
        return None
    p = (run_dir / rel)
    if not p.exists():
        return None
    return np.load(p)


def _stack_history(outer: List[Dict[str, Any]], key: str, *, default=np.nan) -> np.ndarray:
    out: List[float] = []
    for d in outer:
        v = d.get(key, default)
        try:
            out.append(float(v))
        except Exception:
            out.append(float(default))
    return np.asarray(out, dtype=np.float64)


def _stack_vec_history(outer: List[Dict[str, Any]], key: str, n: int) -> np.ndarray:
    arr = np.full((len(outer), n), np.nan, dtype=np.float64)
    for i, d in enumerate(outer):
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


def plot_3d_centerlines(
    run_dir: Path,
    *,
    out_path: Path,
    max_lines: int = 200,
) -> None:
    cfg = _read_json(run_dir / "run_config.json")

    p_des = _maybe_get(cfg, ["p_des"], None)
    p_des_np = None if p_des is None else np.asarray(p_des, dtype=np.float64).reshape(3,)

    init_cl = _load_centerline(run_dir, _maybe_get(cfg, ["init", "centerline_npy"], None))
    fin_cl = _load_centerline(run_dir, _maybe_get(cfg, ["final", "centerline_npy"], None))

    accept_dir = run_dir / "poses" / "accept"
    cls = sorted(accept_dir.glob("centerline_iter_*.npy"))
    if max_lines > 0 and len(cls) > max_lines:
        idx = np.linspace(0, len(cls) - 1, max_lines).astype(int)
        cls = [cls[i] for i in idx]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    for p in cls:
        cl = np.load(p)
        if cl.size == 0:
            continue
        ax.plot(cl[:, 0], cl[:, 1], cl[:, 2], linewidth=1.0, alpha=0.25, c="k")

    if init_cl is not None and init_cl.size != 0:
        ax.plot(init_cl[:, 0], init_cl[:, 1], init_cl[:, 2], linewidth=2.0, alpha=0.9, c="r", label="init")
        ax.scatter([init_cl[-1, 0]], [init_cl[-1, 1]], [init_cl[-1, 2]], c="r", s=35)

    if fin_cl is not None and fin_cl.size != 0:
        ax.plot(fin_cl[:, 0], fin_cl[:, 1], fin_cl[:, 2], linewidth=2.0, alpha=0.9, c="g", label="final")
        ax.scatter([fin_cl[-1, 0]], [fin_cl[-1, 1]], [fin_cl[-1, 2]], c="g", s=35)

    if p_des_np is not None and np.all(np.isfinite(p_des_np)):
        ax.scatter([p_des_np[0]], [p_des_np[1]], [p_des_np[2]], c="b", s=60, marker="*", label="target")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("IK centerline convergence (accepted states)")
    ax.legend(loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_timeseries(run_dir: Path, *, out_dir: Path) -> None:
    hist = _read_json(run_dir / "history.json")
    outer = hist.get("outer", [])
    if not outer:
        return

    k = np.asarray([int(d.get("iter", i)) for i, d in enumerate(outer)], dtype=np.int64)

    err_mm = _stack_history(outer, "err_mm")
    cost = _stack_history(outer, "cost")
    lam = _stack_history(outer, "lam")
    rho = _stack_history(outer, "rho")
    pred_red = _stack_history(outer, "pred_red")
    act_red = _stack_history(outer, "act_red")

    x_m = _stack_history(outer, "x_m")
    dx_m = _stack_history(outer, "dx_m")

    I_A = _stack_vec_history(outer, "I_A", 8)
    dI_A = _stack_vec_history(outer, "dI_A", 8)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) error + cost
    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    ax1.plot(k, err_mm, label="||e|| (mm)")
    ax1.set_xlabel("IK iter")
    ax1.set_ylabel("error (mm)")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(k, cost, color="tab:orange", alpha=0.8, label="cost")
    ax2.set_ylabel("cost")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "ts_err_cost.png", dpi=200)
    plt.close(fig)

    # 2) x and dx
    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    ax1.plot(k, x_m * 1e3, label="x (mm)")
    ax1.set_xlabel("IK iter")
    ax1.set_ylabel("x (mm)")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(k, dx_m * 1e3, color="tab:green", alpha=0.8, label="Δx (mm)")
    ax2.set_ylabel("Δx (mm)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "ts_x_dx.png", dpi=200)
    plt.close(fig)

    # 3) currents
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for i in range(I_A.shape[1]):
        ax.plot(k, I_A[:, i], linewidth=1.2, alpha=0.85, label=f"I{i}")
    ax.set_xlabel("IK iter")
    ax.set_ylabel("I (A)")
    ax.set_title("coil currents")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "ts_I.png", dpi=200)
    plt.close(fig)

    # 4) delta currents
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for i in range(dI_A.shape[1]):
        ax.plot(k, dI_A[:, i], linewidth=1.2, alpha=0.85, label=f"ΔI{i}")
    ax.set_xlabel("IK iter")
    ax.set_ylabel("ΔI (A)")
    ax.set_title("coil current increments")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "ts_dI.png", dpi=200)
    plt.close(fig)

    # 5) lm params
    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
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
    fig.savefig(out_dir / "ts_lam_rho.png", dpi=200)
    plt.close(fig)

    # 6) predicted vs actual reduction
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(k, pred_red, label="pred_red")
    ax.plot(k, act_red, label="act_red")
    ax.set_xlabel("IK iter")
    ax.set_ylabel("reduction")
    ax.set_title("predicted vs actual reduction")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "ts_pred_act_red.png", dpi=200)
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, default="ik_out/run_20260203_224212", help="Path to ik_out/run_xxx")
    p.add_argument("--out_dir", type=str, default=None, help="Output dir for plots. Default: <run_dir>/plots")
    p.add_argument("--max_lines", type=int, default=200)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir is not None else (run_dir / "plots")

    plot_3d_centerlines(run_dir, out_path=out_dir / "centerlines_3d.png", max_lines=int(args.max_lines))
    plot_timeseries(run_dir, out_dir=out_dir)

    print(f"[plot_ik_run] wrote plots to: {str(out_dir)}")


if __name__ == "__main__":
    main()
