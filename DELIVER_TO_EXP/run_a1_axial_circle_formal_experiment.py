"""Run formal A1/A2 single-magnet supervised IK circle experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "catheter_physical_configs.yaml"
OUT_ROOT = ROOT / "formal_experiments"
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _font() -> FontProperties:
    if FONT_PATH.exists():
        return FontProperties(fname=str(FONT_PATH))
    return FontProperties()


def _vec_csv(values: list[float] | tuple[float, ...] | np.ndarray) -> str:
    return ",".join(f"{float(v):.12g}" for v in values)


def _m_body_csv(values: list[list[float]]) -> str:
    return "; ".join(_vec_csv(v) for v in values)


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def _load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _catheter_solver_cfg(config: dict[str, Any], catheter_key: str) -> dict[str, Any]:
    return {
        "shared": config["shared_physical_parameters"],
        "solver_defaults": config["solver_defaults"],
        "catheter": config["catheter_types"][catheter_key],
    }


def _circle_targets(*, n_points: int, center_m: tuple[float, float, float], radius_m: float) -> list[dict[str, Any]]:
    """Generate targets on a circle whose normal is the x-axis."""
    targets: list[dict[str, Any]] = []
    cx, cy, cz = center_m
    for i in range(n_points):
        theta = 2.0 * math.pi * float(i) / float(n_points)
        targets.append(
            {
                "index": int(i),
                "theta_rad": float(theta),
                "p_des_m": [
                    float(cx),
                    float(cy + radius_m * math.cos(theta)),
                    float(cz + radius_m * math.sin(theta)),
                ],
            }
        )
    return targets


def _smooth_centerline(points_m: np.ndarray, *, samples_per_segment: int = 24) -> np.ndarray:
    """Fit a smooth display curve through sparse centerline nodes."""
    points = np.asarray(points_m, dtype=float)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 3:
        return points
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    keep = np.concatenate([[True], np.diff(s) > 1e-12])
    points = points[keep]
    s = s[keep]
    if points.shape[0] < 3 or s[-1] <= 1e-12:
        return points
    n_samples = max(int(samples_per_segment) * (points.shape[0] - 1) + 1, points.shape[0])
    s_new = np.linspace(0.0, s[-1], n_samples)
    try:
        from scipy.interpolate import CubicSpline

        return np.column_stack([CubicSpline(s, points[:, j], bc_type="natural")(s_new) for j in range(3)])
    except Exception:
        return np.column_stack([np.interp(s_new, s, points[:, j]) for j in range(3)])


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _history_counts(history: dict[str, Any]) -> tuple[int, int, int]:
    outer = list(history.get("outer", []))
    accepted = sum(1 for item in outer if bool(item.get("accept", False)))
    rejected = sum(1 for item in outer if str(item.get("status", "")).upper() == "REJECT")
    last_iter = max([int(item.get("iter", 0)) for item in outer] or [0])
    return accepted, rejected, last_iter


def _best_error(history: dict[str, Any], field: str) -> float:
    vals = [float(item[field]) for item in history.get("outer", []) if item.get(field) is not None]
    if not vals:
        return math.nan
    return float(np.nanmin(np.asarray(vals, dtype=float)))


def _centerline_path(point_dir: Path) -> Path | None:
    run_config = _read_json(point_dir / "run_config.json", {})
    for section in ("final", "last_accept", "init"):
        rel = run_config.get(section, {}).get("centerline_npy")
        if rel:
            path = point_dir / rel
            if path.exists():
                return path
    return None


def _infer_stop_reason(*, result: dict[str, Any], history: dict[str, Any], outer_max_iter: int, threshold_mm: float) -> str:
    err_true = float(result.get("err_true_mm", math.inf))
    err_flex = float(result.get("err_flex_mm", result.get("err_mm", math.inf)))
    cls = str(result.get("cls", ""))
    _accepted, rejected, last_iter = _history_counts(history)
    if err_true <= float(threshold_mm):
        return "reached_threshold_true_tip"
    if err_flex <= float(threshold_mm) and err_true > float(threshold_mm):
        return "flex_tip_only_reached"
    if err_true <= 10.0:
        return "reached_10mm_only"
    if cls == "REJECT":
        return "fk_rejected"
    if rejected >= 5 and rejected >= max(3, last_iter // 2):
        return "reject_stall"
    if last_iter >= int(outer_max_iter):
        return "max_outer_iter"
    return "early_stop_or_plateau"


def _build_ik_args(
    *,
    cfg: dict[str, Any],
    catheter_key: str,
    target: dict[str, Any],
    phase_dir: Path,
    run_name: str,
    initial_I_A: list[float] | None,
    initial_x_m: float,
    x_search_max_m: float,
    enable_gravity: bool,
    outer_max_iter: int,
    eps_p_m: float,
) -> list[str]:
    shared = cfg["shared"]
    catheter = cfg["catheter"]
    solver = catheter["solver_input"]
    magnet = shared["magnet"]
    flex = shared["flexible_body"]
    actuation = shared["actuation"]
    defaults = cfg["solver_defaults"]
    initial_I_A = initial_I_A if initial_I_A is not None else [0.0] * int(actuation["coil_count"])
    x_max = min(float(solver["L_protrude_max_m"]), float(x_search_max_m))
    args = [
        "--flex_lengths",
        _vec_csv(solver["cli_flex_lengths_placeholder_m"]),
        "--rigid_lengths",
        _vec_csv(solver["rigid_lengths_m"]),
        "--L_protrude",
        f"{float(initial_x_m):.12g}",
        "--x0",
        f"{float(initial_x_m):.12g}",
        "--x_min",
        f"{float(solver['L_protrude_min_m']):.12g}",
        "--x_max",
        f"{x_max:.12g}",
        "--x_ref",
        f"{float(initial_x_m):.12g}",
        "--L1_min",
        f"{float(defaults['L1_min_m']):.12g}",
        "--M_list",
        _vec_csv(solver["M_list_default"]),
        "--flex_d_outer",
        f"{float(flex['diameter_m']):.12g}",
        "--flex_E",
        f"{float(flex['E_Pa']):.12g}",
        "--flex_G",
        f"{float(flex['G_Pa']):.12g}",
        "--flex_rho",
        f"{float(flex['density_kg_m3']):.12g}",
        "--rigid_d_outer",
        f"{float(magnet['diameter_m']):.12g}",
        "--rigid_rho",
        f"{float(magnet['density_kg_m3']):.12g}",
        "--p0",
        _vec_csv(defaults["p0_dim_default_m"]),
        "--Q0",
        _vec_csv(defaults["Q0_wxyz_default"]),
        "--axis_body",
        _vec_csv(defaults["axis_body_default"]),
        "--enable_magnetics",
        "--calib_file",
        str(REPO / actuation["default_calib_file"]),
        "--m_body_list",
        _m_body_csv(solver["m_body_list_A_m2"]),
        "--max_iter",
        str(int(defaults["max_iter_default"])),
        "--tol",
        f"{float(defaults['tol_default']):.12g}",
        "--lm_damping",
        "1e-1",
        "--jac_method",
        "fwd",
        "--p_des",
        _vec_csv(target["p_des_m"]),
        "--eps_p",
        f"{float(eps_p_m):.12g}",
        "--sigma_p",
        f"{float(eps_p_m):.12g}",
        "--I_max",
        f"{float(actuation['current_limit_A_recommended']):.12g}",
        f"--I0={_vec_csv(initial_I_A)}",
        "--outer_max_iter",
        str(int(outer_max_iter)),
        "--step_I_max",
        "5.0",
        "--step_x_max",
        "0.02",
        "--out_dir",
        str(phase_dir),
        "--run_name",
        run_name,
        "--runtime_profile",
        "--print_devices",
        "--seed",
        str(20260705 + 1000 * (1 if catheter_key == "A1" else 2) + int(target["index"])),
    ]
    if enable_gravity:
        insert_at = args.index("--enable_magnetics")
        args[insert_at:insert_at] = [
            "--enable_gravity",
            "--g_world",
            _vec_csv(shared["gravity"]["g_world_m_s2"]),
        ]
    return args


def _plot_point_artifacts(point_dir: Path, target: dict[str, Any], result: dict[str, Any], history: dict[str, Any]) -> dict[str, str]:
    font = _font()
    out: dict[str, str] = {}
    outer = list(history.get("outer", []))
    point_dir.mkdir(parents=True, exist_ok=True)

    if outer:
        threshold_mm = float(result.get("sigma_p", 0.003)) * 1000.0
        iters = np.asarray([int(item.get("iter", i)) for i, item in enumerate(outer)], dtype=float)
        err_true = np.asarray([float(item.get("err_true_mm", item.get("err_mm", math.nan))) for item in outer], dtype=float)
        x_m = np.asarray([float(item.get("x_m", math.nan)) for item in outer], dtype=float)
        currents = np.asarray([item.get("I_A", [math.nan] * 8) for item in outer], dtype=float)

        fig, axes = plt.subplots(3, 1, figsize=(8.6, 9.2), dpi=170, sharex=True)
        axes[0].plot(iters, err_true, color="#cc3d00", lw=2.0)
        axes[0].axhline(threshold_mm, color="#145cff", ls="-.", lw=1.2, label=f"{threshold_mm:.1f} mm 阈值")
        axes[0].axhline(3.0, color="#168a2f", ls="--", lw=1.2, label="3 mm 阈值")
        axes[0].axhline(10.0, color="#808080", ls=":", lw=1.2, label="10 mm 参考")
        axes[0].set_ylabel("真实末端误差 (mm)", fontproperties=font)
        axes[0].legend(prop=font)
        for j in range(currents.shape[1]):
            axes[1].plot(iters, currents[:, j], lw=1.0, alpha=0.85)
        axes[1].set_ylabel("线圈电流 (A)", fontproperties=font)
        axes[2].plot(iters, x_m * 1000.0, color="#145cff", lw=2.0)
        axes[2].set_ylabel("推进量 (mm)", fontproperties=font)
        axes[2].set_xlabel("外层迭代步", fontproperties=font)
        fig.suptitle("收敛诊断", fontproperties=font, fontsize=13)
        fig.tight_layout()
        diag = point_dir / "diagnostics.png"
        fig.savefig(diag, bbox_inches="tight")
        fig.savefig(diag.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)
        out["diagnostics_png"] = str(diag.resolve())

    fig = plt.figure(figsize=(8.0, 6.8), dpi=170)
    ax = fig.add_subplot(111, projection="3d")
    accepted = [item for item in outer if bool(item.get("accept", False)) and item.get("pose", {}).get("centerline_npy")]
    if len(accepted) > 8:
        idx = np.linspace(0, len(accepted) - 1, 8, dtype=int)
        accepted = [accepted[int(i)] for i in idx]
    for item in accepted:
        rel = item.get("pose", {}).get("centerline_npy")
        cl_path = point_dir / rel if rel else None
        if cl_path and cl_path.exists():
            cl = np.load(cl_path)
            cl_s = _smooth_centerline(cl)
            alpha = 0.35 + 0.55 * (int(item.get("iter", 0)) / max(1, int(outer[-1].get("iter", 1))))
            ax.plot(cl_s[:, 0] * 1000.0, cl_s[:, 1] * 1000.0, cl_s[:, 2] * 1000.0, lw=1.8, alpha=alpha)
    final_cl = _centerline_path(point_dir)
    if final_cl is not None:
        cl = np.load(final_cl)
        cl_s = _smooth_centerline(cl)
        ax.plot(cl_s[:, 0] * 1000.0, cl_s[:, 1] * 1000.0, cl_s[:, 2] * 1000.0, color="#202020", lw=2.8, label="最终姿态")
    target_p = np.asarray(target["p_des_m"], dtype=float) * 1000.0
    ax.scatter([target_p[0]], [target_p[1]], [target_p[2]], marker="*", s=110, color="#145cff", label="目标点")
    if result.get("p_tip_true"):
        p_true = np.asarray(result["p_tip_true"], dtype=float) * 1000.0
        color = "#168a2f" if float(result.get("err_true_mm", math.inf)) <= 3.0 else "#d92525"
        ax.scatter([p_true[0]], [p_true[1]], [p_true[2]], s=58, color=color, label="真实末端")
    ax.set_xlabel("x (mm)", fontproperties=font)
    ax.set_ylabel("y (mm)", fontproperties=font)
    ax.set_zlabel("z (mm)", fontproperties=font)
    ax.set_title("三维姿态演化", fontproperties=font, fontsize=13)
    ax.legend(prop=font)
    ax.view_init(elev=24, azim=-58)
    ax.grid(True)
    fig.tight_layout()
    posture = point_dir / "posture_evolution.png"
    fig.savefig(posture, bbox_inches="tight")
    fig.savefig(posture.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    out["posture_evolution_png"] = str(posture.resolve())
    return out


def _run_one_target(
    *,
    catheter_key: str,
    cfg: dict[str, Any],
    target: dict[str, Any],
    phase_dir: Path,
    phase: str,
    initial_I_A: list[float] | None,
    initial_x_m: float,
    x_search_max_m: float,
    enable_gravity: bool,
    warm_start_source: str,
    outer_max_iter: int,
    per_point_timeout_s: float,
    eps_p_m: float,
) -> dict[str, Any]:
    run_name = f"pt_{int(target['index']):02d}"
    log_path = phase_dir / f"{run_name}.log"
    args = _build_ik_args(
        cfg=cfg,
        catheter_key=catheter_key,
        target=target,
        phase_dir=phase_dir,
        run_name=run_name,
        initial_I_A=initial_I_A,
        initial_x_m=initial_x_m,
        x_search_max_m=x_search_max_m,
        enable_gravity=enable_gravity,
        outer_max_iter=outer_max_iter,
        eps_p_m=eps_p_m,
    )
    phase_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    timed_out = False
    return_code: int | None = None
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND_ARGS:\n")
        log_file.write(json.dumps(args, ensure_ascii=False, indent=2))
        log_file.write("\n\nSTDOUT_STDERR:\n")
        log_file.flush()
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "inverse_kinematics.ik_multisite_lm_opt", *args],
                cwd=REPO,
                env={**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "JAX_PLATFORMS": "cuda"},
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=float(per_point_timeout_s),
                check=False,
            )
            return_code = int(completed.returncode)
        except subprocess.TimeoutExpired:
            timed_out = True
            return_code = None
            log_file.write(f"\n[AUTO_TIMEOUT] exceeded {float(per_point_timeout_s):.1f} s\n")
    wall_s = float(time.perf_counter() - t0)

    point_dir = phase_dir / run_name
    result = _read_json(point_dir / "result.json", {})
    history = _read_json(point_dir / "history.json", {})
    run_config = _read_json(point_dir / "run_config.json", {})
    accepted_count, reject_count, outer_iter_count = _history_counts(history)
    p_des = np.asarray(target["p_des_m"], dtype=float)
    p_flex = np.asarray(result.get("p_tip_flex", result.get("p_tip", [math.nan] * 3)), dtype=float)
    p_true = np.asarray(result.get("p_tip_true", [math.nan] * 3), dtype=float)
    err_flex_mm = float(np.linalg.norm(p_flex - p_des) * 1000.0)
    err_true_mm = float(np.linalg.norm(p_true - p_des) * 1000.0)
    threshold_mm = float(eps_p_m) * 1000.0
    success_eps = bool((not timed_out) and np.isfinite(err_true_mm) and err_true_mm <= threshold_mm)
    success_3mm = bool((not timed_out) and np.isfinite(err_true_mm) and err_true_mm <= 3.0)
    success_5mm = bool((not timed_out) and np.isfinite(err_true_mm) and err_true_mm <= 5.0)
    success_10mm = bool((not timed_out) and np.isfinite(err_true_mm) and err_true_mm <= 10.0)
    artifacts = _plot_point_artifacts(point_dir, target, result, history)
    centerline = _centerline_path(point_dir)
    I_A = result.get("I")
    x_m = result.get("x")
    x_bounds = run_config.get("x_bounds", [math.nan, math.nan])
    summary = {
        "catheter_type": catheter_key,
        "phase": phase,
        "target_index": int(target["index"]),
        "theta_rad": float(target["theta_rad"]),
        "p_des": [float(v) for v in p_des.tolist()],
        "p_tip_flex": [float(v) for v in p_flex.tolist()],
        "p_tip_true": [float(v) for v in p_true.tolist()],
        "err_flex_mm": err_flex_mm,
        "err_true_mm": err_true_mm,
        "best_err_true_mm": _best_error(history, "err_true_mm"),
        "success_eps": success_eps,
        "success_threshold_mm": threshold_mm,
        "success_3mm": success_3mm,
        "success_5mm": success_5mm,
        "success_10mm": success_10mm,
        "I_A": I_A,
        "x_m": x_m,
        "normE": result.get("normE"),
        "cls": result.get("cls"),
        "physical_status_good": str(result.get("cls", "")) == "GOOD",
        "outer_iter_count": int(outer_iter_count),
        "accept_count": int(accepted_count),
        "reject_count": int(reject_count),
        "wall_s": wall_s,
        "stop_reason": _infer_stop_reason(result=result, history=history, outer_max_iter=outer_max_iter, threshold_mm=threshold_mm),
        "warm_start_source": warm_start_source,
        "timed_out": bool(timed_out),
        "subprocess_return_code": return_code,
        "run_dir": str(point_dir.resolve()),
        "log_path": str(log_path.resolve()),
        "centerline_npy": str(centerline.resolve()) if centerline is not None else None,
        "artifacts": artifacts,
        "x_bounds": x_bounds,
        "max_abs_current_A": float(np.nanmax(np.abs(np.asarray(I_A, dtype=float)))) if I_A is not None else math.nan,
    }
    if timed_out:
        summary["stop_reason"] = "auto_timeout"
    elif return_code not in (0, None):
        summary["stop_reason"] = "subprocess_failed"
    (point_dir / "formal_point_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "catheter_type",
        "phase",
        "target_index",
        "theta_rad",
        "err_true_mm",
        "err_flex_mm",
        "success_eps",
        "success_3mm",
        "success_5mm",
        "success_10mm",
        "cls",
        "normE",
        "outer_iter_count",
        "accept_count",
        "reject_count",
        "wall_s",
        "stop_reason",
        "warm_start_source",
        "x_m",
        "max_abs_current_A",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _plot_phase_overview(run_root: Path, rows: list[dict[str, Any]], targets: list[dict[str, Any]], name: str) -> str | None:
    if not rows:
        return None
    font = _font()
    fig = plt.figure(figsize=(9.0, 7.4), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    target_arr = np.asarray([t["p_des_m"] for t in targets], dtype=float) * 1000.0
    ax.plot(target_arr[:, 0], target_arr[:, 1], target_arr[:, 2], color="#808080", ls="--", lw=1.3, label="目标圆")
    ax.scatter(target_arr[:, 0], target_arr[:, 1], target_arr[:, 2], marker="*", s=55, color="#145cff", label="目标点")
    for row in rows:
        p = np.asarray(row.get("p_tip_true", [math.nan] * 3), dtype=float) * 1000.0
        if not np.all(np.isfinite(p)):
            continue
        color = "#168a2f" if row.get("success_3mm") else ("#dd8a00" if row.get("success_10mm") else "#d92525")
        ax.scatter([p[0]], [p[1]], [p[2]], s=48, color=color)
        ax.plot([row["p_des"][0] * 1000.0, p[0]], [row["p_des"][1] * 1000.0, p[1]], [row["p_des"][2] * 1000.0, p[2]], color=color, lw=0.8, alpha=0.55)
        ax.text(p[0], p[1], p[2], str(row["target_index"]), fontproperties=font, fontsize=8, color=color)
    ax.set_xlabel("x (mm)", fontproperties=font)
    ax.set_ylabel("y (mm)", fontproperties=font)
    ax.set_zlabel("z (mm)", fontproperties=font)
    if name.startswith("A1_"):
        title = "A1 真实末端落点与目标圆总览"
    elif name.startswith("A2_"):
        title = "A2 真实末端落点与目标圆总览"
    else:
        title = "真实末端落点与目标圆总览"
    ax.set_title(title, fontproperties=font, fontsize=13)
    ax.legend(prop=font)
    ax.view_init(elev=24, azim=-58)
    ax.grid(True)
    fig.tight_layout()
    out = run_root / f"{name}.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return str(out.resolve())


def _run_catheter(
    *,
    run_root: Path,
    config: dict[str, Any],
    catheter_key: str,
    targets: list[dict[str, Any]],
    gate_indices: list[int],
    diag_indices: list[int],
    outer_max_iter: int,
    per_point_timeout_s: float,
    x_search_max_m: float,
    enable_gravity: bool,
    force_formal_20: bool,
    eps_p_m: float,
) -> dict[str, Any]:
    cfg = _catheter_solver_cfg(config, catheter_key)
    catheter_root = run_root / catheter_key
    catheter_root.mkdir(parents=True, exist_ok=True)
    initial_x_m = 0.103
    rows: list[dict[str, Any]] = []

    gate_rows: list[dict[str, Any]] = []
    if not force_formal_20:
        for idx in gate_indices:
            row = _run_one_target(
                catheter_key=catheter_key,
                cfg=cfg,
                target=targets[idx],
                phase_dir=catheter_root / "gate_4pt",
                phase="gate_4pt",
                initial_I_A=None,
                initial_x_m=initial_x_m,
                x_search_max_m=x_search_max_m,
                enable_gravity=enable_gravity,
                warm_start_source="cold_start_gate",
                outer_max_iter=outer_max_iter,
                per_point_timeout_s=per_point_timeout_s,
                eps_p_m=eps_p_m,
            )
            gate_rows.append(row)
            rows.append(row)
            print(
                f"{catheter_key} gate pt_{idx:02d}: true_err={row['err_true_mm']:.3f} mm "
                f"eps={row['success_eps']} cls={row['cls']} wall={row['wall_s']:.1f}s",
                flush=True,
            )

    gate_success = sum(1 for row in gate_rows if row["success_eps"])
    formal_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    decision = "formal_20pt_forced" if force_formal_20 else ("formal_20pt" if gate_success >= 3 else "diagnostic_8pt")

    if decision in ("formal_20pt", "formal_20pt_forced"):
        warm_I: list[float] | None = None
        warm_x = initial_x_m
        warm_source = "cold_start_formal"
        for target in targets:
            row = _run_one_target(
                catheter_key=catheter_key,
                cfg=cfg,
                target=target,
                phase_dir=catheter_root / "formal_20pt",
                phase="formal_20pt",
                initial_I_A=warm_I,
                initial_x_m=warm_x,
                x_search_max_m=x_search_max_m,
                enable_gravity=enable_gravity,
                warm_start_source=warm_source,
                outer_max_iter=outer_max_iter,
                per_point_timeout_s=per_point_timeout_s,
                eps_p_m=eps_p_m,
            )
            formal_rows.append(row)
            rows.append(row)
            print(
                f"{catheter_key} formal pt_{row['target_index']:02d}: true_err={row['err_true_mm']:.3f} mm "
                f"eps={row['success_eps']} cls={row['cls']} wall={row['wall_s']:.1f}s",
                flush=True,
            )
            if row["success_eps"] and row.get("I_A") is not None and row.get("x_m") is not None:
                warm_I = [float(v) for v in row["I_A"]]
                warm_x = float(row["x_m"])
                warm_source = f"last_success_pt_{row['target_index']:02d}"
            elif warm_I is None:
                warm_source = "cold_start_after_failure"
            else:
                warm_source = "last_success_before_failed_point"
    else:
        for idx in diag_indices:
            row = _run_one_target(
                catheter_key=catheter_key,
                cfg=cfg,
                target=targets[idx],
                phase_dir=catheter_root / "diagnostic_8pt",
                phase="diagnostic_8pt",
                initial_I_A=None,
                initial_x_m=initial_x_m,
                x_search_max_m=x_search_max_m,
                enable_gravity=enable_gravity,
                warm_start_source="cold_start_diagnostic",
                outer_max_iter=outer_max_iter,
                per_point_timeout_s=per_point_timeout_s,
                eps_p_m=eps_p_m,
            )
            diag_rows.append(row)
            rows.append(row)
            print(
                f"{catheter_key} diagnostic pt_{idx:02d}: true_err={row['err_true_mm']:.3f} mm "
                f"eps={row['success_eps']} cls={row['cls']} wall={row['wall_s']:.1f}s",
                flush=True,
            )

    _write_rows_csv(catheter_root / "all_points_summary.csv", rows)
    overview_rows = formal_rows if formal_rows else diag_rows if diag_rows else gate_rows
    overview_plot = _plot_phase_overview(run_root, overview_rows, targets, f"{catheter_key}_target_circle_true_tip_overview")
    return {
        "catheter_type": catheter_key,
        "decision": decision,
        "gate_success_count_eps": int(gate_success),
        "gate_success_count_3mm": int(sum(1 for row in gate_rows if row["success_3mm"])),
        "gate_success_count_5mm": int(sum(1 for row in gate_rows if row["success_5mm"])),
        "gate_total": int(len(gate_rows)),
        "gate_rows": gate_rows,
        "formal_rows": formal_rows,
        "diagnostic_rows": diag_rows,
        "all_rows_csv": str((catheter_root / "all_points_summary.csv").resolve()),
        "overview_plot": overview_plot,
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "success_count_eps": 0,
            "success_count_3mm": 0,
            "success_count_5mm": 0,
            "success_count_10mm": 0,
            "err_true_mm_mean": math.nan,
            "err_true_mm_max": math.nan,
            "wall_s_total": 0.0,
        }
    errs = np.asarray([row["err_true_mm"] for row in rows], dtype=float)
    return {
        "count": int(len(rows)),
        "success_count_eps": int(sum(1 for row in rows if row["success_eps"])),
        "success_count_3mm": int(sum(1 for row in rows if row["success_3mm"])),
        "success_count_5mm": int(sum(1 for row in rows if row["success_5mm"])),
        "success_count_10mm": int(sum(1 for row in rows if row["success_10mm"])),
        "err_true_mm_mean": float(np.nanmean(errs)),
        "err_true_mm_max": float(np.nanmax(errs)),
        "err_true_mm_min": float(np.nanmin(errs)),
        "wall_s_total": float(sum(float(row.get("wall_s", 0.0)) for row in rows)),
    }


def _write_report(report_path: Path, run_root: Path, metadata: dict[str, Any], results: dict[str, Any]) -> None:
    lines: list[str] = []
    threshold_mm = float(metadata["eps_p_m"]) * 1000.0
    lines.append(f"# A1/A2 单点磁化导管 {threshold_mm:.1f} mm 有监督 IK 正式实验报告")
    lines.append("")
    lines.append(f"- 实验目录：`{run_root}`")
    lines.append(f"- 目标圆：x = {metadata['target_center_m'][0] * 1000:.1f} mm，半径 = {metadata['target_radius_m'] * 1000:.1f} mm，20 点均匀采样。")
    lines.append("- 约束口径：只约束真实刚性远端末端位置，不约束末端姿态。")
    lines.append(f"- 成功阈值：真实末端位置误差不超过 {threshold_mm:.1f} mm；同时保留 3 mm、5 mm、10 mm 参考统计。")
    lines.append(f"- git 版本：`{metadata['git_commit']}`")
    lines.append("")
    lines.append("## 实验设置")
    lines.append("")
    gravity_text = "开启重力" if metadata["enable_gravity"] else "关闭重力"
    lines.append(f"- 数值设置：M=5，{gravity_text}，FK 容差 1e-5，电流限幅 20 A。")
    lines.append(f"- 推进搜索：本批次使用 0.003001 m 到 {metadata['x_search_max_m']:.3f} m 的局部搜索范围；配置文件中的物理长度上限仍保留为 1 m。")
    lines.append(f"- 自动超时：单个目标点超过 {metadata['per_point_timeout_s'] / 60.0:.1f} 分钟未自然退出时，标记为 auto_timeout，不计入成功。")
    lines.append("- A1：末端单磁体沿局部 z 轴轴向磁化。")
    lines.append("- A2：末端单磁体沿局部 x 轴径向磁化。")
    if metadata.get("force_formal_20"):
        lines.append("- 运行规则：本轮按用户要求强制执行完整 20 点正式圆周实验，不使用门控结果阻断。")
    else:
        lines.append(f"- 门控规则：上、下、左、右 4 个代表点中至少 3 点达到 {threshold_mm:.1f} mm，才进入该导管 20 点正式实验；否则执行 8 点可达性诊断。")
    lines.append("")
    lines.append("## 总览结果")
    lines.append("")
    lines.append(f"| 导管 | 门控阈值成功数 | 决策 | 正式/诊断点数 | {threshold_mm:.1f}mm 成功数 | 3mm 成功数 | 5mm 成功数 | 10mm 成功数 | 最大真实误差(mm) | 平均真实误差(mm) | 总耗时(s) |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    catheter_keys = list(metadata.get("catheter_keys", ["A1", "A2"]))
    for key in catheter_keys:
        item = results[key]
        rows = item["formal_rows"] if item["formal_rows"] else item["diagnostic_rows"]
        stats = _summarize_rows(rows)
        lines.append(
            f"| {key} | {item['gate_success_count_eps']}/{item['gate_total']} | {item['decision']} | "
            f"{stats['count']} | {stats['success_count_eps']} | {stats['success_count_3mm']} | "
            f"{stats['success_count_5mm']} | {stats['success_count_10mm']} | "
            f"{stats['err_true_mm_max']:.3f} | {stats['err_true_mm_mean']:.3f} | {stats['wall_s_total']:.1f} |"
        )
    lines.append("")
    lines.append("## 分导管结果")
    lines.append("")
    for key in catheter_keys:
        item = results[key]
        lines.append(f"### {key}")
        lines.append("")
        if metadata.get("force_formal_20"):
            lines.append("- 门控结果：本轮强制完整 20 点，未使用门控阻断。")
        else:
            lines.append(f"- 门控结果：{item['gate_success_count_eps']}/{item['gate_total']} 点达到 {threshold_mm:.1f} mm。")
        if item["decision"] in ("formal_20pt", "formal_20pt_forced", "formal_20pt_forced_resumed"):
            lines.append("- 决策：已执行 20 点正式圆周实验。")
            rows = item["formal_rows"]
        else:
            lines.append("- 决策：门控未通过，未执行 20 点正式实验，已执行 8 点可达性诊断。")
            rows = item["diagnostic_rows"]
        if item.get("overview_plot"):
            lines.append(f"- 总览图：`{item['overview_plot']}`")
        lines.append("")
        lines.append(f"| 点号 | 阶段 | 真实误差(mm) | 柔性节点误差(mm) | {threshold_mm:.1f}mm | 3mm | 5mm | 10mm | cls | 外层迭代 | 耗时(s) | 停止原因 |")
        lines.append("|---:|---|---:|---:|---|---|---|---|---|---:|---:|---|")
        for row in rows:
            lines.append(
                f"| {row['target_index']} | {row['phase']} | {row['err_true_mm']:.3f} | "
                f"{row['err_flex_mm']:.3f} | {row['success_eps']} | {row['success_3mm']} | "
                f"{row['success_5mm']} | {row['success_10mm']} | "
                f"{row['cls']} | {row['outer_iter_count']} | {row['wall_s']:.1f} | {row['stop_reason']} |"
            )
        lines.append("")
    lines.append("## 输出索引")
    lines.append("")
    lines.append(f"- 批次元数据：`{run_root / 'batch_metadata.json'}`")
    lines.append(f"- 汇总 JSON：`{run_root / 'summary.json'}`")
    lines.append(f"- 汇总 CSV：`{run_root / 'summary.csv'}`")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _write_output_index(run_root: Path, metadata: dict[str, Any], results: dict[str, Any]) -> None:
    """Write a concise artifact index for the delivery folder."""
    threshold_mm = float(metadata["eps_p_m"]) * 1000.0
    lines = [
        "# 输出索引",
        "",
        f"本目录为 A1/A2 单点磁化导管 {threshold_mm:.1f} mm 有监督 IK 实验输出。",
        "",
        "## 核心文件",
        "",
        "- `batch_metadata.json`：实验版本、目标圆、阈值、配置文件与标定文件哈希。",
        "- `targets.json`：20 个圆周目标点坐标。",
        "- `summary.json`：完整机器可读汇总结果。",
        "- `summary.csv`：全部运行点的表格汇总。",
        "- `report.md`：本目录内的中文实验报告副本。",
        "- `commands.sh`：复现实验的启动命令。",
        "",
        "## 总览图",
        "",
        "- `A1_target_circle_true_tip_overview.png`：A1 真实末端落点与目标圆总览。",
        "- `A2_target_circle_true_tip_overview.png`：A2 真实末端落点与目标圆总览。",
        "",
        "## 分点输出",
        "",
        "每个点目录均包含 `result.json`、`history.json`、`runtime_profile.json`、`formal_point_summary.json`、`diagnostics.png`、`posture_evolution.png` 和 `poses/` 中心线数据。",
        "",
        "## 结论索引",
        "",
    ]
    for key in metadata.get("catheter_keys", ["A1", "A2"]):
        item = results[key]
        rows = item["formal_rows"] if item["formal_rows"] else item["diagnostic_rows"]
        stats = _summarize_rows(rows)
        lines.append(
            f"- {key}：决策 `{item['decision']}`，运行点数 {stats['count']}，"
            f"{threshold_mm:.1f} mm 成功 {stats['success_count_eps']}，"
            f"最大误差 {stats['err_true_mm_max']:.3f} mm，平均误差 {stats['err_true_mm_mean']:.3f} mm。"
        )
    (run_root / "OUTPUT_INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    """Build the formal experiment command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    gravity = parser.add_mutually_exclusive_group()
    gravity.add_argument("--enable-gravity", dest="enable_gravity", action="store_true", help="Run with gravity enabled.")
    gravity.add_argument("--disable-gravity", dest="enable_gravity", action="store_false", help="Run with gravity disabled.")
    parser.set_defaults(enable_gravity=True)
    parser.add_argument("--eps-p-mm", type=float, default=3.0, help="Position success threshold in millimeters.")
    parser.add_argument("--force-formal-20", action="store_true", help="Run all 20 formal circle targets without gate blocking.")
    parser.add_argument("--catheters", type=str, default="A1,A2", help="Comma-separated catheter keys to run.")
    parser.add_argument("--target-indices", type=str, default=None, help="Comma-separated target indices. Defaults to all formal targets when forced.")
    parser.add_argument("--outer-max-iter", type=int, default=50, help="Outer IK iteration limit.")
    parser.add_argument("--per-point-timeout-s", type=float, default=2700.0, help="Per-target timeout in seconds.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    enable_gravity = bool(args.enable_gravity)
    eps_p_m = float(args.eps_p_mm) / 1000.0
    threshold_tag = f"{float(args.eps_p_mm):g}mm".replace(".", "p")
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    gravity_suffix = "gravity" if enable_gravity else "no_gravity"
    run_root = OUT_ROOT / f"{run_id}_A1_A2_{threshold_tag}_supervised_IK_{gravity_suffix}"
    run_root.mkdir(parents=True, exist_ok=True)
    report_path = OUT_ROOT / f"{run_id}_A1_A2_{threshold_tag}_supervised_IK_{gravity_suffix}_report.md"
    config = _load_config()
    center_m = (0.06, 0.0, 0.0)
    radius_m = 0.1
    targets = _circle_targets(n_points=20, center_m=center_m, radius_m=radius_m)
    if args.target_indices:
        requested_indices = [int(v.strip()) for v in str(args.target_indices).split(",") if v.strip()]
        target_by_index = {int(item["index"]): item for item in targets}
        targets = [target_by_index[idx] for idx in requested_indices]
    catheter_keys = [v.strip() for v in str(args.catheters).split(",") if v.strip()]
    for key in catheter_keys:
        if key not in config["catheter_types"]:
            raise ValueError(f"Unknown catheter key: {key}")
    gate_indices = [0, 5, 10, 15]
    diag_indices = [0, 2, 5, 7, 10, 12, 15, 17]
    outer_max_iter = int(args.outer_max_iter)
    per_point_timeout_s = float(args.per_point_timeout_s)
    x_search_max_m = 0.203
    calib_path = REPO / config["shared_physical_parameters"]["actuation"]["default_calib_file"]
    metadata = {
        "run_id": run_id,
        "experiment_name": f"A1_A2_{threshold_tag}_supervised_IK_{gravity_suffix}",
        "run_root": str(run_root.resolve()),
        "report_path": str(report_path.resolve()),
        "git_commit": _git_commit(),
        "config_path": str(CONFIG_PATH.resolve()),
        "config_sha256": _sha256(CONFIG_PATH),
        "calib_file": str(calib_path.resolve()),
        "calib_sha256": _sha256(calib_path),
        "target_center_m": list(center_m),
        "target_radius_m": radius_m,
        "target_count": len(targets),
        "catheter_keys": catheter_keys,
        "requested_target_indices": [int(item["index"]) for item in targets],
        "gate_indices": gate_indices,
        "diagnostic_indices": diag_indices,
        "outer_max_iter": outer_max_iter,
        "per_point_timeout_s": per_point_timeout_s,
        "x_search_max_m": x_search_max_m,
        "x_search_note_zh": "本次 x=60 mm 圆周实验采用 0.203 m 局部推进搜索上限；物理配置文件仍保留 1 m 最大长度。",
        "eps_p_m": eps_p_m,
        "sigma_p_m": eps_p_m,
        "force_formal_20": bool(args.force_formal_20),
        "M": 5,
        "tol": 1e-5,
        "enable_gravity": enable_gravity,
        "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
    }
    (run_root / "batch_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_root / "targets.json").write_text(json.dumps(targets, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_root / "commands.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python "
        f"{Path(__file__).resolve()} "
        f"{'--enable-gravity' if enable_gravity else '--disable-gravity'} "
        f"--eps-p-mm {float(args.eps_p_mm):.12g} "
        f"--catheters {','.join(catheter_keys)} "
        f"--target-indices {','.join(str(int(item['index'])) for item in targets)} "
        f"--outer-max-iter {outer_max_iter} "
        f"--per-point-timeout-s {per_point_timeout_s:.12g} "
        f"{'--force-formal-20' if args.force_formal_20 else ''}\n",
        encoding="utf-8",
    )

    t0 = time.perf_counter()
    results = {}
    for catheter_key in catheter_keys:
        results[catheter_key] = _run_catheter(
            run_root=run_root,
            config=config,
            catheter_key=catheter_key,
            targets=targets,
            gate_indices=gate_indices,
            diag_indices=diag_indices,
            outer_max_iter=outer_max_iter,
            per_point_timeout_s=per_point_timeout_s,
            x_search_max_m=x_search_max_m,
            enable_gravity=enable_gravity,
            force_formal_20=bool(args.force_formal_20),
            eps_p_m=eps_p_m,
        )
    metadata["total_wall_s"] = float(time.perf_counter() - t0)

    all_rows: list[dict[str, Any]] = []
    for key in catheter_keys:
        all_rows.extend(results[key]["gate_rows"])
        all_rows.extend(results[key]["formal_rows"])
        all_rows.extend(results[key]["diagnostic_rows"])
    _write_rows_csv(run_root / "summary.csv", all_rows)
    summary = {"metadata": metadata, "results": results, "all_rows_csv": str((run_root / "summary.csv").resolve())}
    (run_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(report_path, run_root, metadata, results)
    (run_root / "report.md").write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    _write_output_index(run_root, metadata, results)
    print(f"summary: {run_root / 'summary.json'}", flush=True)
    print(f"report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
