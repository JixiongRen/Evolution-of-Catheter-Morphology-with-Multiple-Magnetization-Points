"""Run a small supervised IK circle demo for the A1 axial single-magnet catheter."""

from __future__ import annotations

import contextlib
import importlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "catheter_physical_configs.yaml"
OUT_ROOT = ROOT / "supervised_ik_demos"
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _font() -> FontProperties:
    if FONT_PATH.exists():
        return FontProperties(fname=str(FONT_PATH))
    return FontProperties()


def _vec_csv(values: list[float] | tuple[float, ...] | np.ndarray) -> str:
    return ",".join(f"{float(v):.12g}" for v in values)


def _smooth_centerline(points_m: np.ndarray, *, samples_per_segment: int = 24) -> np.ndarray:
    """Fit a smooth display curve through centerline nodes."""
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


def _load_a1_config() -> dict[str, Any]:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return {
        "shared": cfg["shared_physical_parameters"],
        "solver_defaults": cfg["solver_defaults"],
        "catheter": cfg["catheter_types"]["A1"],
    }


def _circle_targets(*, n_points: int, center_m: tuple[float, float, float], radius_m: float) -> list[dict[str, Any]]:
    """Generate uniformly spaced targets on the circle normal to the x-axis."""
    targets = []
    cx, cy, cz = center_m
    for i in range(n_points):
        theta = (i + 0.5) * 2.0 * math.pi / float(n_points)
        targets.append(
            {
                "index": i,
                "theta_rad": theta,
                "p_des_m": [
                    cx,
                    cy + radius_m * math.cos(theta),
                    cz + radius_m * math.sin(theta),
                ],
            }
        )
    return targets


def _run_one_target(
    *,
    target: dict[str, Any],
    run_root: Path,
    cfg: dict[str, Any],
    enable_gravity: bool = True,
    outer_max_iter: int = 20,
    initial_I_A: list[float] | None = None,
    initial_x_m: float = 0.103,
) -> dict[str, Any]:
    # Import lazily after the output directory is ready. The module sets JAX flags.
    ik_module = importlib.import_module("inverse_kinematics.ik_multisite_lm_opt")

    shared = cfg["shared"]
    catheter = cfg["catheter"]
    solver = catheter["solver_input"]
    magnet = shared["magnet"]
    flex = shared["flexible_body"]
    actuation = shared["actuation"]
    defaults = cfg["solver_defaults"]

    run_name = f"pt_{target['index']:02d}"
    p_des = target["p_des_m"]
    log_path = run_root / f"{run_name}.log"

    x0 = float(initial_x_m)
    demo_m_list = solver["M_list_default"]
    initial_I_A = initial_I_A if initial_I_A is not None else [0.0] * 8
    args = [
        "--flex_lengths",
        _vec_csv(solver["cli_flex_lengths_placeholder_m"]),
        "--rigid_lengths",
        _vec_csv(solver["rigid_lengths_m"]),
        "--L_protrude",
        f"{x0:.12g}",
        "--x0",
        f"{x0:.12g}",
        "--x_min",
        "0.003001",
        "--x_max",
        "0.203",
        "--x_ref",
        f"{x0:.12g}",
        "--L1_min",
        f"{defaults['L1_min_m']:.12g}",
        "--M_list",
        _vec_csv(demo_m_list),
        "--flex_d_outer",
        f"{flex['diameter_m']:.12g}",
        "--flex_E",
        f"{flex['E_Pa']:.12g}",
        "--flex_G",
        f"{flex['G_Pa']:.12g}",
        "--flex_rho",
        f"{flex['density_kg_m3']:.12g}",
        "--rigid_d_outer",
        f"{magnet['diameter_m']:.12g}",
        "--rigid_rho",
        f"{magnet['density_kg_m3']:.12g}",
        "--p0",
        "0,0,0",
        "--Q0",
        _vec_csv(defaults["Q0_wxyz_default"]),
        "--axis_body",
        _vec_csv(defaults["axis_body_default"]),
        "--enable_magnetics",
        "--calib_file",
        str(REPO / actuation["default_calib_file"]),
        "--m_body_list",
        _vec_csv(solver["m_body_list_A_m2"][0]),
        "--max_iter",
        f"{defaults['max_iter_default']}",
        "--tol",
        "1e-5",
        "--lm_damping",
        "1e-1",
        "--jac_method",
        "fwd",
        "--p_des",
        _vec_csv(p_des),
        "--eps_p",
        "0.01",
        "--sigma_p",
        "0.01",
        "--I_max",
        f"{actuation['current_limit_A_recommended']:.12g}",
        "--I0",
        _vec_csv(initial_I_A),
        "--outer_max_iter",
        str(int(outer_max_iter)),
        "--step_I_max",
        "5.0",
        "--step_x_max",
        "0.02",
        "--out_dir",
        str(run_root),
        "--run_name",
        run_name,
        "--runtime_profile",
        "--seed",
        str(20260705 + int(target["index"])),
    ]
    if enable_gravity:
        args.extend(["--enable_gravity", "--g_world", _vec_csv(shared["gravity"]["g_world_m_s2"])])

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND_ARGS:\n")
        log_file.write(json.dumps(args, ensure_ascii=False, indent=2))
        log_file.write("\n\nSTDOUT_STDERR:\n")
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            ik_module.main(args)

    result_path = run_root / run_name / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    run_config = json.loads((run_root / run_name / "run_config.json").read_text(encoding="utf-8"))
    final = run_config.get("final", {})
    centerline_rel = final.get("centerline_npy")
    centerline_path = (run_root / run_name / centerline_rel).resolve() if centerline_rel else None
    return {
        "target_index": int(target["index"]),
        "theta_rad": float(target["theta_rad"]),
        "p_des_m": p_des,
        "success": bool(result.get("success", False)),
        "err_m": float(result.get("err_m", math.nan)),
        "err_mm": float(result.get("err_mm", math.nan)),
        "p_tip_m": result.get("p_tip"),
        "I_A": result.get("I"),
        "x_m": result.get("x"),
        "normE": result.get("normE"),
        "cls": result.get("cls"),
        "run_dir": str((run_root / run_name).resolve()),
        "log_path": str(log_path.resolve()),
        "centerline_npy": str(centerline_path) if centerline_path else None,
    }


def _plot_summary(run_root: Path, summary: dict[str, Any]) -> Path:
    font = _font()
    fig = plt.figure(figsize=(9.2, 7.6), dpi=190)
    ax = fig.add_subplot(111, projection="3d")

    # Initial straight observation line. The IK target is the flexible distal node,
    # not the physical distal end of the rigid magnet.
    ax.plot([0, 100], [0, 0], [0, 0], color="#202020", lw=2.5, label="初始观测点路径")

    targets = np.asarray([r["p_des_m"] for r in summary["results"]], dtype=float) * 1000.0
    ax.plot(targets[:, 0], targets[:, 1], targets[:, 2], color="#999999", ls="--", lw=1.2, label="目标圆")
    ax.scatter(targets[:, 0], targets[:, 1], targets[:, 2], marker="*", s=88, color="#145cff", label="目标点")

    for r in summary["results"]:
        cl_path = r.get("centerline_npy")
        if cl_path and Path(cl_path).exists():
            cl = np.load(cl_path) * 1000.0
            cl_smooth = _smooth_centerline(cl / 1000.0) * 1000.0
            ax.plot(cl_smooth[:, 0], cl_smooth[:, 1], cl_smooth[:, 2], lw=1.9, alpha=0.9)
            ax.scatter(cl[:, 0], cl[:, 1], cl[:, 2], s=9, color="#202020", alpha=0.32)
        if r.get("p_tip_m") is not None:
            p = np.asarray(r["p_tip_m"], dtype=float) * 1000.0
            color = "#168a2f" if r.get("success") else "#d92525"
            ax.scatter([p[0]], [p[1]], [p[2]], s=45, color=color)
            ax.text(p[0], p[1], p[2] + 5.0, f"{r['target_index']}", fontproperties=font, fontsize=8, color=color)

    ax.set_xlabel("x (mm)", fontproperties=font)
    ax.set_ylabel("y (mm)", fontproperties=font)
    ax.set_zlabel("z (mm)", fontproperties=font)
    ax.set_title("A1 单点轴向磁化导管：垂直 x 轴圆周 IK demo", fontproperties=font, fontsize=14)
    ax.legend(prop=font, loc="upper left")
    ax.view_init(elev=24, azim=-58)
    ax.grid(True)
    fig.tight_layout()
    out = run_root / "a1_axial_circle_demo_summary.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    cfg = _load_a1_config()
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    run_root = OUT_ROOT / f"{run_id}_a1_axial_position_only_circle_demo"
    run_root.mkdir(parents=True, exist_ok=True)

    targets = _circle_targets(n_points=6, center_m=(0.1, 0.0, 0.0), radius_m=0.1)
    summary = {
        "demo_name": "a1_axial_position_only_circle_demo",
        "catheter_type": "A1",
        "target_plane_zh": "y-z 平面，x=0.1 m，圆平面法向为 x 轴",
        "target_center_m": [0.1, 0.0, 0.0],
        "target_radius_m": 0.1,
        "target_count": len(targets),
        "note_zh": "当前双层 IK 主线只约束 tip_p_dim，即最后柔性段远端节点位置；不约束真实刚性磁体远端姿态。",
        "demo_numerics_zh": "本 demo 采用配置文件默认 M=5 并开启重力；IK 内部调用 forward_kinematics_optimized.fk_engine.ForwardKinematicsEngine。",
        "results": [],
    }

    for target in targets:
        result = _run_one_target(target=target, run_root=run_root, cfg=cfg)
        summary["results"].append(result)
        print(
            f"pt_{result['target_index']:02d}: success={result['success']} "
            f"err_mm={result['err_mm']:.3f} cls={result['cls']} x={result['x_m']:.5f}"
        )

    success_count = sum(1 for r in summary["results"] if r["success"])
    errors = [r["err_mm"] for r in summary["results"]]
    summary["success_count"] = success_count
    summary["err_mm_mean"] = float(np.nanmean(errors))
    summary["err_mm_max"] = float(np.nanmax(errors))
    plot_path = _plot_summary(run_root, summary)
    summary["summary_plot"] = str(plot_path.resolve())
    (run_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary: {run_root / 'summary.json'}")
    print(f"plot: {plot_path}")


if __name__ == "__main__":
    main()
