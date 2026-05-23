from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


@dataclass(frozen=True)
class CaseConfig:
    """Configuration for one FK regression case."""

    case_id: str
    M_list: tuple[int, int, int]
    L_protrude: float
    enable_gravity: bool
    enable_magnetics: bool
    coil_currents: tuple[float, ...]
    m_body_list: Optional[tuple[tuple[float, float, float], ...]] = None


def _case_set() -> list[CaseConfig]:
    """Return the representative dense-FK regression cases."""
    mixed_m = (
        (0.0, 0.0, -0.005301),
        (0.0, 0.0, 0.005301),
        (0.005301, 0.0, 0.0),
    )
    return [
        CaseConfig("straight_nomag_nogravity", (3, 2, 2), 0.10, False, False, (0.0,) * 8),
        CaseConfig("gravity_only_mid", (3, 2, 2), 0.10, True, False, (0.0,) * 8),
        CaseConfig("mag_zero_mid", (3, 2, 2), 0.10, False, True, (0.0,) * 8, mixed_m),
        CaseConfig(
            "coil5_pos",
            (3, 2, 2),
            0.10,
            False,
            True,
            (0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 0.0, 0.0),
            mixed_m,
        ),
        CaseConfig(
            "mixed_balanced",
            (3, 2, 2),
            0.10,
            False,
            True,
            (10.0, -8.0, 6.0, -4.0, 2.0, -6.0, 8.0, -10.0),
            mixed_m,
        ),
    ]


def _select_cases(case_ids: str) -> list[CaseConfig]:
    cases = _case_set()
    if not case_ids.strip():
        return cases
    requested = {item.strip() for item in case_ids.split(",") if item.strip()}
    return [case for case in cases if case.case_id in requested]


def _repo_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _run_cases(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if str(repo_root) in sys.path:
        sys.path.remove(str(repo_root))
    sys.path.insert(0, str(repo_root))

    import jax
    import jax.numpy as jnp

    from single_layer_ik.forward_kinematics_optimized_bak.equilibrium_solver_nondim import residual_bar
    from single_layer_ik.forward_kinematics_optimized_bak.fk_engine import ForwardKinematicsEngine
    from single_layer_ik.forward_kinematics_optimized_bak.nondim import x_bar_to_dim
    from single_layer_ik.forward_kinematics_optimized_bak.utils_nondim import unpack_z_bar_jax
    from single_layer_ik.joint_problem import (
        extract_centerline_p_dim,
        extract_interface_position_gaps_dim,
        extract_shape_smooth_residual,
    )

    def _to_list(value: Any) -> list[float]:
        return [float(v) for v in np.asarray(jax.device_get(value)).reshape(-1).tolist()]

    def _build_engine(case: CaseConfig) -> Any:
        return ForwardKinematicsEngine(
            flex_lengths_tail=(0.03, 0.03),
            rigid_lengths=(0.003, 0.003, 0.003),
            M_list=case.M_list,
            L1_min=1e-6,
            flex_d_outer=(0.0015,),
            flex_E=(1.8e6,),
            flex_G=(0.6e6,),
            flex_rho=(970.0,),
            rigid_d_outer=(0.0015,),
            rigid_rho=(7500.0,),
            p0_dim=(0.0, 0.0, -0.05),
            Q0_wxyz=(1.0, 0.0, 0.0, 0.0),
            axis_body=(1.0, 0.0, 0.0),
            enable_gravity=bool(case.enable_gravity),
            enable_magnetics=bool(case.enable_magnetics),
            calib_file=str(args.calib_file),
            actuation_table_pkl=str(args.actuation_table_pkl),
            m_body_list=case.m_body_list,
            max_iter=int(args.max_iter),
            tol=float(args.tol),
            lm_damping=float(args.lm_damping),
            jac_method=str(args.jac_method),
            L_protrude_max=float(args.L_protrude_max),
        )

    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": str(repo_root),
        "git_commit": _repo_commit(repo_root),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", ""),
        "max_iter": int(args.max_iter),
        "tol": float(args.tol),
        "case_ids": [case.case_id for case in _select_cases(str(args.case_ids))],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for case in _select_cases(str(args.case_ids)):
        start = time.perf_counter()
        engine = _build_engine(case)
        z_star, params, _meshes, ok, stats = engine.solve_with_stats(
            coil_currents=case.coil_currents,
            L_protrude=float(case.L_protrude),
            warm_start=False,
            return_stats=True,
            stats_include_jacobian=False,
            verbose=False,
        )
        if hasattr(z_star, "block_until_ready"):
            z_star.block_until_ready()
        wall_s = time.perf_counter() - start

        centerline = extract_centerline_p_dim(
            z_bar=z_star,
            scales=params.scales,
            M_list=params.M_list,
            include_rigid=True,
            rigid_sample_count=2,
        )
        centerline_np = np.asarray(jax.device_get(centerline), dtype=np.float64)
        centerline_file = f"{case.case_id}_centerline.npy"
        np.save(out_dir / centerline_file, centerline_np)

        gap = extract_interface_position_gaps_dim(z_bar=z_star, scales=params.scales, M_list=params.M_list)
        smooth = extract_shape_smooth_residual(
            z_bar=z_star,
            scales=params.scales,
            M_list=params.M_list,
            include_rigid=True,
            rigid_sample_count=2,
        )
        E = residual_bar(z_star, params)
        if hasattr(E, "block_until_ready"):
            E.block_until_ready()

        x_nodes_list, _k_list, x_rigid_list = unpack_z_bar_jax(z_star, M_list=params.M_list)
        flex_last = x_bar_to_dim(x_nodes_list[-1][-1], params.scales)
        rigid_last = x_bar_to_dim(x_rigid_list[-1], params.scales)

        row = {
            "case_id": case.case_id,
            "ok": bool(ok),
            "wall_s": float(wall_s),
            "stop_reason": None if stats is None else str(stats.stop_reason),
            "n_iter": None if stats is None else int(stats.n_iter),
            "normE": float(jnp.linalg.norm(E)),
            "max_abs_E": float(jnp.max(jnp.abs(E))),
            "M_list": list(case.M_list),
            "L_protrude": float(case.L_protrude),
            "L_ref": float(params.scales.L_ref),
            "L1_dim": float(params.L1_dim),
            "enable_gravity": bool(case.enable_gravity),
            "enable_magnetics": bool(case.enable_magnetics),
            "coil_currents": list(case.coil_currents),
            "m_body_list": None if case.m_body_list is None else [list(v) for v in case.m_body_list],
            "centerline_file": centerline_file,
            "centerline_shape": list(centerline_np.shape),
            "centerline_start_m": centerline_np[0].tolist(),
            "centerline_end_m": centerline_np[-1].tolist(),
            "last_flex_node_m": _to_list(flex_last[0:3]),
            "last_rigid_state_m": _to_list(rigid_last[0:3]),
            "interface_gap_max_m": float(np.max(np.linalg.norm(np.asarray(jax.device_get(gap)), axis=1))) if gap.size else 0.0,
            "smooth_norm": float(np.linalg.norm(np.asarray(jax.device_get(smooth)))) if smooth.size else 0.0,
        }
        rows.append(row)
        with (out_dir / "fk_pointwise_results.jsonl").open("w", encoding="utf-8") as handle:
            for item in rows:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return 0


def _pointwise_metrics(ref: np.ndarray, cand: np.ndarray, *, L_ref: float) -> dict[str, Any]:
    n = min(ref.shape[0], cand.shape[0])
    if n <= 0:
        return {"status": "missing_centerline"}
    ref_n = ref[:n]
    cand_n = cand[:n]
    delta = np.linalg.norm(cand_n - ref_n, axis=1)
    ref_pos_norm = np.maximum(np.linalg.norm(ref_n, axis=1), 1e-12)
    rel_to_position = delta / ref_pos_norm
    rel_to_L_ref = delta / max(float(L_ref), 1e-12)
    edges_ref = np.diff(ref_n, axis=0)
    edges_cand = np.diff(cand_n, axis=0)
    edge_ref_norm = np.linalg.norm(edges_ref, axis=1)
    edge_cand_norm = np.linalg.norm(edges_cand, axis=1)
    unit_ref = edges_ref / np.maximum(edge_ref_norm[:, None], 1e-12)
    unit_cand = edges_cand / np.maximum(edge_cand_norm[:, None], 1e-12)
    tangent_delta = np.linalg.norm(unit_cand - unit_ref, axis=1) if unit_ref.size else np.zeros((0,))
    return {
        "status": "ok",
        "point_count": int(n),
        "max_abs_point_error_m": float(np.max(delta)),
        "mean_abs_point_error_m": float(np.mean(delta)),
        "rms_abs_point_error_m": float(np.sqrt(np.mean(delta * delta))),
        "max_rel_to_position": float(np.max(rel_to_position)),
        "mean_rel_to_position": float(np.mean(rel_to_position)),
        "max_rel_to_L_ref": float(np.max(rel_to_L_ref)),
        "mean_rel_to_L_ref": float(np.mean(rel_to_L_ref)),
        "end_abs_error_m": float(delta[-1]),
        "end_rel_to_position": float(rel_to_position[-1]),
        "max_tangent_delta": float(np.max(tangent_delta)) if tangent_delta.size else 0.0,
    }


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["case_id"])] = row
    return rows


def _compare(args: argparse.Namespace) -> int:
    before_dir = Path(args.before_dir).resolve()
    after_dir = Path(args.after_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    before_rows = _read_jsonl(before_dir / "fk_pointwise_results.jsonl")
    after_rows = _read_jsonl(after_dir / "fk_pointwise_results.jsonl")
    case_ids = sorted(set(before_rows) & set(after_rows))
    threshold = float(args.threshold)
    summary: list[dict[str, Any]] = []
    for case_id in case_ids:
        before = before_rows[case_id]
        after = after_rows[case_id]
        before_cl = np.load(before_dir / str(before["centerline_file"]))
        after_cl = np.load(after_dir / str(after["centerline_file"]))
        L_ref = float(before.get("L_ref", after.get("L_ref", 1.0)))
        metrics = _pointwise_metrics(before_cl, after_cl, L_ref=L_ref)
        target_delta = 0.0
        metrics.update(
            {
                "case_id": case_id,
                "before_ok": bool(before.get("ok")),
                "after_ok": bool(after.get("ok")),
                "before_normE": float(before.get("normE", float("nan"))),
                "after_normE": float(after.get("normE", float("nan"))),
                "normE_ratio": float(after.get("normE", float("nan"))) / max(float(before.get("normE", 0.0)), 1e-12),
                "target_abs_error_m": target_delta,
                "threshold": threshold,
                "pass_rel_to_position_10pct": bool(metrics.get("max_rel_to_position", float("inf")) <= threshold),
                "pass_rel_to_L_ref_10pct": bool(metrics.get("max_rel_to_L_ref", float("inf")) <= threshold),
                "pass_target_10pct": bool(target_delta <= threshold * max(L_ref, 1e-12)),
            }
        )
        summary.append(metrics)

    (out_dir / "fk_pointwise_compare.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "fk_pointwise_compare.csv").open("w", encoding="utf-8") as handle:
        fields = [
            "case_id",
            "point_count",
            "max_abs_point_error_m",
            "mean_abs_point_error_m",
            "rms_abs_point_error_m",
            "max_rel_to_position",
            "max_rel_to_L_ref",
            "end_abs_error_m",
            "target_abs_error_m",
            "before_normE",
            "after_normE",
            "normE_ratio",
            "pass_rel_to_position_10pct",
            "pass_rel_to_L_ref_10pct",
            "pass_target_10pct",
        ]
        handle.write(",".join(fields) + "\n")
        for row in summary:
            handle.write(",".join(str(row.get(field, "")) for field in fields) + "\n")
    return 0


def _plot(args: argparse.Namespace) -> int:
    before_dir = Path(args.before_dir).resolve()
    after_dir = Path(args.after_dir).resolve()
    compare_dir = Path(args.compare_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.loads((compare_dir / "fk_pointwise_compare.json").read_text(encoding="utf-8"))
    if not rows:
        raise ValueError("No comparison rows found.")

    fig = plt.figure(figsize=(14, 10))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    axerr = fig.add_subplot(2, 2, 2)
    axmax = fig.add_subplot(2, 1, 2)
    colors = plt.cm.tab10(np.linspace(0, 1, len(rows)))
    labels: list[str] = []
    max_mm: list[float] = []

    for color, row in zip(colors, rows):
        case_id = str(row["case_id"])
        before_cl = np.load(before_dir / f"{case_id}_centerline.npy")
        after_cl = np.load(after_dir / f"{case_id}_centerline.npy")
        n = min(before_cl.shape[0], after_cl.shape[0])
        point_error_mm = np.linalg.norm(after_cl[:n] - before_cl[:n], axis=1) * 1000.0
        labels.append(case_id)
        max_mm.append(float(np.max(point_error_mm)))
        ax3d.plot(before_cl[:, 0], before_cl[:, 1], before_cl[:, 2], color=color, linewidth=2.0, label=f"{case_id} before")
        ax3d.plot(after_cl[:, 0], after_cl[:, 1], after_cl[:, 2], color=color, linewidth=1.5, linestyle="--", label=f"{case_id} after")
        axerr.plot(np.arange(n), point_error_mm, color=color, label=case_id)

    ax3d.set_title("FK before/after centerline overlay")
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.05, 1.0))
    axerr.set_title("Pointwise absolute error")
    axerr.set_xlabel("centerline point index")
    axerr.set_ylabel("error (mm)")
    axerr.set_yscale("symlog", linthresh=1e-12)
    axerr.grid(True, alpha=0.3)
    axerr.legend(fontsize=8)
    axmax.bar(labels, max_mm)
    axmax.set_title("Max pointwise error by case")
    axmax.set_ylabel("max error (mm)")
    axmax.set_yscale("log")
    axmax.tick_params(axis="x", rotation=20)
    axmax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fk_before_after_pointwise_overlay.png", dpi=180)
    plt.close(fig)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run or compare pointwise FK centerline regression cases.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--repo_root", required=True)
    run_parser.add_argument("--out_dir", required=True)
    run_parser.add_argument("--case_ids", default="")
    run_parser.add_argument("--max_iter", type=int, default=80)
    run_parser.add_argument("--tol", type=float, default=1e-5)
    run_parser.add_argument("--lm_damping", type=float, default=1e-1)
    run_parser.add_argument("--jac_method", choices=["fwd", "rev"], default="fwd")
    run_parser.add_argument("--calib_file", default="calib/mpem_calibration_file_sp=40_order=1.yaml")
    run_parser.add_argument(
        "--actuation_table_pkl",
        default="supiee_auto_diff/offline_interpolation_data/actuation_tables/actuation_table.pkl",
    )
    run_parser.add_argument("--L_protrude_max", type=float, default=0.18)
    run_parser.set_defaults(func=_run_cases)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--before_dir", required=True)
    compare_parser.add_argument("--after_dir", required=True)
    compare_parser.add_argument("--out_dir", required=True)
    compare_parser.add_argument("--threshold", type=float, default=0.10)
    compare_parser.set_defaults(func=_compare)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("--before_dir", required=True)
    plot_parser.add_argument("--after_dir", required=True)
    plot_parser.add_argument("--compare_dir", required=True)
    plot_parser.add_argument("--out_dir", required=True)
    plot_parser.set_defaults(func=_plot)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
