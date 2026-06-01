"""Run non-production shooting-sensitivity diagnostics.

This experiment-local script tests whether a low-dimensional shape shooting
state can provide better-conditioned task sensitivities than the dense FK LM
normal equation. It does not modify or replace the production IK path.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from forward_kinematics_optimized.fk_engine import ForwardKinematicsEngine
from inverse_kinematics.ik_artifacts import extract_centerline_dim
from inverse_kinematics.ik_observations import axis_from_quat

jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ShapeFit:
    """Low-dimensional cubic Bezier shooting state and reconstructed geometry."""

    theta: np.ndarray
    centerline: np.ndarray
    rms_error: float
    max_error: float
    tip_error: float
    site_s: dict[str, float]


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(ROOT), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc!r}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _parse_m_list(text: str) -> tuple[int, int, int]:
    vals = [int(float(x.strip())) for x in str(text).split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError("--M_list must contain three comma-separated integers")
    return vals[0], vals[1], vals[2]


def _build_engine(M_list: tuple[int, int, int], *, max_iter: int, tol: float) -> ForwardKinematicsEngine:
    calib = ROOT / "calib" / "mpem_calibration_file_sp=40_order=1.yaml"
    return ForwardKinematicsEngine(
        flex_lengths_tail=(0.03, 0.03),
        rigid_lengths=(0.003, 0.003, 0.003),
        M_list=M_list,
        L1_min=1e-6,
        L_protrude_max=0.11,
        p0_dim=(0.0, 0.0, 0.0),
        Q0_wxyz=(0.70710678, 0.0, 0.70710678, 0.0),
        axis_body=(0.0, 0.0, 1.0),
        enable_gravity=True,
        enable_magnetics=True,
        calib_file=str(calib),
        m_body_list=[(0.0, 0.0, -0.005301), (0.0, 0.0, 0.005301), (0.005301, 0.0, 0.0)],
        max_iter=int(max_iter),
        tol=float(tol),
        lm_damping=1e-1,
        jac_method="fwd",
        linear_solver="dense_fused_cholesky",
        scale_include_external=True,
    )


def _directions() -> list[tuple[str, np.ndarray]]:
    directions = [
        ("dA", np.asarray([0.2, -0.3, 0.1, 0.0, -0.15, 0.25, -0.1, 0.05], dtype=float)),
        ("dB", np.asarray([-0.1, 0.25, -0.2, 0.15, 0.1, -0.05, 0.3, -0.2], dtype=float)),
    ]
    return [(name, vec / max(float(np.linalg.norm(vec)), 1e-12)) for name, vec in directions]


def _arc_s(centerline: np.ndarray) -> np.ndarray:
    diffs = np.diff(centerline, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(lengths)])
    total = max(float(arc[-1]), 1e-12)
    return arc / total


def _bezier_basis(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=float).reshape(-1)
    return np.stack([(1 - s) ** 3, 3 * (1 - s) ** 2 * s, 3 * (1 - s) * s**2, s**3], axis=1)


def _eval_bezier(theta: np.ndarray, s: np.ndarray) -> np.ndarray:
    ctrl = np.asarray(theta, dtype=float).reshape(4, 3)
    return _bezier_basis(s) @ ctrl


def _fit_shape(centerline: np.ndarray, site_positions: dict[str, np.ndarray]) -> ShapeFit:
    centerline = np.asarray(centerline, dtype=float)
    s = _arc_s(centerline)
    basis = _bezier_basis(s)
    p0 = centerline[0]
    p3 = centerline[-1]
    # P0 and P3 are fixed. Fit only the two internal control points so the
    # reconstructed tip remains identical to the oracle tip.
    rhs = centerline - basis[:, [0]] * p0.reshape(1, 3) - basis[:, [3]] * p3.reshape(1, 3)
    ctrl_mid, *_ = np.linalg.lstsq(basis[:, 1:3], rhs, rcond=None)
    ctrl = np.vstack([p0.reshape(1, 3), ctrl_mid, p3.reshape(1, 3)])
    recon = _eval_bezier(ctrl.reshape(-1), s)
    errors = np.linalg.norm(recon - centerline, axis=1)
    site_s: dict[str, float] = {}
    for name, point in site_positions.items():
        idx = int(np.argmin(np.linalg.norm(centerline - np.asarray(point, dtype=float).reshape(1, 3), axis=1)))
        site_s[name] = float(s[idx])
    return ShapeFit(
        theta=ctrl.reshape(-1),
        centerline=recon,
        rms_error=float(np.sqrt(np.mean(errors**2))),
        max_error=float(np.max(errors)),
        tip_error=float(np.linalg.norm(recon[-1] - centerline[-1])),
        site_s=site_s,
    )


def _axis_from_centerline(centerline: np.ndarray, s_grid: np.ndarray, s0: float) -> np.ndarray:
    idx = int(np.argmin(np.abs(s_grid - float(s0))))
    if idx <= 0:
        tangent = centerline[1] - centerline[0]
    elif idx >= len(centerline) - 1:
        tangent = centerline[-1] - centerline[-2]
    else:
        tangent = centerline[idx + 1] - centerline[idx - 1]
    norm = max(float(np.linalg.norm(tangent)), 1e-12)
    return tangent / norm


def _shooting_sites(fit: ShapeFit, s_grid: np.ndarray, target_site_s: dict[str, float]) -> dict[str, np.ndarray]:
    centerline = _eval_bezier(fit.theta, s_grid)
    tip_axis = _axis_from_centerline(centerline, s_grid, 1.0)
    rigid0_axis = _axis_from_centerline(centerline, s_grid, target_site_s["rigid0"])
    return {
        "centerline": centerline,
        "tip_p": centerline[-1],
        "tip_axis": tip_axis,
        "rigid0_axis": rigid0_axis,
    }


def _residual_from_shooting(
    sites: dict[str, np.ndarray],
    *,
    residual_type: str,
    target_tip: np.ndarray,
    target_axis: np.ndarray,
) -> np.ndarray:
    pieces = []
    if residual_type in ("position", "position_axis"):
        pieces.append(np.asarray(sites["tip_p"], dtype=float) - np.asarray(target_tip, dtype=float))
    if residual_type in ("axis", "position_axis"):
        cur_axis = np.asarray(sites["rigid0_axis"], dtype=float)
        des_axis = np.asarray(target_axis, dtype=float)
        cur_axis = cur_axis / max(float(np.linalg.norm(cur_axis)), 1e-12)
        des_axis = des_axis / max(float(np.linalg.norm(des_axis)), 1e-12)
        pieces.append(np.cross(des_axis, cur_axis))
    if not pieces:
        raise ValueError(f"unsupported residual_type: {residual_type}")
    return np.concatenate([piece.reshape(-1) for piece in pieces], axis=0)


def _residual_from_oracle_sites(
    sites: dict[str, Any],
    *,
    residual_type: str,
    target_tip: np.ndarray,
    target_axis: np.ndarray,
) -> np.ndarray:
    pieces = []
    if residual_type in ("position", "position_axis"):
        pieces.append(np.asarray(jax.device_get(sites["tip_p_dim"]), dtype=float) - np.asarray(target_tip, dtype=float))
    if residual_type in ("axis", "position_axis"):
        cur_axis = np.asarray(jax.device_get(axis_from_quat(sites["rigid_mid_Q_wxyz"][0])), dtype=float)
        des_axis = np.asarray(target_axis, dtype=float)
        cur_axis = cur_axis / max(float(np.linalg.norm(cur_axis)), 1e-12)
        des_axis = des_axis / max(float(np.linalg.norm(des_axis)), 1e-12)
        pieces.append(np.cross(des_axis, cur_axis))
    return np.concatenate([piece.reshape(-1) for piece in pieces], axis=0)


def _solve_oracle(
    engine: ForwardKinematicsEngine,
    currents: np.ndarray,
    *,
    z0: Any | None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    z, params, _meshes, ok, stats = engine.solve_with_stats(
        coil_currents=jnp.asarray(currents, dtype=jnp.float64),
        L_protrude=0.11,
        warm_start=True,
        override_z0_bar=z0,
        return_stats=True,
        stats_include_jacobian=False,
    )
    if hasattr(z, "block_until_ready"):
        z.block_until_ready()
    wall_s = time.perf_counter() - t0
    centerline = extract_centerline_dim(z, M_list=params.M_list, scales=params.scales)
    sites = engine.query_sites(z_bar=z, scales=params.scales)
    return {"z": z, "params": params, "ok": bool(ok), "stats": stats, "wall_s": float(wall_s), "centerline": centerline, "sites": sites}


def _svd_summary(J: np.ndarray, cutoff_ratio: float) -> dict[str, Any]:
    singular = np.linalg.svd(J, compute_uv=False)
    if singular.size == 0:
        return {"singular_values": [], "effective_rank": 0, "condition_number": None, "truncated_count": 0}
    smax = float(np.max(singular))
    cutoff = float(cutoff_ratio) * smax
    kept = singular > cutoff
    kept_vals = singular[kept]
    cond = None if kept_vals.size == 0 else float(np.max(kept_vals) / max(float(np.min(kept_vals)), 1e-300))
    return {
        "singular_values": singular.tolist(),
        "effective_rank": int(np.sum(kept)),
        "condition_number": cond,
        "truncated_count": int(singular.size - np.sum(kept)),
    }


def _grade(rel_error: float) -> str:
    if rel_error < 0.2:
        return "usable"
    if rel_error < 0.5:
        return "diagnostic-only"
    return "untrusted"


def _plot_centerline(case_dir: Path, oracle: np.ndarray, shooting: np.ndarray, title: str) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(oracle[:, 0], oracle[:, 1], oracle[:, 2], "k-", lw=2, label="oracle FK")
    ax.plot(shooting[:, 0], shooting[:, 1], shooting[:, 2], "r--", lw=2, label="shooting fit")
    ax.scatter(oracle[0, 0], oracle[0, 1], oracle[0, 2], c="blue", s=30, label="base")
    ax.scatter(oracle[-1, 0], oracle[-1, 1], oracle[-1, 2], c="green", s=40, label="oracle tip")
    ax.scatter(shooting[-1, 0], shooting[-1, 1], shooting[-1, 2], c="red", s=30, label="shooting tip")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(case_dir / "oracle_vs_shooting_centerline.png", dpi=180)
    plt.close(fig)


def _plot_singular(case_dir: Path, rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in rows:
        values = row["singular_values"]
        ax.semilogy(range(1, len(values) + 1), values, marker="o", label=row["residual_type"])
    ax.set_title("Shooting task Jacobian singular values")
    ax.set_xlabel("index")
    ax.set_ylabel("singular value")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(case_dir / "singular_values.png", dpi=180)
    plt.close(fig)


def _plot_fd_errors(case_dir: Path, rows: list[dict[str, Any]]) -> None:
    labels = [f"{r['residual_type']}\\n{r['direction']} eps={r['eps']}" for r in rows]
    vals = [r["relative_error"] for r in rows]
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(rows)), 5))
    ax.bar(range(len(vals)), vals)
    ax.axhline(0.2, color="green", linestyle="--", label="usable")
    ax.axhline(0.5, color="orange", linestyle="--", label="diagnostic-only")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=70, ha="right")
    ax.set_ylabel("relative error")
    ax.set_title("Shooting sensitivity FD audit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(case_dir / "fd_audit_errors.png", dpi=180)
    plt.close(fig)


def run_case(
    *,
    M_list: tuple[int, int, int],
    residual_types: list[str],
    jac_eps: float,
    audit_eps_values: list[float],
    cutoff_ratio: float,
    max_iter: int,
    tol: float,
) -> dict[str, Any]:
    case_dir = OUT_DIR / f"M{''.join(str(m) for m in M_list)}"
    case_dir.mkdir(parents=True, exist_ok=True)
    engine = _build_engine(M_list, max_iter=max_iter, tol=tol)
    I0 = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0], dtype=float)

    base = _solve_oracle(engine, I0, z0=None)
    base_sites = base["sites"]
    base_centerline = np.asarray(base["centerline"], dtype=float)
    target_tip = np.asarray(jax.device_get(base_sites["tip_p_dim"]), dtype=float)
    target_axis = np.asarray(jax.device_get(axis_from_quat(base_sites["rigid_mid_Q_wxyz"][0])), dtype=float)
    site_positions = {"rigid0": np.asarray(jax.device_get(base_sites["rigid_mid_p_dim"][0]), dtype=float)}
    base_fit = _fit_shape(base_centerline, site_positions)
    s_grid = _arc_s(base_centerline)
    shooting_base_sites = _shooting_sites(base_fit, s_grid, base_fit.site_s)

    cache: dict[tuple[float, tuple[float, ...]], dict[str, Any]] = {}

    def eval_current(delta: np.ndarray) -> dict[str, Any]:
        key = (float(np.linalg.norm(delta)), tuple(np.round(delta, 12).tolist()))
        if key not in cache:
            item = _solve_oracle(engine, I0 + delta, z0=base["z"])
            fit = _fit_shape(np.asarray(item["centerline"], dtype=float), site_positions)
            shoot_sites = _shooting_sites(fit, s_grid, base_fit.site_s)
            item["fit"] = fit
            item["shooting_sites"] = shoot_sites
            cache[key] = item
        return cache[key]

    nI = int(I0.size)
    eye = np.eye(nI, dtype=float)
    jac_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    residual_jacobians: dict[str, np.ndarray] = {}

    for residual_type in residual_types:
        base_res = _residual_from_shooting(
            shooting_base_sites,
            residual_type=residual_type,
            target_tip=target_tip,
            target_axis=target_axis,
        )
        J_cols = []
        for col in range(nI):
            plus = eval_current(float(jac_eps) * eye[col])
            minus = eval_current(-float(jac_eps) * eye[col])
            rp = _residual_from_shooting(
                plus["shooting_sites"],
                residual_type=residual_type,
                target_tip=target_tip,
                target_axis=target_axis,
            )
            rm = _residual_from_shooting(
                minus["shooting_sites"],
                residual_type=residual_type,
                target_tip=target_tip,
                target_axis=target_axis,
            )
            J_cols.append((rp - rm) / (2.0 * float(jac_eps)))
        J = np.stack(J_cols, axis=1)
        residual_jacobians[residual_type] = J
        svd = _svd_summary(J, cutoff_ratio)
        jac_rows.append(
            {
                "M_list": ",".join(str(m) for m in M_list),
                "residual_type": residual_type,
                "jac_eps": float(jac_eps),
                "cutoff_ratio": float(cutoff_ratio),
                "residual_dim": int(base_res.size),
                **svd,
            }
        )

        for direction_name, direction in _directions():
            pred = J @ direction
            for eps in audit_eps_values:
                plus = eval_current(float(eps) * direction)
                minus = eval_current(-float(eps) * direction)
                rp_oracle = _residual_from_oracle_sites(
                    plus["sites"],
                    residual_type=residual_type,
                    target_tip=target_tip,
                    target_axis=target_axis,
                )
                rm_oracle = _residual_from_oracle_sites(
                    minus["sites"],
                    residual_type=residual_type,
                    target_tip=target_tip,
                    target_axis=target_axis,
                )
                fd = (rp_oracle - rm_oracle) / (2.0 * float(eps))
                diff = float(np.linalg.norm(pred - fd))
                fd_norm = max(float(np.linalg.norm(fd)), 1e-12)
                rel = diff / fd_norm
                audit_rows.append(
                    {
                        "M_list": ",".join(str(m) for m in M_list),
                        "residual_type": residual_type,
                        "direction": direction_name,
                        "eps": float(eps),
                        "jac_eps": float(jac_eps),
                        "relative_error": float(rel),
                        "difference_norm": float(diff),
                        "fd_norm": float(fd_norm),
                        "pred_norm": float(np.linalg.norm(pred)),
                        "grade": _grade(rel),
                    }
                )

    np.save(case_dir / "oracle_centerline.npy", base_centerline)
    np.save(case_dir / "shooting_centerline.npy", shooting_base_sites["centerline"])
    np.savez(
        case_dir / "shooting_data.npz",
        theta=base_fit.theta,
        oracle_centerline=base_centerline,
        shooting_centerline=shooting_base_sites["centerline"],
        target_tip=target_tip,
        target_axis=target_axis,
    )
    with (case_dir / "jacobians.json").open("w", encoding="utf-8") as f:
        json.dump({k: v.tolist() for k, v in residual_jacobians.items()}, f, indent=2)

    _plot_centerline(case_dir, base_centerline, shooting_base_sites["centerline"], f"M={M_list} oracle vs shooting")
    _plot_singular(case_dir, jac_rows)
    _plot_fd_errors(case_dir, audit_rows)

    return {
        "case_dir": str(case_dir.relative_to(OUT_DIR)),
        "M_list": M_list,
        "base_ok": bool(base["ok"]),
        "base_normE": float(base["stats"].normE),
        "base_wall_s": float(base["wall_s"]),
        "shooting_state_dim": int(base_fit.theta.size),
        "shooting_free_fit_dim": 6,
        "shooting_state_type": "cubic_bezier_shape_coefficients",
        "geometry": {
            "tip_error": float(base_fit.tip_error),
            "rms_centerline_error": float(base_fit.rms_error),
            "max_centerline_error": float(base_fit.max_error),
        },
        "jacobian_rows": jac_rows,
        "audit_rows": audit_rows,
        "solve_count": int(len(cache) + 1),
        "cache_solve_wall_s": float(sum(item["wall_s"] for item in cache.values())),
    }


def _write_outputs(payload: dict[str, Any]) -> None:
    rows = [row for case in payload["cases"] for row in case["audit_rows"]]
    payload["row_count"] = len(rows)
    payload["grade_counts"] = {grade: sum(1 for row in rows if row["grade"] == grade) for grade in ("usable", "diagnostic-only", "untrusted")}
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    fields = sorted(rows[0].keys()) if rows else []
    with (OUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--M_list", default="1,1,1")
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--jac_eps", type=float, default=0.05)
    parser.add_argument("--audit_eps", default="0.10,0.25")
    parser.add_argument("--cutoff_ratio", type=float, default=1e-5)
    args = parser.parse_args()

    residual_types = ["position", "axis", "position_axis"]
    if args.mode == "full":
        cases = [_parse_m_list(args.M_list), (3, 3, 3)]
    else:
        cases = [_parse_m_list(args.M_list)]
    audit_eps_values = [float(x.strip()) for x in str(args.audit_eps).split(",") if x.strip()]

    payload = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "python": sys.executable,
            "mode": args.mode,
            "max_iter": int(args.max_iter),
            "tol": float(args.tol),
            "jac_eps": float(args.jac_eps),
            "audit_eps_values": audit_eps_values,
            "cutoff_ratio": float(args.cutoff_ratio),
            "non_production": True,
        },
        "cases": [],
    }
    for M_list in cases:
        payload["cases"].append(
            run_case(
                M_list=M_list,
                residual_types=residual_types,
                jac_eps=float(args.jac_eps),
                audit_eps_values=audit_eps_values,
                cutoff_ratio=float(args.cutoff_ratio),
                max_iter=int(args.max_iter),
                tol=float(args.tol),
            )
        )
        _write_outputs(payload)
    _write_outputs(payload)
    print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
