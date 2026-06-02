"""Run terminal-pose constraints on the c3 segment-wrench shooting path."""

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
P2C1_DIR = ROOT / "experiment_records_ik_optimize" / "20260601110758_p2c1_physical_shooting_convergence"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(P2C1_DIR) not in sys.path:
    sys.path.insert(0, str(P2C1_DIR))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inverse_kinematics.ik_observations import quaternion_so3_residual, quat_to_rotmat_wxyz

import run_p2c1_convergence as p2c1

jax.config.update("jax_enable_x64", True)
OUT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PoseCandidateSpec:
    """Terminal-pose shooting candidate settings."""

    name: str
    max_nfev: int
    centerline_weight: float = 20.0
    tip_weight: float = 120.0
    so3_weight: float = 2.0
    rigid_weight: float = 20.0
    boundary_weight: float = 0.02
    loss: str = "linear"


@dataclass
class PoseCandidateResult:
    """Terminal-pose shooting solve result."""

    spec: PoseCandidateSpec
    y: np.ndarray
    out: dict[str, np.ndarray]
    residual_history: list[float]
    success: bool
    cost: float
    nfev: int
    wall_s: float


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


def _np(x: Any) -> np.ndarray:
    return np.asarray(jax.device_get(x), dtype=float)


def _rotmat(q: np.ndarray) -> np.ndarray:
    return _np(quat_to_rotmat_wxyz(jnp.asarray(q, dtype=jnp.float64))).reshape(3, 3)


def _so3_residual(current_q: np.ndarray, target_q: np.ndarray) -> np.ndarray:
    return _np(
        quaternion_so3_residual(
            jnp.asarray(current_q, dtype=jnp.float64),
            jnp.asarray(target_q, dtype=jnp.float64),
            sigma_angle=1.0,
        )
    ).reshape(3)


def _axis_body(engine: Any) -> np.ndarray:
    axis_body = _np(engine.axis_body).reshape(3)
    return axis_body / max(float(np.linalg.norm(axis_body)), 1e-12)


def _oracle_terminal_pose(oracle: dict[str, Any], engine: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the physical distal-end pose of the last rigid segment.

    ``query_sites`` currently names the rigid states as midpoints, but the FK
    residual constrains them at ``sigma=rigid.length``. Treat them as distal
    rigid states here and record the naming mismatch in the experiment report.
    """
    sites = oracle["sites"]
    rigid_distal_p = _np(sites["rigid_mid_p_dim"])
    rigid_distal_q = _np(sites["rigid_mid_Q_wxyz"])
    if rigid_distal_p.size == 0:
        return _np(sites["tip_p_dim"]).reshape(3), _np(sites["tip_Q_wxyz"]).reshape(4)
    return rigid_distal_p[-1].reshape(3), rigid_distal_q[-1].reshape(4)


def _oracle_rigid_midpoint_poses(oracle: dict[str, Any], engine: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return true rigid midpoint poses inferred from stored distal states."""
    sites = oracle["sites"]
    rigid_distal_p = _np(sites["rigid_mid_p_dim"])
    rigid_distal_q = _np(sites["rigid_mid_Q_wxyz"])
    if rigid_distal_p.size == 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 4), dtype=float)
    axis = _axis_body(engine)
    mids = []
    for i, (p_dist, q_dist) in enumerate(zip(rigid_distal_p, rigid_distal_q, strict=False)):
        mids.append(p_dist.reshape(3) - 0.5 * float(engine.rigid_lengths[i]) * (_rotmat(q_dist.reshape(4)) @ axis))
    return np.asarray(mids, dtype=float), rigid_distal_q


def _terminal_metrics(oracle: dict[str, Any], engine: Any, out: dict[str, np.ndarray]) -> dict[str, float]:
    p_target, q_target = _oracle_terminal_pose(oracle, engine)
    terminal_so3 = _so3_residual(out["tip_q"], q_target)
    oracle_cl = np.asarray(oracle["centerline"], dtype=float)
    shoot_cl = p2c1._arc_resample(out["centerline"], len(oracle_cl))
    diffs = np.linalg.norm(shoot_cl - oracle_cl, axis=1)
    rigid_p_oracle, rigid_q_oracle = _oracle_rigid_midpoint_poses(oracle, engine)
    rigid_count = min(len(rigid_p_oracle), len(out["rigid_mid_p"]))
    rigid_err = float("nan")
    axis_err = float("nan")
    if rigid_count:
        rigid_err = float(np.sqrt(np.mean(np.linalg.norm(out["rigid_mid_p"][:rigid_count] - rigid_p_oracle[:rigid_count], axis=1) ** 2)))
        axis_errs = []
        for i in range(rigid_count):
            axis_errs.append(np.linalg.norm(np.cross(p2c1._axis_from_q(rigid_q_oracle[i]), p2c1._axis_from_q(out["rigid_mid_q"][i]))))
        axis_err = float(np.sqrt(np.mean(np.asarray(axis_errs) ** 2)))
    flex_tip_p = _np(oracle["sites"]["tip_p_dim"]).reshape(3)
    flex_tip_q = _np(oracle["sites"]["tip_Q_wxyz"]).reshape(4)
    flex_so3 = _so3_residual(out["tip_q"], flex_tip_q)
    return {
        "tip_error": float(np.linalg.norm(out["tip_p"] - flex_tip_p)),
        "rms_centerline_error": float(np.sqrt(np.mean(diffs**2))),
        "max_centerline_error": float(np.max(diffs)),
        "rigid_midpoint_rms_error": rigid_err,
        "axis_proxy_rms_error": axis_err,
        "boundary_residual_norm": float(np.linalg.norm(out["boundary"])),
        "terminal_position_error": float(np.linalg.norm(out["tip_p"] - p_target)),
        "terminal_so3_error_rad": float(np.linalg.norm(terminal_so3)),
        "terminal_so3_error_deg": float(np.degrees(np.linalg.norm(terminal_so3))),
        "flex_tip_position_error": float(np.linalg.norm(out["tip_p"] - flex_tip_p)),
        "flex_tip_so3_error_deg": float(np.degrees(np.linalg.norm(flex_so3))),
    }


def _pose_stage(metrics: dict[str, float], length: float) -> str:
    pos_ratio = float(metrics["terminal_position_error"]) / max(float(length), 1e-12)
    angle_deg = float(metrics["terminal_so3_error_deg"])
    if pos_ratio < 0.05 and angle_deg < 10.0:
        return "B"
    if pos_ratio < 0.10 and angle_deg < 20.0:
        return "A"
    return "fail"


def _pose_residual(spec: PoseCandidateSpec, oracle: dict[str, Any], engine: Any, out: dict[str, np.ndarray]) -> np.ndarray:
    oracle_cl = np.asarray(oracle["centerline"], dtype=float)
    shoot_cl = p2c1._arc_resample(out["centerline"], len(oracle_cl))
    sites = oracle["sites"]
    p_target, q_target = _oracle_terminal_pose(oracle, engine)
    pieces = []
    if spec.centerline_weight:
        pieces.append(spec.centerline_weight * (shoot_cl - oracle_cl).reshape(-1))
    if spec.tip_weight:
        pieces.append(spec.tip_weight * (out["tip_p"] - p_target))
    if spec.so3_weight:
        pieces.append(spec.so3_weight * _so3_residual(out["tip_q"], q_target))
    if spec.rigid_weight:
        rigid_target, _rigid_q = _oracle_rigid_midpoint_poses(oracle, engine)
        n = min(len(rigid_target), len(out["rigid_mid_p"]))
        pieces.append(spec.rigid_weight * (out["rigid_mid_p"][:n] - rigid_target[:n]).reshape(-1))
    if spec.boundary_weight:
        pieces.append(spec.boundary_weight * out["boundary"])
    return np.concatenate([piece.reshape(-1) for piece in pieces])


def _solve_pose_candidate(
    spec: PoseCandidateSpec,
    oracle: dict[str, Any],
    engine: Any,
    *,
    steps_per_flex: int,
    y0: np.ndarray | None = None,
) -> PoseCandidateResult:
    if y0 is None:
        y0 = p2c1._y_guess_from_oracle(oracle["z"], oracle["params"], "segment_wrench")
    history: list[float] = []

    def residual(y: np.ndarray) -> np.ndarray:
        out = p2c1._forward_modes(oracle["params"], y, "segment_wrench", steps_per_flex=steps_per_flex)
        r = _pose_residual(spec, oracle, engine, out)
        history.append(float(np.linalg.norm(r)))
        return r

    t0 = time.perf_counter()
    opt = least_squares(
        residual,
        np.asarray(y0, dtype=float),
        method="trf",
        x_scale=np.ones_like(y0),
        loss=spec.loss,
        max_nfev=int(spec.max_nfev),
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    wall_s = time.perf_counter() - t0
    out = p2c1._forward_modes(oracle["params"], opt.x, "segment_wrench", steps_per_flex=steps_per_flex)
    return PoseCandidateResult(
        spec=spec,
        y=np.asarray(opt.x, dtype=float),
        out=out,
        residual_history=history,
        success=bool(opt.success),
        cost=float(opt.cost),
        nfev=int(opt.nfev),
        wall_s=float(wall_s),
    )


def _solve_old_c3(oracle: dict[str, Any], *, steps_per_flex: int, max_nfev: int) -> p2c1.CandidateResult:
    spec = p2c1.CandidateSpec("c3_segment_wrench_18d", "segment_wrench", "calib_modes", max_nfev, 40.0, 80.0, 40.0, 1.0, 0.02)
    return p2c1._solve_candidate(spec, oracle, steps_per_flex=steps_per_flex)


def _row_from_result(
    *,
    name: str,
    result: Any,
    oracle: dict[str, Any],
    engine: Any,
    length: float,
    mode: str,
) -> dict[str, Any]:
    out = result.out if isinstance(result, PoseCandidateResult) else {
        "centerline": result.centerline,
        "rigid_mid_p": result.rigid_mid_p,
        "rigid_mid_q": result.rigid_mid_q,
        "tip_p": result.tip_p,
        "tip_q": result.tip_q,
        "boundary": result.boundary,
    }
    metrics = _terminal_metrics(oracle, engine, out)
    return {
        "candidate": name,
        "mode": mode,
        "state_mode": "segment_wrench",
        "state_dim": int(result.y.size),
        "success": bool(result.success),
        "wall_s": float(result.wall_s),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "pose_stage": _pose_stage(metrics, length),
        **metrics,
    }


def _task_pose_vector(oracle: dict[str, Any], engine: Any, *, base_target_q: np.ndarray | None = None) -> np.ndarray:
    p, q = _oracle_terminal_pose(oracle, engine)
    if base_target_q is None:
        return np.concatenate([p, np.zeros(3, dtype=float)])
    return np.concatenate([p, _so3_residual(q, base_target_q)])


def _shooting_pose_vector(out: dict[str, np.ndarray], *, base_target_q: np.ndarray) -> np.ndarray:
    return np.concatenate([out["tip_p"], _so3_residual(out["tip_q"], base_target_q)])


def _run_reoptimized_fd_audit(
    *,
    engine: Any,
    base_oracle: dict[str, Any],
    base_result: PoseCandidateResult,
    spec: PoseCandidateSpec,
    steps_per_flex: int,
    audit_nfev: int,
    eps_values: list[float],
) -> dict[str, Any]:
    I0 = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0], dtype=float)
    active_index = 5
    _, base_q = _oracle_terminal_pose(base_oracle, engine)
    rows = []
    cache: dict[float, dict[str, Any]] = {}

    def solve_delta(delta_i: float) -> dict[str, Any]:
        if delta_i not in cache:
            dI = np.zeros_like(I0)
            dI[active_index] = delta_i
            oracle = p2c1.p2c._solve_oracle(engine, I0 + dI, z0=base_oracle["z"])
            shooting = _solve_pose_candidate(
                spec,
                oracle,
                engine,
                steps_per_flex=steps_per_flex,
                y0=base_result.y,
            )
            cache[delta_i] = {"oracle": oracle, "shooting": shooting}
        return cache[delta_i]

    t0 = time.perf_counter()
    for eps in eps_values:
        plus = solve_delta(float(eps))
        minus = solve_delta(-float(eps))
        oracle_fd = (
            _task_pose_vector(plus["oracle"], engine, base_target_q=base_q)
            - _task_pose_vector(minus["oracle"], engine, base_target_q=base_q)
        ) / (2.0 * eps)
        shooting_fd = (
            _shooting_pose_vector(plus["shooting"].out, base_target_q=base_q)
            - _shooting_pose_vector(minus["shooting"].out, base_target_q=base_q)
        ) / (2.0 * eps)
        for label, sl in (("position", slice(0, 3)), ("so3", slice(3, 6)), ("pose", slice(0, 6))):
            diff = shooting_fd[sl] - oracle_fd[sl]
            denom = max(float(np.linalg.norm(oracle_fd[sl])), 1e-12)
            rows.append(
                {
                    "candidate": spec.name,
                    "mode": "reoptimized",
                    "active_current_index": active_index,
                    "eps": float(eps),
                    "task": label,
                    "oracle_fd_norm": float(np.linalg.norm(oracle_fd[sl])),
                    "shooting_fd_norm": float(np.linalg.norm(shooting_fd[sl])),
                    "difference_norm": float(np.linalg.norm(diff)),
                    "relative_error": float(np.linalg.norm(diff) / denom),
                }
            )
    return {
        "metadata": {
            "mode": "reoptimized",
            "candidate": spec.name,
            "active_current_index": active_index,
            "eps_values": eps_values,
            "audit_nfev": int(audit_nfev),
            "wall_s": float(time.perf_counter() - t0),
            "solve_count": int(len(cache)),
        },
        "rows": rows,
    }


def _plot_centerline(case_dir: Path, oracle: dict[str, Any], result: Any, title: str) -> None:
    out = result.out if isinstance(result, PoseCandidateResult) else {
        "centerline": result.centerline,
        "tip_p": result.tip_p,
    }
    oracle_cl = np.asarray(oracle["centerline"], dtype=float)
    shoot_cl = p2c1._arc_resample(out["centerline"], len(oracle_cl))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(oracle_cl[:, 0], oracle_cl[:, 1], oracle_cl[:, 2], "k-", lw=2, label="FK oracle centerline")
    ax.plot(shoot_cl[:, 0], shoot_cl[:, 1], shoot_cl[:, 2], "r--", lw=2, label="shooting")
    ax.scatter([oracle_cl[0, 0]], [oracle_cl[0, 1]], [oracle_cl[0, 2]], c="blue", s=35, label="base")
    ax.scatter([shoot_cl[-1, 0]], [shoot_cl[-1, 1]], [shoot_cl[-1, 2]], c="red", s=35, label="shooting terminal")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(case_dir / f"{title}.png", dpi=180)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iter", type=int, default=120)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--steps_per_flex", type=int, default=4)
    parser.add_argument("--candidate_max_nfev", type=int, default=16)
    parser.add_argument("--audit_nfev", type=int, default=8)
    parser.add_argument("--audit_eps", default="0.02,0.05")
    parser.add_argument("--skip_fd_audit", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    engine = p2c1.p2c._build_engine((1, 1, 1), max_iter=int(args.max_iter), tol=float(args.tol))
    I0 = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0], dtype=float)
    oracle = p2c1.p2c._solve_oracle(engine, I0, z0=None)
    length = float(np.sum(engine.rigid_lengths) + sum(engine._compute_flex_lengths(L_protrude=0.11)))
    case_dir = OUT_DIR / "M111"
    case_dir.mkdir(parents=True, exist_ok=True)

    old_c3 = _solve_old_c3(oracle, steps_per_flex=int(args.steps_per_flex), max_nfev=int(args.candidate_max_nfev))
    pose_spec = PoseCandidateSpec("c3_pose_terminal_v1", max_nfev=int(args.candidate_max_nfev))
    pose_c3 = _solve_pose_candidate(pose_spec, oracle, engine, steps_per_flex=int(args.steps_per_flex))

    rows = [
        _row_from_result(name="c3_segment_wrench_18d", result=old_c3, oracle=oracle, engine=engine, length=length, mode="legacy_axis_proxy"),
        _row_from_result(name=pose_spec.name, result=pose_c3, oracle=oracle, engine=engine, length=length, mode="terminal_pose_so3"),
    ]

    np.save(case_dir / "oracle_centerline.npy", np.asarray(oracle["centerline"], dtype=float))
    np.save(case_dir / "legacy_c3_centerline.npy", old_c3.centerline)
    np.save(case_dir / "legacy_c3_state_y.npy", old_c3.y)
    np.save(case_dir / "pose_c3_centerline.npy", pose_c3.out["centerline"])
    np.save(case_dir / "pose_c3_state_y.npy", pose_c3.y)
    np.save(case_dir / "pose_c3_residual_history.npy", np.asarray(pose_c3.residual_history, dtype=float))
    _plot_centerline(case_dir, oracle, old_c3, "legacy_c3_oracle_vs_shooting")
    _plot_centerline(case_dir, oracle, pose_c3, "pose_c3_oracle_vs_shooting")

    fd_payload: dict[str, Any] | None = None
    if not args.skip_fd_audit:
        fd_payload = _run_reoptimized_fd_audit(
            engine=engine,
            base_oracle=oracle,
            base_result=pose_c3,
            spec=PoseCandidateSpec("c3_pose_terminal_v1", max_nfev=int(args.audit_nfev)),
            steps_per_flex=int(args.steps_per_flex),
            audit_nfev=int(args.audit_nfev),
            eps_values=[float(x.strip()) for x in str(args.audit_eps).split(",") if x.strip()],
        )
        (OUT_DIR / "fd_audit.json").write_text(json.dumps(_jsonable(fd_payload), ensure_ascii=False, indent=2), encoding="utf-8")
        _write_csv(OUT_DIR / "fd_audit.csv", fd_payload["rows"])

    payload = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "M_list": [1, 1, 1],
            "max_iter": int(args.max_iter),
            "tol": float(args.tol),
            "steps_per_flex": int(args.steps_per_flex),
            "candidate_max_nfev": int(args.candidate_max_nfev),
            "terminal_convention": "distal end of the last rigid segment",
            "flex_tip_convention": "distal node of the last flexible segment; recorded for drift diagnostics only",
            "catheter_length": length,
            "total_wall_s": float(time.perf_counter() - t0),
            "non_production": True,
        },
        "oracle": {
            "base_ok": bool(oracle["ok"]),
            "base_normE": float(oracle["stats"].normE),
            "base_wall_s": float(oracle["wall_s"]),
        },
        "rows": rows,
        "best_candidate": min(rows, key=lambda row: (row["pose_stage"] == "fail", row["terminal_position_error"], row["terminal_so3_error_deg"])),
        "fd_audit": fd_payload,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "summary.csv", rows)
    print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
