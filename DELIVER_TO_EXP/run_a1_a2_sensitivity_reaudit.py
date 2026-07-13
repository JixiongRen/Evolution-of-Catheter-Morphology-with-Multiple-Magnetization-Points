"""Run branch-preserving sensitivity audits for supervised IK snapshots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FK_DIR = _REPO_ROOT / "forward_kinematics_optimized"
for _path in (_REPO_ROOT, _FK_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import forward_kinematics_optimized.fk as fk_cli
from forward_kinematics_optimized.fk_engine import ForwardKinematicsEngine
from inverse_kinematics.ik_artifacts import extract_centerline_dim
from inverse_kinematics.ik_branch_state import array_sha256, current_git_commit, git_tracked_dirty, git_untracked_count
from inverse_kinematics.ik_diff import compute_tip_sensitivities_via_lm_adjoint, extract_terminal_p_dim

jax.config.update("jax_enable_x64", True)

PROTOCOL_VERSION = "p1a_reaudit_v2"
GOOD_CLASS = "GOOD"


def _runtime_environment() -> dict[str, Any]:
    devices = []
    for dev in jax.devices():
        devices.append(
            {
                "id": getattr(dev, "id", None),
                "platform": getattr(dev, "platform", None),
                "device_kind": getattr(dev, "device_kind", None),
                "repr": str(dev),
            }
        )
    return {
        "code_commit": current_git_commit(_REPO_ROOT),
        "git_tracked_dirty": git_tracked_dirty(_REPO_ROOT),
        "git_untracked_count": git_untracked_count(_REPO_ROOT),
        "python_executable": sys.executable,
        "jax_version": getattr(jax, "__version__", None),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_platforms_env": os.environ.get("JAX_PLATFORMS"),
        "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "devices": devices,
    }


@dataclass
class ReplayResult:
    """Replay output for one saved or perturbed control state."""

    z_bar: jnp.ndarray
    params: Any
    stats: Any
    terminal_m: np.ndarray
    centerline_m: np.ndarray
    ok: bool
    normE: float


def _parse_csv_float(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in str(text).split(",") if x.strip())


def _parse_csv_int(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(text).split(",") if x.strip())


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, sort_keys=True))
        f.write("\n")
        f.flush()


def _load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_manifest(point_dir: Path) -> list[dict[str, Any]]:
    manifest = point_dir / "branch_states" / "branch_state_manifest.jsonl"
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]


def _iter_point_dirs(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        path = path.resolve()
        if (path / "branch_states" / "branch_state_manifest.jsonl").exists():
            yield path
            continue
        for manifest in sorted(path.rglob("branch_state_manifest.jsonl")):
            yield manifest.parents[1]


def _select_manifest_row(rows: Sequence[dict[str, Any]], selector: str) -> dict[str, Any]:
    accepted = [row for row in rows if bool(row.get("accept"))]
    if selector == "last_accept":
        if accepted:
            return accepted[-1]
        finals = [row for row in rows if row.get("phase") == "final"]
        return finals[-1] if finals else rows[-1]
    if selector == "best_accept":
        pool = accepted or list(rows)
        return min(pool, key=lambda row: float(row.get("terminal_error_mm", float("inf"))))
    if selector == "final":
        finals = [row for row in rows if row.get("phase") == "final"]
        return finals[-1] if finals else rows[-1]
    if selector == "init":
        inits = [row for row in rows if row.get("phase") == "init"]
        return inits[0] if inits else rows[0]
    raise ValueError(f"unknown selector: {selector}")


def _build_engine_from_run_config(run_config: dict[str, Any]) -> ForwardKinematicsEngine:
    argv = dict(run_config["argv"])
    flex_lengths = fk_cli._parse_csv_floats(argv["flex_lengths"])
    rigid_lengths = fk_cli._parse_csv_floats(argv["rigid_lengths"])
    return ForwardKinematicsEngine(
        flex_lengths_tail=flex_lengths,
        rigid_lengths=rigid_lengths,
        M_list=fk_cli._parse_csv_ints(argv["M_list"]),
        L1_min=float(argv.get("L1_min", 1e-6)),
        L_protrude_max=float(argv.get("x_max") or argv.get("L_protrude")),
        flex_d_outer=fk_cli._parse_csv_floats(argv.get("flex_d_outer", "0.0015")),
        flex_E=fk_cli._parse_csv_floats(argv.get("flex_E", "1800000")),
        flex_G=fk_cli._parse_csv_floats(argv.get("flex_G", "600000")),
        flex_rho=fk_cli._parse_csv_floats(argv.get("flex_rho", "970")),
        rigid_d_outer=fk_cli._parse_csv_floats(argv.get("rigid_d_outer", "0.0015")),
        rigid_rho=fk_cli._parse_csv_floats(argv.get("rigid_rho", "7500")),
        p0_dim=jnp.asarray(_parse_csv_float(argv.get("p0", "0,0,0")), dtype=jnp.float64),
        Q0_wxyz=jnp.asarray(_parse_csv_float(argv.get("Q0", "1,0,0,0")), dtype=jnp.float64),
        axis_body=jnp.asarray(_parse_csv_float(argv.get("axis_body", "0,0,1")), dtype=jnp.float64),
        enable_gravity=bool(argv.get("enable_gravity", False)),
        g_world=jnp.asarray(_parse_csv_float(argv.get("g_world", "0,0,-9.81")), dtype=jnp.float64),
        enable_magnetics=bool(argv.get("enable_magnetics", False)),
        calib_file=argv.get("calib_file"),
        actuation_table_pkl=argv.get("actuation_table_pkl"),
        m_body_list=fk_cli._parse_list_vec3(argv["m_body_list"]) if argv.get("m_body_list") else None,
        max_iter=int(argv.get("max_iter", 100000)),
        tol=float(argv.get("tol", 1e-5)),
        lm_damping=float(argv.get("lm_damping", 0.1)),
        jac_method=str(argv.get("jac_method", "fwd")),
        scale_mode=str(argv.get("scale_mode", "optimized")),
    )


def _replay(
    engine: ForwardKinematicsEngine,
    *,
    I_A: np.ndarray,
    x_m: float,
    z0_bar: np.ndarray,
    include_jacobian: bool,
) -> ReplayResult:
    z_bar, params, _meshes, ok, stats = engine.solve_with_stats(
        coil_currents=jnp.asarray(I_A, dtype=jnp.float64),
        L_protrude=float(x_m),
        warm_start=True,
        override_z0_bar=jnp.asarray(z0_bar, dtype=jnp.float64),
        return_stats=True,
        stats_include_jacobian=bool(include_jacobian),
    )
    if hasattr(z_bar, "block_until_ready"):
        z_bar.block_until_ready()
    terminal = np.asarray(extract_terminal_p_dim(z_bar, M_list=params.M_list, scales=params.scales), dtype=float)
    centerline = np.asarray(extract_centerline_dim(z_bar, M_list=params.M_list, scales=params.scales), dtype=float)
    return ReplayResult(
        z_bar=jnp.asarray(z_bar, dtype=jnp.float64),
        params=params,
        stats=stats,
        terminal_m=terminal.reshape(3),
        centerline_m=centerline,
        ok=bool(ok),
        normE=float(getattr(stats, "normE", np.nan)),
    )


def _replay_segmented(
    engine: ForwardKinematicsEngine,
    *,
    I_start_A: np.ndarray,
    x_start_m: float,
    I_end_A: np.ndarray,
    x_end_m: float,
    z0_bar: np.ndarray,
    segments: int,
    include_jacobian: bool,
) -> ReplayResult:
    """Replay a control change through several warm-started substeps."""

    n_segments = max(1, int(segments))
    z_current = np.asarray(z0_bar, dtype=float)
    result: ReplayResult | None = None
    for idx in range(1, n_segments + 1):
        alpha = float(idx) / float(n_segments)
        I_step = np.asarray(I_start_A, dtype=float) + alpha * (np.asarray(I_end_A, dtype=float) - np.asarray(I_start_A, dtype=float))
        x_step = float(x_start_m) + alpha * (float(x_end_m) - float(x_start_m))
        result = _replay(
            engine,
            I_A=I_step,
            x_m=x_step,
            z0_bar=z_current,
            include_jacobian=bool(include_jacobian and idx == n_segments),
        )
        z_current = np.asarray(result.z_bar)
    if result is None:
        raise RuntimeError("segmented replay did not execute")
    return result


def _centerline_error_mm(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.linalg.norm(a - b, axis=1)) * 1000.0)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-15 or nb <= 1e-15:
        return float("nan")
    cosv = float(np.dot(a, b) / (na * nb))
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cosv)))))


def _apply_channel_level_stat_flags(rows: list[dict[str, Any]], weak_displacement_mm: float) -> None:
    """Set branch, weak-response, and statistics eligibility labels."""

    max_displacement: dict[tuple[Any, ...], float] = {}
    for row in rows:
        if not bool(row.get("fd_ok", False)):
            continue
        key = (
            row.get("point_dir"),
            row.get("eval_id"),
            row.get("selector"),
            row.get("kind"),
            row.get("channel"),
        )
        displacement = float(row.get("fd_displacement_mm", 0.0))
        max_displacement[key] = max(max_displacement.get(key, 0.0), displacement)

    for row in rows:
        if not bool(row.get("fd_ok", False)):
            continue
        key = (
            row.get("point_dir"),
            row.get("eval_id"),
            row.get("selector"),
            row.get("kind"),
            row.get("channel"),
        )
        channel_max_displacement_mm = float(max_displacement.get(key, 0.0))
        step_weak_raw = row.get("step_weak_gate_triggered", False)
        if isinstance(step_weak_raw, str):
            step_weak = step_weak_raw.strip().lower() == "true"
        else:
            step_weak = bool(step_weak_raw)
        weak_gate_triggered = bool(step_weak)
        branch_consistent = bool(row.get("roundtrip_ok", False))
        replay_converged = bool(row.get("replay_converged", False))
        used_in_stats = bool(branch_consistent and replay_converged and not weak_gate_triggered)
        if not branch_consistent or not replay_converged:
            stats_exclusion_reason = "branch_or_replay_failed"
        elif weak_gate_triggered:
            stats_exclusion_reason = "weak_response"
        else:
            stats_exclusion_reason = "ok"

        row["channel_max_displacement_mm"] = channel_max_displacement_mm
        row["weak_gate_triggered"] = weak_gate_triggered
        row["weak_channel"] = weak_gate_triggered
        row["branch_consistent"] = branch_consistent
        row["used_in_stats"] = used_in_stats
        row["stats_exclusion_reason"] = stats_exclusion_reason


def _finalize_fd_row(row: dict[str, Any], weak_displacement_mm: float) -> dict[str, Any]:
    """Populate finite-difference gate labels for one row."""

    if not bool(row.get("fd_ok", False)):
        row.setdefault("fd_valid", False)
        row.setdefault("replay_converged", False)
        row.setdefault("roundtrip_ok", False)
        row.setdefault("branch_consistent", False)
        row.setdefault("weak_gate_triggered", "")
        row.setdefault("weak_channel", "")
        row.setdefault("channel_max_displacement_mm", "")
        row.setdefault("used_in_stats", False)
        row.setdefault("stats_exclusion_reason", row.get("reason", "not_evaluated"))
        return row

    displacement = float(row.get("fd_displacement_mm", 0.0))
    step_weak = bool(displacement < float(weak_displacement_mm))
    branch_consistent = bool(row.get("roundtrip_ok", False))
    replay_converged = bool(row.get("replay_converged", False))
    used_in_stats = bool(branch_consistent and replay_converged and not step_weak)
    if not branch_consistent or not replay_converged:
        reason = "branch_or_replay_failed"
    elif step_weak:
        reason = "weak_response"
    else:
        reason = "ok"
    row["step_weak_gate_triggered"] = step_weak
    row["weak_gate_triggered"] = step_weak
    row["weak_channel"] = step_weak
    row["channel_max_displacement_mm"] = displacement
    row["branch_consistent"] = branch_consistent
    row["used_in_stats"] = used_in_stats
    row["stats_exclusion_reason"] = reason
    return row


def _current_channels_from_text(text: str | None) -> list[int] | None:
    if text is None or str(text).strip().lower() in ("", "auto"):
        return None
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _choose_channels(sensitivities: Sequence[Any], requested: list[int] | None, auto_n: int) -> list[int]:
    if requested is not None:
        return requested
    selected: list[int] = []
    for sens in sensitivities:
        norms = np.linalg.norm(np.asarray(sens.J_p_I, dtype=float), axis=0)
        for idx in list(np.argsort(-norms))[: max(0, int(auto_n))]:
            idx = int(idx)
            if idx not in selected:
                selected.append(idx)
    return selected


def _point_label(point_dir: Path) -> str:
    parts = [point_dir.parent.parent.name, point_dir.name]
    return "_".join(str(x).replace("/", "_") for x in parts if x)


def _row_get(row: dict[str, Any], key: str) -> str:
    value = row.get(key, "")
    return "" if value is None else str(value)


def _as_bool(value: Any) -> bool:
    """Parse booleans written directly or reloaded from CSV."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y")
    return bool(value)


def _sens_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        _row_get(row, "protocol_version"),
        _row_get(row, "point_dir"),
        _row_get(row, "eval_id"),
        _row_get(row, "selector"),
        _row_get(row, "mode"),
        _row_get(row, "solve_mode"),
    )


def _fd_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        _row_get(row, "protocol_version"),
        _row_get(row, "point_dir"),
        _row_get(row, "eval_id"),
        _row_get(row, "selector"),
        _row_get(row, "kind"),
        _row_get(row, "channel"),
        _row_get(row, "step"),
        _row_get(row, "mode"),
        _row_get(row, "roundtrip_segments"),
    )


class AuditRecorder:
    """Persist audit rows and run-package status incrementally."""

    def __init__(self, out_dir: Path, *, protocol_version: str, resume: bool) -> None:
        self.out_dir = Path(out_dir)
        self.protocol_version = str(protocol_version)
        self.sensitivity_path = self.out_dir / "sensitivity_modes.csv"
        self.fd_path = self.out_dir / "finite_difference_audit.csv"
        self.progress_path = self.out_dir / "progress.jsonl"
        self.completed_path = self.out_dir / "completed_points.csv"
        self.failures_dir = self.out_dir / "failures"
        self.sens_rows = _load_csv(self.sensitivity_path) if resume else []
        self.fd_rows = _load_csv(self.fd_path) if resume else []
        self.completed_rows = _load_csv(self.completed_path) if resume else []
        self.sens_keys = {_sens_key(row) for row in self.sens_rows}
        self.fd_keys = {_fd_key(row) for row in self.fd_rows}

    def point_is_complete(self, point_dir: Path) -> bool:
        """Return whether a point already has a terminal package status."""
        point_text = str(point_dir)
        complete_statuses = {"success", "light_diagnostic"}
        return any(row.get("point_dir") == point_text and row.get("status") in complete_statuses for row in self.completed_rows)

    def fd_exists(
        self,
        *,
        point_dir: Path,
        eval_id: Any,
        selector: str,
        kind: str,
        channel: Any,
        step: Any,
        mode: str,
        roundtrip_segments: int,
    ) -> bool:
        """Return whether a finite-difference task key has already been stored."""
        row = {
            "protocol_version": self.protocol_version,
            "point_dir": str(point_dir),
            "eval_id": str(eval_id),
            "selector": str(selector),
            "kind": str(kind),
            "channel": str(channel),
            "step": str(step),
            "mode": str(mode),
            "roundtrip_segments": str(roundtrip_segments),
        }
        return _fd_key(row) in self.fd_keys

    def heartbeat(self, event: str, **payload: Any) -> None:
        item = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "event": str(event)}
        item.update(payload)
        _append_jsonl(self.progress_path, item)

    def add_sens(self, row: dict[str, Any]) -> bool:
        row = dict(row)
        row.setdefault("protocol_version", self.protocol_version)
        key = _sens_key(row)
        if key in self.sens_keys:
            return False
        self.sens_keys.add(key)
        self.sens_rows.append(row)
        _write_csv(self.sensitivity_path, self.sens_rows)
        return True

    def add_fd(self, row: dict[str, Any]) -> bool:
        row = dict(row)
        row.setdefault("protocol_version", self.protocol_version)
        key = _fd_key(row)
        if key in self.fd_keys:
            return False
        self.fd_keys.add(key)
        self.fd_rows.append(row)
        _write_csv(self.fd_path, self.fd_rows)
        return True

    def point_done(self, point_dir: Path, *, status: str, reason: str, elapsed_s: float) -> None:
        row = {
            "point_dir": str(point_dir),
            "point_label": _point_label(point_dir),
            "status": str(status),
            "reason": str(reason),
            "elapsed_s": float(elapsed_s),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "protocol_version": self.protocol_version,
        }
        self.completed_rows = [r for r in self.completed_rows if r.get("point_dir") != str(point_dir)]
        self.completed_rows.append(row)
        _write_csv(self.completed_path, self.completed_rows)
        if status not in ("success", "light_diagnostic"):
            self.failures_dir.mkdir(parents=True, exist_ok=True)
            _write_json(self.failures_dir / f"{_point_label(point_dir)}.json", row)

    def summary(self) -> dict[str, Any]:
        return {
            "sensitivity_rows": len(self.sens_rows),
            "fd_rows": len(self.fd_rows),
            "fd_gate_summary": _fd_gate_summary(self.fd_rows),
            "completed_points": len(self.completed_rows),
        }


def _audit_one_point(
    point_dir: Path,
    *,
    out_dir: Path,
    recorder: AuditRecorder | None,
    selector: str,
    current_steps_a: Sequence[float],
    x_steps_m: Sequence[float],
    current_channels: list[int] | None,
    auto_current_channels: int,
    sensitivity_modes: Sequence[str],
    solve_mode: str,
    ridge: float,
    svd_rcond: float,
    spectral_filter: str,
    spectral_damping: float,
    roundtrip_segments: int,
    terminal_gate_mm: float,
    centerline_gate_mm: float,
    weak_displacement_mm: float,
    protocol_version: str,
    health_gate: str,
    point_timeout_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    point_started = time.perf_counter()
    run_config = _load_json(point_dir / "run_config.json")
    rows = _load_manifest(point_dir)
    base_row = _select_manifest_row(rows, selector)
    state_root = point_dir / "branch_states"
    base_npz = np.load(state_root / base_row["npz_path"])
    z_base_saved = np.asarray(base_npz["z_star_bar"], dtype=float)
    I_base = np.asarray(base_npz["I_A"], dtype=float).reshape(-1)
    x_base = float(np.asarray(base_npz["x_m"], dtype=float).reshape(-1)[0])
    terminal_saved = np.asarray(base_npz["terminal_true_m"], dtype=float).reshape(3)
    flex_tip_saved = np.asarray(base_npz["flex_tip_m"], dtype=float).reshape(3) if "flex_tip_m" in base_npz else np.full(3, np.nan)
    centerline_saved = np.asarray(base_npz["centerline_m"], dtype=float)
    target_m = np.asarray(base_npz["target_m"], dtype=float).reshape(3)
    control_hash = array_sha256(np.concatenate([I_base.reshape(-1), np.asarray([x_base], dtype=float)]))
    true_flex_offset_mm = float(np.linalg.norm(terminal_saved - flex_tip_saved) * 1000.0)

    engine = _build_engine_from_run_config(run_config)
    base = _replay(engine, I_A=I_base, x_m=x_base, z0_bar=z_base_saved, include_jacobian=True)
    terminal_reload_mm = float(np.linalg.norm(base.terminal_m - terminal_saved) * 1000.0)
    centerline_reload_mm = _centerline_error_mm(base.centerline_m, centerline_saved)
    base_classification = str(base_row.get("classification", ""))
    health_reasons: list[str] = []
    if not bool(base_row.get("accept")):
        health_reasons.append("not_accepted")
    if base_classification != GOOD_CLASS:
        health_reasons.append(f"classification_{base_classification or 'missing'}")
    if not bool(base.ok):
        health_reasons.append("base_replay_not_strict")
    if terminal_reload_mm > float(terminal_gate_mm):
        health_reasons.append("terminal_reload_gate_failed")
    if centerline_reload_mm > float(centerline_gate_mm):
        health_reasons.append("centerline_reload_gate_failed")
    health_gate_passed = not health_reasons
    health_gate_mode = str(health_gate)
    if recorder is not None:
        recorder.heartbeat(
            "point_base_replayed",
            point_dir=str(point_dir),
            eval_id=base_row.get("eval_id"),
            classification=base_classification,
            health_gate_passed=health_gate_passed,
            health_reasons=health_reasons,
        )

    sens_rows: list[dict[str, Any]] = []
    fd_rows: list[dict[str, Any]] = []
    sens_by_mode: dict[str, Any] = {}
    for mode in sensitivity_modes:
        t0 = time.perf_counter()
        try:
            sens = compute_tip_sensitivities_via_lm_adjoint(
                z_star_bar=base.z_bar,
                params=base.params,
                lm_stats=base.stats,
                coil_currents=jnp.asarray(I_base, dtype=jnp.float64),
                ridge=float(ridge),
                sensitivity_mode=str(mode),
                solve_mode=str(solve_mode),
                svd_rcond=float(svd_rcond),
                spectral_filter=str(spectral_filter),
                spectral_damping=float(spectral_damping),
                compute_conditioning=True,
            )
            elapsed = time.perf_counter() - t0
            sens_by_mode[str(mode)] = sens
            sens_rows.append(
                {
                    "point_dir": str(point_dir),
                    "eval_id": base_row["eval_id"],
                    "protocol_version": str(protocol_version),
                    "selector": str(selector),
                    "phase": str(base_row.get("phase", "")),
                    "accept": bool(base_row.get("accept")),
                    "base_classification": base_classification,
                    "base_replay_ok": bool(base.ok),
                    "health_gate_mode": health_gate_mode,
                    "health_gate_passed": bool(health_gate_passed),
                    "health_gate_reasons": ";".join(health_reasons),
                    "z_star_hash": str(base_row.get("z_star_hash", "")),
                    "warm_start_hash": str(base_row.get("warm_start_hash", "")),
                    "centerline_hash": str(base_row.get("centerline_hash", "")),
                    "control_hash": control_hash,
                    "mode": str(mode),
                    "solve_mode": str(solve_mode),
                    "ridge": float(ridge),
                    "svd_rcond": float(svd_rcond),
                    "ok": True,
                    "status": sens.status,
                    "method": sens.method,
                    "elapsed_s": elapsed,
                    "effective_rank": sens.effective_rank,
                    "condition_number": sens.condition_number,
                    "sv_min": sens.sv_min,
                    "sv_max": sens.sv_max,
                    "spectral_filter": sens.spectral_filter,
                    "spectral_damping": float(spectral_damping),
                    "spectral_lambda": sens.spectral_lambda,
                    "spectral_cutoff": sens.spectral_cutoff,
                    "retained_spectral_energy": sens.retained_spectral_energy,
                    "stationarity_norm": sens.stationarity_norm,
                    "terminal_reload_error_mm": terminal_reload_mm,
                    "centerline_reload_max_error_mm": centerline_reload_mm,
                    "true_flex_offset_mm": true_flex_offset_mm,
                    "base_err_mm": float(np.linalg.norm(base.terminal_m - target_m) * 1000.0),
                }
            )
            if recorder is not None:
                recorder.add_sens(sens_rows[-1])
        except Exception as exc:
            sens_rows.append(
                {
                    "point_dir": str(point_dir),
                    "eval_id": base_row["eval_id"],
                    "protocol_version": str(protocol_version),
                    "selector": str(selector),
                    "phase": str(base_row.get("phase", "")),
                    "accept": bool(base_row.get("accept")),
                    "base_classification": base_classification,
                    "base_replay_ok": bool(base.ok),
                    "health_gate_mode": health_gate_mode,
                    "health_gate_passed": bool(health_gate_passed),
                    "health_gate_reasons": ";".join(health_reasons),
                    "z_star_hash": str(base_row.get("z_star_hash", "")),
                    "warm_start_hash": str(base_row.get("warm_start_hash", "")),
                    "centerline_hash": str(base_row.get("centerline_hash", "")),
                    "control_hash": control_hash,
                    "mode": str(mode),
                    "solve_mode": str(solve_mode),
                    "ridge": float(ridge),
                    "svd_rcond": float(svd_rcond),
                    "ok": False,
                    "status": "error",
                    "method": "",
                    "elapsed_s": time.perf_counter() - t0,
                    "effective_rank": "",
                    "condition_number": "",
                    "sv_min": "",
                    "sv_max": "",
                    "spectral_filter": str(spectral_filter),
                    "spectral_damping": float(spectral_damping),
                    "spectral_lambda": "",
                    "spectral_cutoff": "",
                    "retained_spectral_energy": "",
                    "stationarity_norm": "",
                    "terminal_reload_error_mm": terminal_reload_mm,
                    "centerline_reload_max_error_mm": centerline_reload_mm,
                    "true_flex_offset_mm": true_flex_offset_mm,
                    "base_err_mm": float(np.linalg.norm(base.terminal_m - target_m) * 1000.0),
                    "error": repr(exc),
                }
            )
            if recorder is not None:
                recorder.add_sens(sens_rows[-1])

    if not sens_by_mode:
        return sens_rows, fd_rows

    if health_gate_mode == "skip_full_fd" and not health_gate_passed:
        row = _finalize_fd_row(
            {
                "point_dir": str(point_dir),
                "eval_id": base_row["eval_id"],
                "protocol_version": str(protocol_version),
                "selector": str(selector),
                "phase": str(base_row.get("phase", "")),
                "accept": bool(base_row.get("accept")),
                "kind": "health_gate",
                "channel": "",
                "step": "",
                "step_family": "health_gate",
                "mode": "",
                "roundtrip_segments": int(roundtrip_segments),
                "base_classification": base_classification,
                "base_replay_ok": bool(base.ok),
                "health_gate_passed": False,
                "health_gate_reasons": ";".join(health_reasons),
                "fd_ok": False,
                "fd_valid": False,
                "reason": "health_gate_failed",
            },
            weak_displacement_mm,
        )
        fd_rows.append(row)
        if recorder is not None:
            recorder.add_fd(row)
        return sens_rows, fd_rows

    channels = _choose_channels(list(sens_by_mode.values()), current_channels, auto_current_channels)
    x_min = float(run_config["argv"].get("x_min") or -float("inf"))
    x_max = float(run_config["argv"].get("x_max") or float("inf"))
    I_max = float(run_config.get("I_max", run_config["argv"].get("I_max", 20.0)))
    I_min = float(run_config.get("I_min", -I_max))

    perturb_specs: list[tuple[str, int, float]] = []
    for channel in channels:
        for step in current_steps_a:
            perturb_specs.append(("I", int(channel), float(step)))
    for step in x_steps_m:
        perturb_specs.append(("x", -1, float(step)))

    for kind, channel, step in perturb_specs:
        if recorder is not None and sens_by_mode:
            if all(
                recorder.fd_exists(
                    point_dir=point_dir,
                    eval_id=base_row["eval_id"],
                    selector=str(selector),
                    kind=kind,
                    channel=channel,
                    step=step,
                    mode=mode,
                    roundtrip_segments=int(roundtrip_segments),
                )
                for mode in sens_by_mode
            ):
                recorder.heartbeat("perturbation_skipped_existing", point_dir=str(point_dir), kind=kind, channel=channel, step=step)
                continue
        if point_timeout_s > 0 and (time.perf_counter() - point_started) > float(point_timeout_s):
            row = _finalize_fd_row(
                {
                    "point_dir": str(point_dir),
                    "eval_id": base_row["eval_id"],
                    "protocol_version": str(protocol_version),
                    "selector": str(selector),
                    "phase": str(base_row.get("phase", "")),
                    "accept": bool(base_row.get("accept")),
                    "kind": kind,
                    "channel": channel,
                    "step": step,
                    "step_family": f"{kind}:{channel}",
                    "mode": "",
                    "roundtrip_segments": int(roundtrip_segments),
                    "base_classification": base_classification,
                    "health_gate_passed": bool(health_gate_passed),
                    "health_gate_reasons": ";".join(health_reasons),
                    "fd_ok": False,
                    "fd_valid": False,
                    "reason": "point_timeout",
                },
                weak_displacement_mm,
            )
            fd_rows.append(row)
            if recorder is not None:
                recorder.add_fd(row)
                recorder.heartbeat("point_timeout", point_dir=str(point_dir), elapsed_s=time.perf_counter() - point_started)
            break
        if recorder is not None:
            recorder.heartbeat("perturbation_started", point_dir=str(point_dir), kind=kind, channel=channel, step=step)
        I_plus = I_base.copy()
        I_minus = I_base.copy()
        x_plus = x_base
        x_minus = x_base
        if kind == "I":
            I_plus[channel] += step
            I_minus[channel] -= step
            bounds_ok = bool(I_min <= I_plus[channel] <= I_max and I_min <= I_minus[channel] <= I_max)
        else:
            x_plus += step
            x_minus -= step
            bounds_ok = bool(x_min <= x_plus <= x_max and x_min <= x_minus <= x_max)
        if not bounds_ok:
            fd_rows.append(
                _finalize_fd_row(
                    {
                    "point_dir": str(point_dir),
                    "eval_id": base_row["eval_id"],
                    "protocol_version": str(protocol_version),
                    "selector": str(selector),
                    "phase": str(base_row.get("phase", "")),
                    "accept": bool(base_row.get("accept")),
                    "kind": kind,
                    "channel": channel,
                    "step": step,
                    "step_family": f"{kind}:{channel}",
                    "displacement_gate_mm": float(weak_displacement_mm),
                    "fd_ok": False,
                    "fd_valid": False,
                    "replay_converged": False,
                    "roundtrip_ok": False,
                    "branch_consistent": False,
                    "step_weak_gate_triggered": "",
                    "weak_gate_triggered": "",
                    "weak_channel": "",
                    "channel_max_displacement_mm": "",
                    "used_in_stats": False,
                    "stats_exclusion_reason": "bounds",
                    "reason": "bounds",
                    },
                    weak_displacement_mm,
                )
            )
            if recorder is not None:
                recorder.add_fd(fd_rows[-1])
            continue

        plus = _replay_segmented(
            engine,
            I_start_A=I_base,
            x_start_m=x_base,
            I_end_A=I_plus,
            x_end_m=x_plus,
            z0_bar=np.asarray(base.z_bar),
            segments=roundtrip_segments,
            include_jacobian=False,
        )
        minus = _replay_segmented(
            engine,
            I_start_A=I_base,
            x_start_m=x_base,
            I_end_A=I_minus,
            x_end_m=x_minus,
            z0_bar=np.asarray(base.z_bar),
            segments=roundtrip_segments,
            include_jacobian=False,
        )
        back_plus = _replay_segmented(
            engine,
            I_start_A=I_plus,
            x_start_m=x_plus,
            I_end_A=I_base,
            x_end_m=x_base,
            z0_bar=np.asarray(plus.z_bar),
            segments=roundtrip_segments,
            include_jacobian=False,
        )
        back_minus = _replay_segmented(
            engine,
            I_start_A=I_minus,
            x_start_m=x_minus,
            I_end_A=I_base,
            x_end_m=x_base,
            z0_bar=np.asarray(minus.z_bar),
            segments=roundtrip_segments,
            include_jacobian=False,
        )
        fd_col = (plus.terminal_m - minus.terminal_m) / (2.0 * step)
        displacement_mm = float(np.linalg.norm(fd_col * step) * 1000.0)
        step_weak = bool(displacement_mm < weak_displacement_mm)
        back_plus_terminal_mm = float(np.linalg.norm(back_plus.terminal_m - base.terminal_m) * 1000.0)
        back_minus_terminal_mm = float(np.linalg.norm(back_minus.terminal_m - base.terminal_m) * 1000.0)
        back_plus_centerline_mm = _centerline_error_mm(back_plus.centerline_m, base.centerline_m)
        back_minus_centerline_mm = _centerline_error_mm(back_minus.centerline_m, base.centerline_m)
        replay_converged = bool(plus.ok and minus.ok and back_plus.ok and back_minus.ok)
        roundtrip_ok = bool(
            back_plus_terminal_mm <= terminal_gate_mm
            and back_minus_terminal_mm <= terminal_gate_mm
            and back_plus_centerline_mm <= centerline_gate_mm
            and back_minus_centerline_mm <= centerline_gate_mm
        )

        fd_strict_ok = bool(roundtrip_ok and replay_converged)
        for mode, sens in sens_by_mode.items():
            jac_col = np.asarray(sens.J_p_I[:, channel] if kind == "I" else sens.J_p_x, dtype=float).reshape(3)
            abs_err_mm_per_unit = float(np.linalg.norm(jac_col - fd_col) * 1000.0)
            fd_norm = float(np.linalg.norm(fd_col))
            jac_norm = float(np.linalg.norm(jac_col))
            rel_err = float(np.linalg.norm(jac_col - fd_col) / max(np.linalg.norm(fd_col), 1e-12))
            fd_rows.append(
                _finalize_fd_row(
                    {
                    "point_dir": str(point_dir),
                    "eval_id": base_row["eval_id"],
                    "protocol_version": str(protocol_version),
                    "selector": str(selector),
                    "phase": str(base_row.get("phase", "")),
                    "accept": bool(base_row.get("accept")),
                    "base_classification": base_classification,
                    "base_replay_ok": bool(base.ok),
                    "health_gate_passed": bool(health_gate_passed),
                    "health_gate_reasons": ";".join(health_reasons),
                    "z_star_hash": str(base_row.get("z_star_hash", "")),
                    "warm_start_hash": str(base_row.get("warm_start_hash", "")),
                    "centerline_hash": str(base_row.get("centerline_hash", "")),
                    "control_hash": control_hash,
                    "kind": kind,
                    "channel": channel,
                    "step": step,
                    "step_family": f"{kind}:{channel}",
                    "mode": mode,
                    "solve_mode": str(solve_mode),
                    "ridge": float(ridge),
                    "svd_rcond": float(svd_rcond),
                    "spectral_filter": getattr(sens, "spectral_filter", "none"),
                    "spectral_damping": float(spectral_damping),
                    "roundtrip_segments": int(roundtrip_segments),
                    "spectral_lambda": getattr(sens, "spectral_lambda", ""),
                    "spectral_cutoff": getattr(sens, "spectral_cutoff", ""),
                    "retained_spectral_energy": getattr(sens, "retained_spectral_energy", ""),
                    "fd_ok": True,
                    "fd_valid": fd_strict_ok,
                    "replay_converged": replay_converged,
                    "roundtrip_ok": roundtrip_ok,
                    "branch_consistent": roundtrip_ok,
                    "displacement_gate_mm": float(weak_displacement_mm),
                    "step_weak_gate_triggered": step_weak,
                    "weak_gate_triggered": "",
                    "weak_channel": "",
                    "channel_max_displacement_mm": "",
                    "used_in_stats": "",
                    "stats_exclusion_reason": "",
                    "fd_displacement_mm": displacement_mm,
                    "fd_norm_per_unit": fd_norm,
                    "jac_norm_per_unit": jac_norm,
                    "norm_ratio": float(jac_norm / max(fd_norm, 1e-12)),
                    "relative_error": rel_err,
                    "abs_error_mm_per_unit": abs_err_mm_per_unit,
                    "angle_deg": _angle_deg(jac_col, fd_col),
                    "plus_normE": plus.normE,
                    "minus_normE": minus.normE,
                    "back_plus_terminal_mm": back_plus_terminal_mm,
                    "back_minus_terminal_mm": back_minus_terminal_mm,
                    "back_plus_centerline_mm": back_plus_centerline_mm,
                    "back_minus_centerline_mm": back_minus_centerline_mm,
                    "base_terminal_reload_mm": terminal_reload_mm,
                    "base_centerline_reload_mm": centerline_reload_mm,
                    },
                    weak_displacement_mm,
                )
            )
            if recorder is not None:
                recorder.add_fd(fd_rows[-1])

    _apply_channel_level_stat_flags(fd_rows, weak_displacement_mm)
    return sens_rows, fd_rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fd_gate_summary(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Summarize finite-difference audit gate counts."""

    return {
        "fd_ok": sum(_as_bool(row.get("fd_ok")) for row in rows),
        "replay_converged": sum(_as_bool(row.get("replay_converged")) for row in rows),
        "branch_consistent": sum(_as_bool(row.get("branch_consistent")) for row in rows),
        "weak_gate_triggered": sum(_as_bool(row.get("weak_gate_triggered")) for row in rows),
        "used_in_stats": sum(_as_bool(row.get("used_in_stats")) for row in rows),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Point directories or parent experiment directories.")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--protocol-version", type=str, default=PROTOCOL_VERSION)
    parser.add_argument("--resume", action="store_true", help="Reuse completed rows in the output directory.")
    parser.add_argument("--selector", choices=("last_accept", "best_accept", "final", "init"), default="best_accept")
    parser.add_argument("--current-steps-a", type=str, default="0.25")
    parser.add_argument("--x-steps-mm", type=str, default="0.5")
    parser.add_argument("--current-channels", type=str, default="auto")
    parser.add_argument("--auto-current-channels", type=int, default=2)
    parser.add_argument(
        "--sensitivity-modes",
        type=str,
        default="gn_svd",
    )
    parser.add_argument("--solve-mode", choices=("solve", "svd"), default="svd")
    parser.add_argument("--ridge", type=float, default=1e-10)
    parser.add_argument("--svd-rcond", type=float, default=1e-9)
    parser.add_argument("--spectral-filter", choices=("hard", "tikhonov", "hybrid"), default="hard")
    parser.add_argument("--spectral-damping", type=float, default=0.0)
    parser.add_argument("--roundtrip-segments", type=int, default=1, help="Number of warm-started substeps used for perturbation and roundtrip replay.")
    parser.add_argument("--terminal-gate-mm", type=float, default=0.25)
    parser.add_argument("--centerline-gate-mm", type=float, default=0.5)
    parser.add_argument("--weak-displacement-mm", type=float, default=0.25)
    parser.add_argument("--health-gate", choices=("off", "warn", "skip_full_fd"), default="skip_full_fd")
    parser.add_argument("--point-timeout-s", type=float, default=0.0)
    parser.add_argument("--max-points", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    recorder = AuditRecorder(out_dir, protocol_version=str(args.protocol_version), resume=bool(args.resume))
    runtime_environment = _runtime_environment()
    _write_json(out_dir / "runtime_environment.json", runtime_environment)
    point_dirs = list(_iter_point_dirs(Path(p) for p in args.paths))
    if args.max_points and args.max_points > 0:
        point_dirs = point_dirs[: int(args.max_points)]
    planned_rows = [
        {
            "point_dir": str(point_dir),
            "point_label": _point_label(point_dir),
            "order": idx,
            "protocol_version": str(args.protocol_version),
        }
        for idx, point_dir in enumerate(point_dirs)
    ]
    _write_csv(out_dir / "planned_points.csv", planned_rows)
    _write_json(
        out_dir / "run_status.json",
        {
            "status": "running",
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "protocol_version": str(args.protocol_version),
            "settings": vars(args),
            "runtime_environment": runtime_environment,
            "planned_point_count": len(point_dirs),
        },
    )
    recorder.heartbeat("run_started", planned_point_count=len(point_dirs))
    current_steps = _parse_csv_float(args.current_steps_a)
    x_steps = tuple(v / 1000.0 for v in _parse_csv_float(args.x_steps_mm))
    modes = [m.strip() for m in args.sensitivity_modes.split(",") if m.strip()]
    requested_channels = _current_channels_from_text(args.current_channels)

    all_sens_rows: list[dict[str, Any]] = list(recorder.sens_rows)
    all_fd_rows: list[dict[str, Any]] = list(recorder.fd_rows)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    for point_dir in point_dirs:
        point_started = time.perf_counter()
        print(f"[reaudit] point={point_dir}", flush=True)
        if bool(args.resume) and recorder.point_is_complete(point_dir):
            recorder.heartbeat("point_skipped_existing", point_dir=str(point_dir))
            continue
        recorder.heartbeat("point_started", point_dir=str(point_dir))
        try:
            sens_rows, fd_rows = _audit_one_point(
                point_dir,
                out_dir=out_dir,
                recorder=recorder,
                selector=str(args.selector),
                current_steps_a=current_steps,
                x_steps_m=x_steps,
                current_channels=requested_channels,
                auto_current_channels=int(args.auto_current_channels),
                sensitivity_modes=modes,
                solve_mode=str(args.solve_mode),
                ridge=float(args.ridge),
                svd_rcond=float(args.svd_rcond),
                spectral_filter=str(args.spectral_filter),
                spectral_damping=float(args.spectral_damping),
                roundtrip_segments=int(args.roundtrip_segments),
                terminal_gate_mm=float(args.terminal_gate_mm),
                centerline_gate_mm=float(args.centerline_gate_mm),
                weak_displacement_mm=float(args.weak_displacement_mm),
                protocol_version=str(args.protocol_version),
                health_gate=str(args.health_gate),
                point_timeout_s=float(args.point_timeout_s),
            )
            all_sens_rows.extend(sens_rows)
            all_fd_rows.extend(fd_rows)
            if any(row.get("reason") == "health_gate_failed" for row in fd_rows):
                point_status = "light_diagnostic"
                reason = "health_gate_failed"
            elif any(row.get("reason") == "point_timeout" for row in fd_rows):
                point_status = "timeout"
                reason = "point_timeout"
            else:
                point_status = "success"
                reason = "completed"
            recorder.point_done(point_dir, status=point_status, reason=reason, elapsed_s=time.perf_counter() - point_started)
            recorder.heartbeat("point_finished", point_dir=str(point_dir), status=point_status, reason=reason)
        except Exception as exc:
            recorder.point_done(point_dir, status="failed", reason=repr(exc), elapsed_s=time.perf_counter() - point_started)
            recorder.heartbeat("point_failed", point_dir=str(point_dir), error=repr(exc))
            print(f"[reaudit] point failed: {point_dir}: {exc!r}", flush=True)

    summary = {
        "started": started,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "point_count": len(point_dirs),
        "sensitivity_rows": len(recorder.sens_rows),
        "fd_rows": len(recorder.fd_rows),
        "fd_gate_summary": _fd_gate_summary(recorder.fd_rows),
        "settings": vars(args),
        "runtime_environment": runtime_environment,
        "paths": [str(p) for p in point_dirs],
        "protocol_version": str(args.protocol_version),
        "completed_points": len(recorder.completed_rows),
    }
    _write_json(out_dir / "summary.json", summary)
    _write_csv(out_dir / "sensitivity_modes.csv", recorder.sens_rows)
    _write_csv(out_dir / "finite_difference_audit.csv", recorder.fd_rows)
    _write_json(
        out_dir / "run_status.json",
        {
            "status": "finished",
            "started": started,
            "finished": summary["finished"],
            "protocol_version": str(args.protocol_version),
            "planned_point_count": len(point_dirs),
            "completed_points": len(recorder.completed_rows),
            "summary": summary,
        },
    )
    recorder.heartbeat("run_finished", completed_points=len(recorder.completed_rows))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
