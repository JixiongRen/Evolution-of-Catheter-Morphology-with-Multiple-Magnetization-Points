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
from inverse_kinematics.ik_branch_state import current_git_commit
from inverse_kinematics.ik_diff import compute_tip_sensitivities_via_lm_adjoint, extract_terminal_p_dim

jax.config.update("jax_enable_x64", True)


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


def _apply_channel_level_weak_flags(rows: list[dict[str, Any]], weak_displacement_mm: float) -> None:
    """Set weak-response labels consistently for each audited control channel."""

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
        row["weak_channel"] = bool(max_displacement.get(key, 0.0) < float(weak_displacement_mm))


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


def _audit_one_point(
    point_dir: Path,
    *,
    out_dir: Path,
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_config = _load_json(point_dir / "run_config.json")
    rows = _load_manifest(point_dir)
    base_row = _select_manifest_row(rows, selector)
    state_root = point_dir / "branch_states"
    base_npz = np.load(state_root / base_row["npz_path"])
    z_base_saved = np.asarray(base_npz["z_star_bar"], dtype=float)
    I_base = np.asarray(base_npz["I_A"], dtype=float).reshape(-1)
    x_base = float(np.asarray(base_npz["x_m"], dtype=float).reshape(-1)[0])
    terminal_saved = np.asarray(base_npz["terminal_true_m"], dtype=float).reshape(3)
    centerline_saved = np.asarray(base_npz["centerline_m"], dtype=float)
    target_m = np.asarray(base_npz["target_m"], dtype=float).reshape(3)

    engine = _build_engine_from_run_config(run_config)
    base = _replay(engine, I_A=I_base, x_m=x_base, z0_bar=z_base_saved, include_jacobian=True)
    terminal_reload_mm = float(np.linalg.norm(base.terminal_m - terminal_saved) * 1000.0)
    centerline_reload_mm = _centerline_error_mm(base.centerline_m, centerline_saved)

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
                    "selector": str(selector),
                    "phase": str(base_row.get("phase", "")),
                    "accept": bool(base_row.get("accept")),
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
                    "base_err_mm": float(np.linalg.norm(base.terminal_m - target_m) * 1000.0),
                }
            )
        except Exception as exc:
            sens_rows.append(
                {
                    "point_dir": str(point_dir),
                    "eval_id": base_row["eval_id"],
                    "selector": str(selector),
                    "phase": str(base_row.get("phase", "")),
                    "accept": bool(base_row.get("accept")),
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
                    "base_err_mm": float(np.linalg.norm(base.terminal_m - target_m) * 1000.0),
                    "error": repr(exc),
                }
            )

    if not sens_by_mode:
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
                {
                    "point_dir": str(point_dir),
                    "eval_id": base_row["eval_id"],
                    "selector": str(selector),
                    "phase": str(base_row.get("phase", "")),
                    "accept": bool(base_row.get("accept")),
                    "kind": kind,
                    "channel": channel,
                    "step": step,
                    "fd_ok": False,
                    "reason": "bounds",
                }
            )
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
        weak = bool(displacement_mm < weak_displacement_mm)
        back_plus_terminal_mm = float(np.linalg.norm(back_plus.terminal_m - base.terminal_m) * 1000.0)
        back_minus_terminal_mm = float(np.linalg.norm(back_minus.terminal_m - base.terminal_m) * 1000.0)
        back_plus_centerline_mm = _centerline_error_mm(back_plus.centerline_m, base.centerline_m)
        back_minus_centerline_mm = _centerline_error_mm(back_minus.centerline_m, base.centerline_m)
        roundtrip_ok = bool(
            back_plus_terminal_mm <= terminal_gate_mm
            and back_minus_terminal_mm <= terminal_gate_mm
            and back_plus_centerline_mm <= centerline_gate_mm
            and back_minus_centerline_mm <= centerline_gate_mm
        )

        fd_strict_ok = bool(roundtrip_ok and plus.ok and minus.ok and back_plus.ok and back_minus.ok)
        for mode, sens in sens_by_mode.items():
            jac_col = np.asarray(sens.J_p_I[:, channel] if kind == "I" else sens.J_p_x, dtype=float).reshape(3)
            abs_err_mm_per_unit = float(np.linalg.norm(jac_col - fd_col) * 1000.0)
            fd_norm = float(np.linalg.norm(fd_col))
            jac_norm = float(np.linalg.norm(jac_col))
            rel_err = float(np.linalg.norm(jac_col - fd_col) / max(np.linalg.norm(fd_col), 1e-12))
            fd_rows.append(
                {
                    "point_dir": str(point_dir),
                    "eval_id": base_row["eval_id"],
                    "selector": str(selector),
                    "phase": str(base_row.get("phase", "")),
                    "accept": bool(base_row.get("accept")),
                    "kind": kind,
                    "channel": channel,
                    "step": step,
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
                    "roundtrip_ok": roundtrip_ok,
                    "weak_channel": weak,
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
                }
            )

    _apply_channel_level_weak_flags(fd_rows, weak_displacement_mm)
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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Point directories or parent experiment directories.")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--selector", choices=("last_accept", "best_accept", "final", "init"), default="best_accept")
    parser.add_argument("--current-steps-a", type=str, default="0.25")
    parser.add_argument("--x-steps-mm", type=str, default="0.5")
    parser.add_argument("--current-channels", type=str, default="auto")
    parser.add_argument("--auto-current-channels", type=int, default=2)
    parser.add_argument(
        "--sensitivity-modes",
        type=str,
        default="solver_response,gn_svd,stationarity_exact,residual_svd",
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
    parser.add_argument("--max-points", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_environment = _runtime_environment()
    _write_json(out_dir / "runtime_environment.json", runtime_environment)
    point_dirs = list(_iter_point_dirs(Path(p) for p in args.paths))
    if args.max_points and args.max_points > 0:
        point_dirs = point_dirs[: int(args.max_points)]
    current_steps = _parse_csv_float(args.current_steps_a)
    x_steps = tuple(v / 1000.0 for v in _parse_csv_float(args.x_steps_mm))
    modes = [m.strip() for m in args.sensitivity_modes.split(",") if m.strip()]
    requested_channels = _current_channels_from_text(args.current_channels)

    all_sens_rows: list[dict[str, Any]] = []
    all_fd_rows: list[dict[str, Any]] = []
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    for point_dir in point_dirs:
        print(f"[reaudit] point={point_dir}", flush=True)
        sens_rows, fd_rows = _audit_one_point(
            point_dir,
            out_dir=out_dir,
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
        )
        all_sens_rows.extend(sens_rows)
        all_fd_rows.extend(fd_rows)
        _write_csv(out_dir / "sensitivity_modes.csv", all_sens_rows)
        _write_csv(out_dir / "finite_difference_audit.csv", all_fd_rows)

    summary = {
        "started": started,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "point_count": len(point_dirs),
        "sensitivity_rows": len(all_sens_rows),
        "fd_rows": len(all_fd_rows),
        "settings": vars(args),
        "runtime_environment": runtime_environment,
        "paths": [str(p) for p in point_dirs],
    }
    _write_json(out_dir / "summary.json", summary)
    _write_csv(out_dir / "sensitivity_modes.csv", all_sens_rows)
    _write_csv(out_dir / "finite_difference_audit.csv", all_fd_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
