"""Run P2c5 JAX clean terminal-pose shooting speed benchmark."""

from __future__ import annotations

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
P2C2_DIR = ROOT / "experiment_records_ik_optimize" / "20260602103750_p2c2_terminal_pose_constraint"
P2C4_DIR = ROOT / "experiment_records_ik_optimize" / "20260603095445_p2c4_clean_terminal_multi_case"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(P2C2_DIR) not in sys.path:
    sys.path.insert(0, str(P2C2_DIR))
if str(P2C4_DIR) not in sys.path:
    sys.path.insert(0, str(P2C4_DIR))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inverse_kinematics.ik_observations import quaternion_so3_residual
from inverse_kinematics.ik_terminal_shooting_jax import (
    CleanShootingWeights,
    clean_terminal_residual,
    forward_segment_wrench,
    output_to_numpy,
)

import run_clean_multi_case as p2c4
import run_p2c2_terminal_pose as p2c2

jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).resolve().parent
WEIGHTS = CleanShootingWeights(tip_weight=120.0, so3_weight=2.0, boundary_weight=0.5)
LAMBDA_MAX = 1e10


@dataclass
class LmResult:
    """Result from the experiment-local dense LM solver."""

    y: np.ndarray
    out: dict[str, np.ndarray]
    history: list[dict[str, Any]]
    success: bool
    cost: float
    n_iter: int
    reject_count: int
    lambda_final: float
    steady_solve_s: float
    residual_eval_s: float
    jacobian_eval_s: float
    linear_solve_s: float


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


def _block(x: Any) -> Any:
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
    elif isinstance(x, (tuple, list)):
        for item in x:
            _block(item)
    return x


def _so3_error_deg(current_q: np.ndarray, target_q: np.ndarray) -> float:
    r = quaternion_so3_residual(jnp.asarray(current_q), jnp.asarray(target_q), sigma_angle=1.0)
    return float(np.degrees(np.linalg.norm(np.asarray(jax.device_get(r), dtype=float))))


def _make_jax_functions(params: Any, target_p: np.ndarray, target_q: np.ndarray, *, steps_per_flex: int):
    target_p_j = jnp.asarray(target_p, dtype=jnp.float64)
    target_q_j = jnp.asarray(target_q, dtype=jnp.float64)

    def residual(y: jnp.ndarray, so3_scale: float) -> jnp.ndarray:
        return clean_terminal_residual(
            params,
            y,
            target_p=target_p_j,
            target_q=target_q_j,
            weights=WEIGHTS,
            steps_per_flex=steps_per_flex,
            so3_scale=so3_scale,
        )

    def output(y: jnp.ndarray):
        return forward_segment_wrench(params, y, steps_per_flex=steps_per_flex)

    residual_jit = jax.jit(residual)
    jacobian_jit = jax.jit(jax.jacfwd(residual, argnums=0))
    output_jit = jax.jit(output)
    return residual_jit, jacobian_jit, output_jit


def _time_compile(residual_jit: Any, jacobian_jit: Any, output_jit: Any, y0: np.ndarray) -> dict[str, float]:
    y = jnp.asarray(y0, dtype=jnp.float64)
    t0 = time.perf_counter()
    _block(residual_jit(y, 1.0))
    residual_compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    _block(jacobian_jit(y, 1.0))
    jacobian_compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    _block(output_jit(y))
    output_compile_s = time.perf_counter() - t0
    return {
        "compile_s": float(residual_compile_s + jacobian_compile_s + output_compile_s),
        "residual_compile_s": float(residual_compile_s),
        "jacobian_compile_s": float(jacobian_compile_s),
        "output_compile_s": float(output_compile_s),
    }


def _jacobian_fd_check(residual_jit: Any, jacobian_jit: Any, y0: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(20260603)
    d = rng.normal(size=y0.shape)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    eps = 1e-5
    y_plus = jnp.asarray(y0 + eps * d, dtype=jnp.float64)
    y_minus = jnp.asarray(y0 - eps * d, dtype=jnp.float64)
    y = jnp.asarray(y0, dtype=jnp.float64)
    fd = np.asarray(jax.device_get((residual_jit(y_plus, 1.0) - residual_jit(y_minus, 1.0)) / (2.0 * eps)), dtype=float)
    jd = np.asarray(jax.device_get(jacobian_jit(y, 1.0) @ jnp.asarray(d, dtype=jnp.float64)), dtype=float)
    abs_err = float(np.linalg.norm(fd - jd))
    rel_err = float(abs_err / max(float(np.linalg.norm(fd)), 1e-12))
    return {"fd_abs_error": abs_err, "fd_rel_error": rel_err}


def _python_jax_parity(params: Any, y0: np.ndarray, output_jit: Any) -> dict[str, float]:
    py = p2c2.p2c1._forward_modes(params, y0, "segment_wrench", steps_per_flex=4)
    jx = output_to_numpy(output_jit(jnp.asarray(y0, dtype=jnp.float64)))
    return {
        "tip_position_diff_m": float(np.linalg.norm(np.asarray(py["tip_p"]) - np.asarray(jx["tip_p"]))),
        "tip_so3_diff_deg": _so3_error_deg(np.asarray(jx["tip_q"]), np.asarray(py["tip_q"])),
        "boundary_diff_norm": float(np.linalg.norm(np.asarray(py["boundary"]) - np.asarray(jx["boundary"]))),
        "centerline_diff_max_m": float(np.max(np.linalg.norm(np.asarray(py["centerline"]) - np.asarray(jx["centerline"]), axis=1))),
    }


def _evaluate_metrics(out: dict[str, np.ndarray], target_p: np.ndarray, target_q: np.ndarray) -> dict[str, float]:
    return {
        "terminal_position_error": float(np.linalg.norm(np.asarray(out["tip_p"]) - target_p)),
        "terminal_so3_error_deg": _so3_error_deg(np.asarray(out["tip_q"]), target_q),
        "boundary_residual_norm": float(np.linalg.norm(np.asarray(out["boundary"]))),
    }


def _lm_solve(
    *,
    y0: np.ndarray,
    residual_jit: Any,
    jacobian_jit: Any,
    output_jit: Any,
    target_p: np.ndarray,
    target_q: np.ndarray,
    max_iter_per_stage: int = 16,
    step_cap: float = 5.0,
) -> LmResult:
    y = np.asarray(y0, dtype=float).reshape(-1)
    lam = 1e-3
    reject_count = 0
    history: list[dict[str, Any]] = []
    residual_eval_s = 0.0
    jacobian_eval_s = 0.0
    linear_solve_s = 0.0
    stages = [0.0, 0.25, 0.6, 1.0]
    t_solve = time.perf_counter()
    done = False

    for stage_index, so3_scale in enumerate(stages):
        for iter_index in range(max_iter_per_stage):
            jy = jnp.asarray(y, dtype=jnp.float64)
            current_out = output_to_numpy(output_jit(jy))
            current_metrics = _evaluate_metrics(current_out, target_p, target_q)
            if so3_scale > 0.0 and current_metrics["terminal_position_error"] <= 1e-3 and current_metrics["terminal_so3_error_deg"] <= 2.0:
                done = True
                break
            t0 = time.perf_counter()
            r = np.asarray(jax.device_get(residual_jit(jy, float(so3_scale))), dtype=float)
            residual_eval_s += time.perf_counter() - t0
            t0 = time.perf_counter()
            J = np.asarray(jax.device_get(jacobian_jit(jy, float(so3_scale))), dtype=float)
            jacobian_eval_s += time.perf_counter() - t0
            cost = 0.5 * float(r @ r)
            g = J.T @ r
            H0 = J.T @ J
            D = np.clip(np.diag(H0), 1e-12, None)
            accepted = False
            rho = float("nan")
            step_norm = float("nan")
            trial_metrics: dict[str, float] = {}
            for _attempt in range(8):
                t0 = time.perf_counter()
                H = H0 + lam * np.diag(D)
                try:
                    dy = np.linalg.solve(H, -g)
                except np.linalg.LinAlgError:
                    dy = np.linalg.lstsq(H, -g, rcond=None)[0]
                linear_solve_s += time.perf_counter() - t0
                step_norm = float(np.linalg.norm(dy))
                if step_norm > step_cap:
                    dy = dy * (step_cap / step_norm)
                    step_norm = float(np.linalg.norm(dy))
                pred = -float(g @ dy + 0.5 * dy @ (H0 @ dy))
                if not np.isfinite(pred) or pred <= 1e-14:
                    lam = min(lam * 10.0, LAMBDA_MAX)
                    continue
                y_trial = y + dy
                t0 = time.perf_counter()
                r_trial = np.asarray(jax.device_get(residual_jit(jnp.asarray(y_trial, dtype=jnp.float64), float(so3_scale))), dtype=float)
                residual_eval_s += time.perf_counter() - t0
                cost_trial = 0.5 * float(r_trial @ r_trial)
                actual = cost - cost_trial
                rho = actual / pred
                out_trial = output_to_numpy(output_jit(jnp.asarray(y_trial, dtype=jnp.float64)))
                trial_metrics = _evaluate_metrics(out_trial, target_p, target_q)
                finite = np.all(np.isfinite(y_trial)) and np.all(np.isfinite(r_trial))
                boundary_ok = trial_metrics["boundary_residual_norm"] <= 5.0 * max(
                    1.0,
                    float(np.linalg.norm(output_to_numpy(output_jit(jy))["boundary"])),
                )
                if finite and actual > 0.0 and boundary_ok and trial_metrics["terminal_so3_error_deg"] < 175.0:
                    y = y_trial
                    accepted = True
                    if rho > 0.75:
                        lam = max(lam * 0.3, 1e-12)
                    elif rho > 0.25:
                        lam = max(lam * 0.5, 1e-12)
                    break
                lam = min(lam * 10.0, LAMBDA_MAX)
            if not accepted:
                reject_count += 1
            row = {
                "stage_index": int(stage_index),
                "so3_scale": float(so3_scale),
                "iter": int(iter_index),
                "cost": float(cost),
                "lambda": float(lam),
                "rho": float(rho),
                "accepted": bool(accepted),
                "step_norm": float(step_norm),
                **trial_metrics,
            }
            history.append(row)
            if lam >= LAMBDA_MAX:
                break
            if so3_scale >= 1.0 and trial_metrics:
                if trial_metrics["terminal_position_error"] <= 1e-3 and trial_metrics["terminal_so3_error_deg"] <= 2.0:
                    done = True
                    break
            if so3_scale > 0.0 and trial_metrics:
                if trial_metrics["terminal_position_error"] <= 1e-3 and trial_metrics["terminal_so3_error_deg"] <= 2.0:
                    done = True
                    break
        if lam >= LAMBDA_MAX:
            break
        if done:
            break

    steady_solve_s = time.perf_counter() - t_solve
    out = output_to_numpy(output_jit(jnp.asarray(y, dtype=jnp.float64)))
    metrics = _evaluate_metrics(out, target_p, target_q)
    success = (
        metrics["terminal_position_error"] <= 1e-3
        and metrics["terminal_so3_error_deg"] <= 2.0
        and reject_count < 20
        and lam < LAMBDA_MAX
    )
    r_final = np.asarray(jax.device_get(residual_jit(jnp.asarray(y, dtype=jnp.float64), 1.0)), dtype=float)
    return LmResult(
        y=y,
        out=out,
        history=history,
        success=bool(success),
        cost=0.5 * float(r_final @ r_final),
        n_iter=len(history),
        reject_count=int(reject_count),
        lambda_final=float(lam),
        steady_solve_s=float(steady_solve_s),
        residual_eval_s=float(residual_eval_s),
        jacobian_eval_s=float(jacobian_eval_s),
        linear_solve_s=float(linear_solve_s),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _plot_case(case_dir: Path, oracle: dict[str, Any], outputs: dict[str, dict[str, np.ndarray]], target_p: np.ndarray) -> None:
    oracle_cl = np.asarray(oracle["centerline"], dtype=float)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(oracle_cl[:, 0], oracle_cl[:, 1], oracle_cl[:, 2], "k-", lw=2.0, label="FK oracle")
    for name, out in outputs.items():
        cl = np.asarray(out["centerline"], dtype=float)
        ax.plot(cl[:, 0], cl[:, 1], cl[:, 2], "--", lw=2.0, label=name)
        ax.scatter([out["tip_p"][0]], [out["tip_p"][1]], [out["tip_p"][2]], s=35)
    ax.scatter([target_p[0]], [target_p[1]], [target_p[2]], c="green", s=60, label="target terminal")
    ax.scatter([oracle_cl[0, 0]], [oracle_cl[0, 1]], [oracle_cl[0, 2]], c="blue", s=35, label="base")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(case_dir.name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(case_dir / "oracle_vs_jax_clean_shooting.png", dpi=180)
    plt.close(fig)


def _plot_summary(rows: list[dict[str, Any]]) -> None:
    labels = [f"{r['case']}\\n{r['init_mode']}" for r in rows]
    x = np.arange(len(rows))
    fig, ax1 = plt.subplots(figsize=(max(10, len(rows) * 1.25), 5))
    ax1.bar(x - 0.25, [r["steady_solve_s"] for r in rows], width=0.25, label="steady solve (s)")
    ax1.axhline(30.0, color="red", ls="--", lw=1.5, label="30s gate")
    ax1.set_ylabel("time (s)")
    ax2 = ax1.twinx()
    ax2.bar(x, [r["terminal_position_error"] * 1000.0 for r in rows], width=0.25, color="tab:green", label="pos (mm)")
    ax2.bar(x + 0.25, [r["terminal_so3_error_deg"] for r in rows], width=0.25, color="tab:orange", label="SO3 (deg)")
    ax2.set_ylabel("error")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.grid(axis="y", alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "p2c5_summary.png", dpi=180)
    plt.close(fig)


def _case_rows() -> list[Any]:
    return p2c4._cases()


def main() -> None:
    t_all = time.perf_counter()
    rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    fd_rows: list[dict[str, Any]] = []
    old_baseline: dict[str, Any] | None = None

    for case_idx, case in enumerate(_case_rows()):
        case_dir = OUT_DIR / case.name
        case_dir.mkdir(parents=True, exist_ok=True)
        engine = p2c4._build_engine(case, max_iter=120, tol=1e-5)
        oracle = p2c2.p2c1.p2c._solve_oracle(engine, np.asarray(case.currents, dtype=float), z0=None)
        target_p, target_q = p2c2._oracle_terminal_pose(oracle, engine)
        params = oracle["params"]
        y_oracle = p2c2.p2c1._y_guess_from_oracle(oracle["z"], params, "segment_wrench")
        init_modes = {"oracle_y0": y_oracle, "zero_y0": np.zeros_like(y_oracle)}

        residual_jit, jacobian_jit, output_jit = _make_jax_functions(params, target_p, target_q, steps_per_flex=4)
        outputs: dict[str, dict[str, np.ndarray]] = {}

        for init_mode, y0 in init_modes.items():
            sub_dir = case_dir / init_mode
            sub_dir.mkdir(parents=True, exist_ok=True)
            compile_profile = _time_compile(residual_jit, jacobian_jit, output_jit, y0)
            if case_idx == 0 and init_mode == "oracle_y0":
                parity = _python_jax_parity(params, y0, output_jit)
                parity_rows.append({"case": case.name, "init_mode": init_mode, **parity})
                fd_rows.append({"case": case.name, "init_mode": init_mode, **_jacobian_fd_check(residual_jit, jacobian_jit, y0)})
            result = _lm_solve(
                y0=y0,
                residual_jit=residual_jit,
                jacobian_jit=jacobian_jit,
                output_jit=output_jit,
                target_p=target_p,
                target_q=target_q,
            )
            metrics = _evaluate_metrics(result.out, target_p, target_q)
            row = {
                "case": case.name,
                "init_mode": init_mode,
                "success": bool(result.success),
                "compile_s": compile_profile["compile_s"],
                "steady_solve_s": result.steady_solve_s,
                "residual_eval_s": result.residual_eval_s,
                "jacobian_eval_s": result.jacobian_eval_s,
                "linear_solve_s": result.linear_solve_s,
                "n_iter": result.n_iter,
                "reject_count": result.reject_count,
                "lambda_final": result.lambda_final,
                "cost": result.cost,
                "oracle_wall_s": float(oracle["wall_s"]),
                "uses_oracle_y0": bool(init_mode == "oracle_y0"),
                "production_candidate": bool(result.success and result.steady_solve_s <= 30.0 and init_mode == "zero_y0"),
                **metrics,
                **compile_profile,
            }
            rows.append(row)
            outputs[init_mode] = result.out
            np.save(sub_dir / "state_y.npy", result.y)
            np.save(sub_dir / "centerline.npy", result.out["centerline"])
            (sub_dir / "history.json").write_text(json.dumps(_jsonable(result.history), ensure_ascii=False, indent=2), encoding="utf-8")
            (sub_dir / "summary.json").write_text(json.dumps(_jsonable(row), ensure_ascii=False, indent=2), encoding="utf-8")

        np.save(case_dir / "oracle_centerline.npy", np.asarray(oracle["centerline"], dtype=float))
        _plot_case(case_dir, oracle, outputs, target_p)

        if case_idx == 0:
            old_path = OUT_DIR / "old_scipy_clean_baseline.json"
            if old_path.exists():
                old_baseline = json.loads(old_path.read_text(encoding="utf-8"))
                continue
            spec = p2c2.PoseCandidateSpec(
                "old_scipy_clean_baseline",
                max_nfev=16,
                centerline_weight=0.0,
                tip_weight=120.0,
                so3_weight=2.0,
                rigid_weight=0.0,
                boundary_weight=0.02,
            )
            old = p2c2._solve_pose_candidate(spec, oracle, engine, steps_per_flex=4, y0=y_oracle)
            old_metrics = p2c2._terminal_metrics(oracle, engine, old.out)
            old_baseline = {
                "case": case.name,
                "init_mode": "oracle_y0",
                "wall_s": float(old.wall_s),
                "nfev": int(old.nfev),
                "cost": float(old.cost),
                **old_metrics,
            }
            (OUT_DIR / "old_scipy_clean_baseline.json").write_text(
                json.dumps(_jsonable(old_baseline), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    _plot_summary(rows)
    _write_csv(OUT_DIR / "summary.csv", rows)
    _write_csv(OUT_DIR / "parity.csv", parity_rows)
    _write_csv(OUT_DIR / "jacobian_fd.csv", fd_rows)
    summary = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "clean_residuals": ["terminal_position", "terminal_so3", "physical_boundary"],
            "excluded_residuals": ["centerline", "rigid_midpoint", "regularization", "history_smoothness"],
            "total_wall_s": float(time.perf_counter() - t_all),
        },
        "parity": parity_rows,
        "jacobian_fd": fd_rows,
        "old_scipy_clean_baseline": old_baseline,
        "rows": rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
