"""Audit strict sensitivity gates under reduced terminal task definitions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
P2C10I_DIR = ROOT / "experiment_records_ik_optimize" / "20260605005323_p2c10i_boundary_candidate_strict_gate"
for path in (ROOT, P2C10I_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inverse_kinematics.ik_terminal_shooting_jax import axis_from_quat_wxyz

import run_p2c10i_boundary_candidate_strict_gate as p2c10i

jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).resolve().parent
AXIS_BODY = jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float64)
RNG = np.random.default_rng(20260605032052)


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


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(ROOT), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc!r}"


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _candidate_specs(text: str) -> list[tuple[str, float]]:
    specs = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        name, alpha_text = raw.split(":", 1)
        specs.append((name.strip(), float(alpha_text)))
    return specs


def _directions(I0: np.ndarray, random_count: int) -> list[tuple[str, np.ndarray]]:
    active = int(np.argmax(np.abs(I0))) if np.max(np.abs(I0)) > 0.0 else 5
    e = np.zeros(8)
    e[active] = 1.0
    directions = [(f"active_{active}", e)]
    for i in range(int(random_count)):
        d = RNG.normal(size=8)
        d = d / max(float(np.linalg.norm(d)), 1.0e-12)
        directions.append((f"random_{i}", d))
    return directions


def _safe_rel(pred: np.ndarray, ref: np.ndarray) -> tuple[float, float, float]:
    pred_norm = float(np.linalg.norm(pred))
    ref_norm = float(np.linalg.norm(ref))
    err = float(np.linalg.norm(pred - ref))
    rel = err / max(ref_norm, 1.0e-12)
    cosine = float(np.dot(pred, ref) / max(pred_norm * ref_norm, 1.0e-30))
    return rel, cosine, ref_norm


def _rank_metrics(matrix: np.ndarray, required_rank: int) -> dict[str, Any]:
    svals = np.linalg.svd(matrix, compute_uv=False)
    if svals.size:
        threshold = max(1.0e-8, 1.0e-3 * float(svals[0]))
        rank = int(np.sum(svals >= threshold))
        cond = float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny))
        sigma_max = float(svals[0])
        sigma_min = float(svals[-1])
    else:
        rank = 0
        cond = float("inf")
        sigma_max = 0.0
        sigma_min = 0.0
    return {
        "rank": rank,
        "required_rank": required_rank,
        "rank_ok": bool(rank >= required_rank),
        "condition": cond,
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "fro_norm": float(np.linalg.norm(matrix)),
        "singular_values": svals.tolist(),
    }


def _make_task_bundle(bundle: dict[str, Any], mode: str) -> dict[str, Any]:
    target_q = jnp.asarray(bundle["target_q"], dtype=jnp.float64).reshape(4)
    target_axis = axis_from_quat_wxyz(target_q, AXIS_BODY)

    def position_task(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        out = bundle["output_jit"](y, I)
        return out.tip_p.reshape(-1)

    def position_axis_task(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        out = bundle["output_jit"](y, I)
        axis = axis_from_quat_wxyz(out.tip_q, AXIS_BODY)
        return jnp.concatenate([out.tip_p.reshape(-1), (axis - target_axis).reshape(-1)], axis=0)

    def full_pose_task(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        return bundle["task_jit"](y, I)

    if mode == "position3":
        task_fun = position_task
        required_rank = 3
    elif mode == "position_axis":
        task_fun = position_axis_task
        required_rank = 5
    elif mode == "full_pose6":
        task_fun = full_pose_task
        required_rank = 6
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    task_jit = jax.jit(task_fun)
    Ty_jit = jax.jit(jax.jacfwd(task_fun, argnums=0))
    TI_jit = jax.jit(jax.jacfwd(task_fun, argnums=1))
    return {
        "mode": mode,
        "task_jit": task_jit,
        "Ty_jit": Ty_jit,
        "TI_jit": TI_jit,
        "required_rank": required_rank,
    }


def _sensitivity_matrix(bundle: dict[str, Any], task_bundle: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    return p2c10i._exact_sensitivity_matrix(
        y=bundle["base"].y,
        I=bundle["I0"],
        grad_jit=bundle["grad_jit"],
        H_jit=bundle["H_jit"],
        GI_jit=bundle["GI_jit"],
        Ty_jit=task_bundle["Ty_jit"],
        TI_jit=task_bundle["TI_jit"],
    )


def _solve_perturbed(bundle: dict[str, Any], delta_I: np.ndarray, max_iter: int) -> tuple[Any, Any]:
    base = bundle["base"]
    I0 = bundle["I0"]
    plus = p2c10i.p2c7._solve_branch(
        y0=base.y,
        I=I0 + delta_I,
        residual_jit=bundle["residual_jit"],
        output_jit=bundle["output_jit"],
        Jy_jit=bundle["Jy_jit"],
        max_iter=max_iter,
    )
    minus = p2c10i.p2c7._solve_branch(
        y0=base.y,
        I=I0 - delta_I,
        residual_jit=bundle["residual_jit"],
        output_jit=bundle["output_jit"],
        Jy_jit=bundle["Jy_jit"],
        max_iter=max_iter,
    )
    return plus, minus


def _classify_case(row: dict[str, Any]) -> tuple[str, str]:
    if not row["base_success"]:
        return "诊断结果", "基线求解未通过"
    if row["stationarity_norm"] > 1.0e-3:
        return "诊断结果", "一阶平稳性不足"
    if row["hessian_condition"] > 1.0e10:
        return "诊断结果", "局部条件数过高"
    if not row["branch_consistent"]:
        return "诊断结果", "多初值分支不一致"
    if not row["rank_ok"]:
        return "诊断结果", "任务有效秩不足"
    return "候选", "基线门禁通过"


def _classify_direction(case_status: str, row: dict[str, Any]) -> tuple[str, str]:
    if case_status != "候选":
        return "诊断结果", row["case_failure_class"]
    if not row["plus_success"] or not row["minus_success"]:
        return "诊断结果", "扰动重求解失败"
    if row["branch_jump"]:
        return "诊断结果", "扰动后分支跳变"
    if row["rel_err_central"] > 0.3 or row["cosine_central"] <= 0.0:
        return "诊断结果", "扰动有限差分不一致"
    if row["linearization_error"] > 0.1:
        return "诊断结果", "局部线性化误差过大"
    return "可发布候选", "通过严格门禁"


def _plot_mode_rank(rows: list[dict[str, Any]]) -> None:
    modes = ["position3", "position_axis", "full_pose6"]
    labels = sorted({f"{row['case']}@{row['alpha']:.2f}" for row in rows})
    matrix = np.full((len(labels), len(modes)), np.nan)
    for i, label in enumerate(labels):
        for j, mode in enumerate(modes):
            item = next(row for row in rows if f"{row['case']}@{row['alpha']:.2f}" == label and row["mode"] == mode)
            matrix[i, j] = item["rank"] / max(item["required_rank"], 1)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(modes)), modes)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("task-rank ratio")
    fig.colorbar(im, ax=ax, label="rank / required rank")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "task_rank_ratio.png", dpi=180)
    plt.close(fig)


def _plot_gate_counts(rows: list[dict[str, Any]]) -> None:
    modes = ["position3", "position_axis", "full_pose6"]
    publishable = [sum(1 for row in rows if row["mode"] == mode and row["status"] == "可发布候选") for mode in modes]
    diagnostic = [sum(1 for row in rows if row["mode"] == mode and row["status"] != "可发布候选") for mode in modes]
    x = np.arange(len(modes))
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.bar(x, publishable, label="publishable", color="#188038")
    ax.bar(x, diagnostic, bottom=publishable, label="diagnostic", color="#c5221f")
    ax.set_xticks(x, modes)
    ax.set_ylabel("direction samples")
    ax.set_title("strict gate outcome by task")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "task_gate_counts.png", dpi=180)
    plt.close(fig)


def _plot_linearization(rows: list[dict[str, Any]]) -> None:
    labels = [f"{row['case']}@{row['alpha']:.2f}\\n{row['mode']}\\n{row['direction']}" for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.6), 5.5))
    ax.bar(x, [row["linearization_error"] for row in rows], color="#1f77b4")
    ax.axhline(0.1, color="r", linestyle="--", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
    ax.set_ylabel("linearization error")
    ax.set_title("local linearization error by reduced task")
    ax.grid(axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "task_linearization_error.png", dpi=180)
    plt.close(fig)


def _plot_time_cost(case_rows: list[dict[str, Any]], total_wall_s: float) -> None:
    labels = [f"{row['case']}@{row['alpha']:.2f}" for row in case_rows] + ["total"]
    values = [row["base_steady_s"] for row in case_rows] + [total_wall_s]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4.8))
    ax.bar(np.arange(len(labels)), values, color="#1f77b4")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    ax.set_ylabel("time (s)")
    ax.set_title("time cost")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "time_cost.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="n1_tip_z_rand15:0.10,n2_z_pair_rand12:0.10,n3_mixed_rand20:0.10,n3_mixed_base_i4:1.00")
    parser.add_argument("--modes", default="position3,position_axis,full_pose6")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--continuity_mu", type=float, default=1.0)
    parser.add_argument("--reg_mu", type=float, default=1.0e-4)
    parser.add_argument("--max_iter", type=int, default=32)
    parser.add_argument("--random_count", type=int, default=1)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    t_all = time.perf_counter()

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    case_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []

    for case_name, alpha in _candidate_specs(args.candidates):
        target_case = p2c10i._case_by_name(case_name)
        case = p2c10i._scaled_case(target_case, alpha)
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "build", "case": case.name})
        bundle = p2c10i._build_case_functions(case, args.continuity_mu, args.reg_mu, args.max_iter)
        base = bundle["base"]
        branch = p2c10i._branch_spread(bundle, args.max_iter)
        directions = _directions(bundle["I0"], args.random_count)
        perturb_solutions: dict[str, tuple[Any, Any, np.ndarray]] = {}
        for direction_name, direction in directions:
            delta_I = float(args.step) * direction
            plus, minus = _solve_perturbed(bundle, delta_I, args.max_iter)
            perturb_solutions[direction_name] = (plus, minus, delta_I)

        case_row = {
            "case": case_name,
            "scaled_case": case.name,
            "alpha": alpha,
            "active_magnet_count": p2c10i._active_magnet_count(target_case),
            "base_success": base.success,
            "base_residual_norm": base.residual_norm,
            "base_stationarity_norm": base.stationarity_norm,
            "base_boundary_norm": base.boundary_norm,
            "base_steady_s": base.steady_s,
            **branch,
        }
        case_rows.append(case_row)

        for mode in modes:
            task_bundle = _make_task_bundle(bundle, mode)
            _ = task_bundle["task_jit"](jnp.asarray(base.y), jnp.asarray(bundle["I0"])).block_until_ready()
            sensitivity, diag = _sensitivity_matrix(bundle, task_bundle)
            rank = _rank_metrics(sensitivity, task_bundle["required_rank"])
            task_row = {
                **case_row,
                "mode": mode,
                "rank": rank["rank"],
                "required_rank": rank["required_rank"],
                "rank_ok": rank["rank_ok"],
                "sensitivity_condition": rank["condition"],
                "sensitivity_sigma_max": rank["sigma_max"],
                "sensitivity_sigma_min": rank["sigma_min"],
                "sensitivity_fro_norm": rank["fro_norm"],
                "singular_values": rank["singular_values"],
                "hessian_condition": diag["hessian_condition"],
                "stationarity_norm": diag["stationarity_norm"],
            }
            case_status, case_failure = _classify_case(task_row)
            task_row["case_status"] = case_status
            task_row["case_failure_class"] = case_failure
            task_rows.append(task_row)

            task0 = np.asarray(jax.device_get(task_bundle["task_jit"](jnp.asarray(base.y), jnp.asarray(bundle["I0"]))), dtype=float)
            for direction_name, _direction in directions:
                plus, minus, delta_I = perturb_solutions[direction_name]
                predicted = sensitivity @ delta_I
                task_plus = np.asarray(
                    jax.device_get(task_bundle["task_jit"](jnp.asarray(plus.y), jnp.asarray(bundle["I0"] + delta_I))),
                    dtype=float,
                )
                task_minus = np.asarray(
                    jax.device_get(task_bundle["task_jit"](jnp.asarray(minus.y), jnp.asarray(bundle["I0"] - delta_I))),
                    dtype=float,
                )
                actual_plus = task_plus - task0
                central_delta = 0.5 * (task_plus - task_minus)
                rel_plus, cosine_plus, actual_norm = _safe_rel(predicted, actual_plus)
                rel_central, cosine_central, central_norm = _safe_rel(predicted, central_delta)
                lin_err = float(np.linalg.norm(task_plus - task0 - predicted) / max(float(np.linalg.norm(task0)), 1.0e-12))
                row = {
                    **task_row,
                    "direction": direction_name,
                    "step_A": args.step,
                    "rel_err_plus": rel_plus,
                    "cosine_plus": cosine_plus,
                    "actual_plus_norm": actual_norm,
                    "rel_err_central": rel_central,
                    "cosine_central": cosine_central,
                    "central_delta_norm": central_norm,
                    "linearization_error": lin_err,
                    "predicted_norm": float(np.linalg.norm(predicted)),
                    "plus_success": plus.success,
                    "minus_success": minus.success,
                    "branch_delta_plus": float(np.linalg.norm(plus.y - base.y)),
                    "branch_delta_minus": float(np.linalg.norm(minus.y - base.y)),
                    "branch_jump": bool(max(np.linalg.norm(plus.y - base.y), np.linalg.norm(minus.y - base.y)) > 5.0),
                }
                status, failure_class = _classify_direction(case_status, row)
                row["status"] = status
                row["failure_class"] = failure_class
                direction_rows.append(row)
                if status != "可发布候选":
                    _append_jsonl(
                        failures,
                        {
                            "ts": time.time(),
                            "case": case.name,
                            "mode": mode,
                            "direction": direction_name,
                            "failure_class": failure_class,
                            "evidence": row,
                        },
                    )

    total_wall_s = float(time.perf_counter() - t_all)
    _plot_mode_rank(task_rows)
    _plot_gate_counts(direction_rows)
    _plot_linearization(direction_rows)
    _plot_time_cost(case_rows, total_wall_s)
    class_counts: dict[str, int] = {}
    for row in direction_rows:
        class_counts[row["failure_class"]] = class_counts.get(row["failure_class"], 0) + 1
    summary = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "total_wall_s": total_wall_s,
            "step_A": args.step,
            "continuity_mu": args.continuity_mu,
            "reg_mu": args.reg_mu,
            "max_iter": args.max_iter,
        },
        "classification": "任务降维严格门禁",
        "class_counts": class_counts,
        "publishable_count": sum(1 for row in direction_rows if row["status"] == "可发布候选"),
        "case_rows": case_rows,
        "task_rows": task_rows,
        "rows": direction_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "case_summary.csv", case_rows)
    _write_csv(OUT_DIR / "task_summary.csv", task_rows)
    _write_csv(OUT_DIR / "summary.csv", direction_rows)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
