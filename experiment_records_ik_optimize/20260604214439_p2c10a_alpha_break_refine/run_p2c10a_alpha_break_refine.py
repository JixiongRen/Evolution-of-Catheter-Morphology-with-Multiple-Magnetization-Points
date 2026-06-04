"""Refine the linearization break threshold along a current continuation path."""

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
P2C2_DIR = ROOT / "experiment_records_ik_optimize" / "20260602103750_p2c2_terminal_pose_constraint"
P2C4_DIR = ROOT / "experiment_records_ik_optimize" / "20260603095445_p2c4_clean_terminal_multi_case"
P2C7_DIR = ROOT / "experiment_records_ik_optimize" / "20260604110544_p2c7_shooting_sensitivity_audit"
P2C8_DIR = ROOT / "experiment_records_ik_optimize" / "20260604114419_p2c8_continuity_branch_sensitivity"
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"
for path in (ROOT, P2C2_DIR, P2C4_DIR, P2C7_DIR, P2C8_DIR, FK_OPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_clean_multi_case as p2c4
import run_p2c2_terminal_pose as p2c2
import run_p2c7_sensitivity_audit as p2c7
import run_p2c8_continuity_branch_sensitivity as p2c8

jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).resolve().parent


GATE = {
    "residual_norm_max": 1.0e-2,
    "boundary_residual_norm_max": 2.0e-1,
    "stationarity_norm_max": 1.0e-3,
    "branch_task_spread_max": 1.0e-3,
    "branch_state_spread_max": 5.0,
    "note": "A point is publishable only when residual, boundary, stationarity, and branch gates all pass.",
}


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


def _case_by_name(name: str) -> Any:
    for case in p2c4._cases():
        if case.name == name:
            return case
    raise ValueError(f"Unknown case: {name}")


def _interpolated_case(base: Any, target: Any, alpha: float) -> Any:
    currents = (1.0 - alpha) * np.asarray(base.currents, dtype=float) + alpha * np.asarray(target.currents, dtype=float)
    return p2c4.CaseSpec(
        name=f"alpha_{alpha:.3f}",
        m_body_list=base.m_body_list,
        currents=tuple(float(x) for x in currents),
    )


def _build_functions(case: Any, *, continuity_mu: float, reg_mu: float) -> dict[str, Any]:
    engine = p2c4._build_engine(case, max_iter=120, tol=1.0e-5)
    I = np.asarray(case.currents, dtype=float)
    oracle = p2c2.p2c1.p2c._solve_oracle(engine, I, z0=None)
    target_p, target_q = p2c2._oracle_terminal_pose(oracle, engine)
    params_base = oracle["params"]
    y_oracle = p2c2.p2c1._y_guess_from_oracle(oracle["z"], params_base, "segment_wrench")
    residual_jit, task_jit, output_jit, _continuity_jit, Jy_jit, _JI_jit, _Ty_jit, _TI_jit = p2c8._make_functions(
        params_base=params_base,
        target_p=target_p,
        target_q=target_q,
        y_ref=y_oracle,
        reg_mu=reg_mu,
        continuity_mu=continuity_mu,
        steps_per_flex=4,
    )
    _ = residual_jit(jnp.asarray(y_oracle), jnp.asarray(I)).block_until_ready()
    return {
        "I": I,
        "y_oracle": y_oracle,
        "residual_jit": residual_jit,
        "task_jit": task_jit,
        "output_jit": output_jit,
        "Jy_jit": Jy_jit,
    }


def _solve(bundle: dict[str, Any], y0: np.ndarray, max_iter: int) -> Any:
    return p2c7._solve_branch(
        y0=y0,
        I=bundle["I"],
        residual_jit=bundle["residual_jit"],
        output_jit=bundle["output_jit"],
        Jy_jit=bundle["Jy_jit"],
        max_iter=max_iter,
    )


def _gate_row(row: dict[str, Any]) -> dict[str, Any]:
    residual_ok = row["residual_norm"] <= GATE["residual_norm_max"]
    boundary_ok = row["boundary_residual_norm"] <= GATE["boundary_residual_norm_max"]
    stationarity_ok = row["stationarity_norm"] <= GATE["stationarity_norm_max"]
    return {
        "residual_ok": residual_ok,
        "boundary_ok": boundary_ok,
        "stationarity_ok": stationarity_ok,
        "linearizable": bool(residual_ok and boundary_ok and stationarity_ok),
    }


def _plot_alpha(per_alpha: list[dict[str, Any]]) -> None:
    alpha = np.asarray([row["alpha"] for row in per_alpha], dtype=float)
    residual = np.asarray([row["best_residual_norm"] for row in per_alpha], dtype=float)
    boundary = np.asarray([row["best_boundary_residual_norm"] for row in per_alpha], dtype=float)
    stationarity = np.asarray([row["best_stationarity_norm"] for row in per_alpha], dtype=float)
    linearizable = [row["any_linearizable"] for row in per_alpha]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(alpha, residual, marker="o", label="residual norm")
    ax.plot(alpha, boundary, marker="o", label="boundary residual")
    ax.plot(alpha, stationarity, marker="o", label="stationarity norm")
    for a, ok in zip(alpha, linearizable):
        ax.scatter([a], [1.0e-3], marker="s", s=70, color="#188038" if ok else "#c5221f", zorder=5)
    ax.axhline(GATE["stationarity_norm_max"], color="#c5221f", linestyle="--", linewidth=1.2, label="stationarity gate")
    ax.set_yscale("log")
    ax.set_xlabel("continuation alpha")
    ax.set_ylabel("metric value, log scale")
    ax.set_title("P2c10a alpha break refinement")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alpha_break_metrics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_case", default="n3_mixed_base_i4")
    parser.add_argument("--target_case", default="n3_mixed_rand20")
    parser.add_argument("--alphas", default="0.20,0.25,0.30,0.35,0.40")
    parser.add_argument("--continuity_mu", type=float, default=1.0)
    parser.add_argument("--reg_mu", type=float, default=1.0e-4)
    parser.add_argument("--max_iter", type=int, default=24)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    (OUT_DIR / "gate_definition.json").write_text(json.dumps(GATE, ensure_ascii=False, indent=2), encoding="utf-8")

    base = _case_by_name(args.base_case)
    target = _case_by_name(args.target_case)
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    branch_tracker: dict[str, Any] = {}
    incumbent_y: np.ndarray | None = None
    t_all = time.perf_counter()

    for alpha in alphas:
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "build", "alpha": alpha})
        case = _interpolated_case(base, target, alpha)
        bundle = _build_functions(case, continuity_mu=args.continuity_mu, reg_mu=args.reg_mu)
        branch_inits: dict[str, np.ndarray] = {
            "cold_reference": bundle["y_oracle"],
            "challenger": 0.5 * bundle["y_oracle"],
        }
        if incumbent_y is not None and incumbent_y.shape == bundle["y_oracle"].shape:
            branch_inits["incumbent"] = incumbent_y

        solves: dict[str, Any] = {}
        tasks: dict[str, np.ndarray] = {}
        alpha_rows = []
        for branch_name, y0 in branch_inits.items():
            _append_jsonl(heartbeat, {"ts": time.time(), "phase": "solve", "alpha": alpha, "branch": branch_name})
            solve = _solve(bundle, y0, args.max_iter)
            task = np.asarray(jax.device_get(bundle["task_jit"](jnp.asarray(solve.y), jnp.asarray(bundle["I"]))), dtype=float)
            row = {
                "alpha": alpha,
                "branch": branch_name,
                "success_flag": bool(solve.success),
                "residual_norm": solve.residual_norm,
                "stationarity_norm": solve.stationarity_norm,
                "boundary_residual_norm": solve.boundary_norm,
                "steady_s": solve.steady_s,
                "state_norm": float(np.linalg.norm(solve.y)),
            }
            row.update(_gate_row(row))
            solves[branch_name] = solve
            tasks[branch_name] = task
            rows.append(row)
            alpha_rows.append(row)

        best = min(alpha_rows, key=lambda row: (not row["linearizable"], row["residual_norm"], row["stationarity_norm"]))
        best_solve = solves[best["branch"]]
        if best["linearizable"]:
            incumbent_y = best_solve.y

        linearizable_branches = [row["branch"] for row in alpha_rows if row["linearizable"]]
        branch_task_spread = 0.0
        branch_state_spread = 0.0
        if len(linearizable_branches) >= 2:
            branch_task_spread = max(float(np.linalg.norm(tasks[a] - tasks[b])) for a in linearizable_branches for b in linearizable_branches)
            branch_state_spread = max(float(np.linalg.norm(solves[a].y - solves[b].y)) for a in linearizable_branches for b in linearizable_branches)
        branch_ok = bool(
            len(linearizable_branches) > 0
            and branch_task_spread <= GATE["branch_task_spread_max"]
            and branch_state_spread <= GATE["branch_state_spread_max"]
        )
        branch_tracker[f"{alpha:.3f}"] = {
            "best_branch": best["branch"],
            "linearizable_branches": linearizable_branches,
            "branch_task_spread": branch_task_spread,
            "branch_state_spread": branch_state_spread,
            "branch_ok": branch_ok,
        }
        if not branch_ok:
            _append_jsonl(
                failures,
                {
                    "ts": time.time(),
                    "alpha": alpha,
                    "failure_class": "可线性化断裂" if not linearizable_branches else "分支一致性不足",
                    "evidence": branch_tracker[f"{alpha:.3f}"],
                },
            )

    per_alpha = []
    for alpha in alphas:
        items = [row for row in rows if row["alpha"] == alpha]
        best = min(items, key=lambda row: (not row["linearizable"], row["residual_norm"], row["stationarity_norm"]))
        tracker = branch_tracker[f"{alpha:.3f}"]
        per_alpha.append(
            {
                "alpha": alpha,
                "any_linearizable": bool(any(row["linearizable"] for row in items)),
                "branch_ok": tracker["branch_ok"],
                "best_branch": best["branch"],
                "best_residual_norm": best["residual_norm"],
                "best_stationarity_norm": best["stationarity_norm"],
                "best_boundary_residual_norm": best["boundary_residual_norm"],
                "linearizable_branch_count": len(tracker["linearizable_branches"]),
                "branch_task_spread": tracker["branch_task_spread"],
                "branch_state_spread": tracker["branch_state_spread"],
            }
        )

    first_break_alpha = next((row["alpha"] for row in per_alpha if not row["any_linearizable"]), None)
    last_safe_alpha = next((row["alpha"] for row in reversed(per_alpha) if row["any_linearizable"] and row["branch_ok"]), None)
    classification = "threshold-refined" if first_break_alpha is not None else "no-break-in-window"
    _plot_alpha(per_alpha)

    summary = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "total_wall_s": float(time.perf_counter() - t_all),
            "base_case": args.base_case,
            "target_case": args.target_case,
        },
        "classification": classification,
        "last_safe_alpha": last_safe_alpha,
        "first_break_alpha": first_break_alpha,
        "per_alpha": per_alpha,
        "rows": rows,
        "branch_tracker": branch_tracker,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "sensitivity_audit.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "branch_tracker.json").write_text(json.dumps(_jsonable(branch_tracker), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "summary.csv", rows)
    _write_csv(OUT_DIR / "alpha_scan_summary.csv", per_alpha)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
