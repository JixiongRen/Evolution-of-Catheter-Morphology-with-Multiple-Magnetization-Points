"""Continuity homotopy audit near the P2c sensitivity break."""

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
    "boundary_norm_max": 2.0e-1,
    "stationarity_norm_max": 1.0e-3,
    "raw_continuity_norm_max": 1.0e-3,
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


def _build_bundle(case: Any, *, reg_mu: float, continuity_mu: float, steps_per_flex: int) -> dict[str, Any]:
    engine = p2c4._build_engine(case, max_iter=120, tol=1.0e-5)
    I = np.asarray(case.currents, dtype=float)
    oracle = p2c2.p2c1.p2c._solve_oracle(engine, I, z0=None)
    target_p, target_q = p2c2._oracle_terminal_pose(oracle, engine)
    params_base = oracle["params"]
    y_ref = p2c2.p2c1._y_guess_from_oracle(oracle["z"], params_base, "segment_wrench")
    residual_jit, task_jit, output_jit, continuity_jit, Jy_jit, JI_jit, Ty_jit, TI_jit = p2c8._make_functions(
        params_base=params_base,
        target_p=target_p,
        target_q=target_q,
        y_ref=y_ref,
        reg_mu=reg_mu,
        continuity_mu=continuity_mu,
        steps_per_flex=steps_per_flex,
    )
    _ = residual_jit(jnp.asarray(y_ref), jnp.asarray(I)).block_until_ready()
    _ = Jy_jit(jnp.asarray(y_ref), jnp.asarray(I)).block_until_ready()
    return {
        "I": I,
        "y_ref": y_ref,
        "residual_jit": residual_jit,
        "output_jit": output_jit,
        "continuity_jit": continuity_jit,
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


def _row_gate(row: dict[str, Any]) -> bool:
    return bool(
        row["residual_norm"] <= GATE["residual_norm_max"]
        and row["boundary_residual_norm"] <= GATE["boundary_norm_max"]
        and row["stationarity_norm"] <= GATE["stationarity_norm_max"]
        and row["raw_continuity_norm"] <= GATE["raw_continuity_norm_max"]
    )


def _plot(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    metrics = [
        ("residual_norm", "total residual"),
        ("boundary_residual_norm", "terminal boundary residual"),
        ("stationarity_norm", "first-order optimality residual"),
        ("raw_continuity_norm", "raw interface continuity residual"),
    ]
    for ax, alpha in zip(axes, sorted({row["alpha"] for row in rows})):
        sub = sorted([row for row in rows if row["alpha"] == alpha], key=lambda item: item["continuity_mu"])
        mus = [row["continuity_mu"] for row in sub]
        for key, label in metrics:
            ax.plot(mus, [row[key] for row in sub], marker="o", label=label)
        ax.axhline(GATE["stationarity_norm_max"], color="#c5221f", linestyle="--", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("continuity weight")
        ax.set_title(f"alpha={alpha:.2f}")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("metric value")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("P2c10e continuity homotopy near sensitivity break")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "continuity_homotopy_metrics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_case", default="n3_mixed_base_i4")
    parser.add_argument("--target_case", default="n3_mixed_rand20")
    parser.add_argument("--alphas", default="0.20,0.25")
    parser.add_argument("--continuity_mus", default="1,10,100,1000")
    parser.add_argument("--reg_mu", type=float, default=1.0e-4)
    parser.add_argument("--max_iter", type=int, default=48)
    parser.add_argument("--steps_per_flex", type=int, default=4)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")

    base = _case_by_name(args.base_case)
    target = _case_by_name(args.target_case)
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    continuity_mus = [float(x) for x in args.continuity_mus.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    t_all = time.perf_counter()

    for alpha in alphas:
        case = _interpolated_case(base, target, alpha)
        y0 = None
        for continuity_mu in continuity_mus:
            _append_jsonl(heartbeat, {"ts": time.time(), "phase": "solve", "alpha": alpha, "continuity_mu": continuity_mu})
            try:
                bundle = _build_bundle(
                    case,
                    reg_mu=args.reg_mu,
                    continuity_mu=continuity_mu,
                    steps_per_flex=args.steps_per_flex,
                )
                if y0 is None:
                    y0 = bundle["y_ref"]
                solve = _solve(bundle, y0, args.max_iter)
                y0 = solve.y
                raw_cont = float(
                    np.linalg.norm(
                        np.asarray(
                            jax.device_get(bundle["continuity_jit"](jnp.asarray(solve.y), jnp.asarray(bundle["I"]))),
                            dtype=float,
                        )
                    )
                )
                row = {
                    "alpha": alpha,
                    "continuity_mu": continuity_mu,
                    "success_flag": bool(solve.success),
                    "residual_norm": solve.residual_norm,
                    "stationarity_norm": solve.stationarity_norm,
                    "boundary_residual_norm": solve.boundary_norm,
                    "raw_continuity_norm": raw_cont,
                    "steady_s": solve.steady_s,
                }
                row["linearizable_with_continuity_gate"] = _row_gate(row)
            except Exception as exc:
                row = {
                    "alpha": alpha,
                    "continuity_mu": continuity_mu,
                    "success_flag": False,
                    "residual_norm": float("inf"),
                    "stationarity_norm": float("inf"),
                    "boundary_residual_norm": float("inf"),
                    "raw_continuity_norm": float("inf"),
                    "steady_s": 0.0,
                    "linearizable_with_continuity_gate": False,
                    "error": repr(exc),
                }
                _append_jsonl(failures, {"ts": time.time(), "failure_class": "连续性同伦求解异常", "evidence": row})
            rows.append(row)
            if not row["linearizable_with_continuity_gate"]:
                _append_jsonl(failures, {"ts": time.time(), "failure_class": "连续性同伦未通过门禁", "evidence": row})

    _plot(rows)
    best_by_alpha = []
    for alpha in alphas:
        sub = [row for row in rows if row["alpha"] == alpha]
        best = min(sub, key=lambda item: item["stationarity_norm"])
        best_by_alpha.append(
            {
                "alpha": alpha,
                "best_continuity_mu": best["continuity_mu"],
                "best_stationarity_norm": best["stationarity_norm"],
                "best_residual_norm": best["residual_norm"],
                "best_boundary_residual_norm": best["boundary_residual_norm"],
                "best_raw_continuity_norm": best["raw_continuity_norm"],
                "any_linearizable_with_continuity_gate": any(row["linearizable_with_continuity_gate"] for row in sub),
            }
        )

    summary = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "total_wall_s": float(time.perf_counter() - t_all),
            "gate": GATE,
        },
        "classification": "continuity-homotopy-audited",
        "best_by_alpha": best_by_alpha,
        "rows": rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "sensitivity_audit.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "summary.csv", rows)
    _write_csv(OUT_DIR / "alpha_best_summary.csv", best_by_alpha)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
