"""Run strict sensitivity gates for P2c10h boundary candidates."""

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
P2C9F_DIR = ROOT / "experiment_records_ik_optimize" / "20260604141002_p2c9f_exact_stationarity_ift_probe"
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"
for path in (ROOT, P2C2_DIR, P2C4_DIR, P2C7_DIR, P2C8_DIR, P2C9F_DIR, FK_OPT_DIR):
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
import run_p2c9f_exact_stationarity_ift_probe as p2c9f

jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260605005323)


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


def _scaled_case(case: Any, alpha: float) -> Any:
    currents = alpha * np.asarray(case.currents, dtype=float)
    return p2c4.CaseSpec(
        name=f"{case.name}_alpha_{alpha:.2f}",
        m_body_list=case.m_body_list,
        currents=tuple(float(x) for x in currents),
    )


def _active_magnet_count(case: Any) -> int:
    return int(sum(np.linalg.norm(np.asarray(m, dtype=float)) > 1.0e-12 for m in case.m_body_list))


def _relative_error_and_cosine(pred: np.ndarray, ref: np.ndarray) -> tuple[float, float, float]:
    pred_norm = float(np.linalg.norm(pred))
    ref_norm = float(np.linalg.norm(ref))
    err = float(np.linalg.norm(pred - ref))
    rel = err / max(ref_norm, 1.0e-12)
    cosine = float(np.dot(pred, ref) / max(pred_norm * ref_norm, 1.0e-30))
    return rel, cosine, ref_norm


def _directions(I0: np.ndarray, random_count: int) -> list[tuple[str, np.ndarray]]:
    active = int(np.argmax(np.abs(I0))) if np.max(np.abs(I0)) > 0.0 else 5
    e = np.zeros(8)
    e[active] = 1.0
    directions = [(f"主激活线圈_{active}", e)]
    for i in range(int(random_count)):
        d = RNG.normal(size=8)
        d = d / max(float(np.linalg.norm(d)), 1.0e-12)
        directions.append((f"随机方向_{i}", d))
    return directions


def _exact_sensitivity_matrix(
    *,
    y: np.ndarray,
    I: np.ndarray,
    grad_jit: Any,
    H_jit: Any,
    GI_jit: Any,
    Ty_jit: Any,
    TI_jit: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    y_j = jnp.asarray(y, dtype=jnp.float64)
    I_j = jnp.asarray(I, dtype=jnp.float64)
    g = np.asarray(jax.device_get(grad_jit(y_j, I_j)), dtype=float)
    H = np.asarray(jax.device_get(H_jit(y_j, I_j)), dtype=float)
    GI = np.asarray(jax.device_get(GI_jit(y_j, I_j)), dtype=float)
    Ty = np.asarray(jax.device_get(Ty_jit(y_j, I_j)), dtype=float)
    TI = np.asarray(jax.device_get(TI_jit(y_j, I_j)), dtype=float)
    diag = np.clip(np.abs(np.diag(H)), 1.0e-12, None)
    H_reg = H + 1.0e-8 * np.diag(diag)
    try:
        dydI = -np.linalg.solve(H_reg, GI)
    except np.linalg.LinAlgError:
        dydI = -np.linalg.lstsq(H_reg, GI, rcond=None)[0]
    sensitivity = TI + Ty @ dydI
    svals = np.linalg.svd(H, compute_uv=False)
    return sensitivity, {
        "stationarity_norm": float(np.linalg.norm(g)),
        "hessian_condition": float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny)) if svals.size else float("inf"),
        "hessian_rank": int(np.linalg.matrix_rank(H)),
        "hessian_min_singular": float(svals[-1]) if svals.size else float("nan"),
        "hessian_max_singular": float(svals[0]) if svals.size else float("nan"),
    }


def _build_case_functions(case: Any, continuity_mu: float, reg_mu: float, max_iter: int) -> dict[str, Any]:
    engine = p2c4._build_engine(case, max_iter=120, tol=1.0e-5)
    I0 = np.asarray(case.currents, dtype=float)
    oracle = p2c2.p2c1.p2c._solve_oracle(engine, I0, z0=None)
    target_p, target_q = p2c2._oracle_terminal_pose(oracle, engine)
    params_base = oracle["params"]
    y_oracle = p2c2.p2c1._y_guess_from_oracle(oracle["z"], params_base, "segment_wrench")
    residual_jit, task_jit, output_jit, continuity_jit, Jy_jit, _JI_jit, _Ty_jit, _TI_jit = p2c8._make_functions(
        params_base=params_base,
        target_p=target_p,
        target_q=target_q,
        y_ref=y_oracle,
        reg_mu=float(reg_mu),
        continuity_mu=float(continuity_mu),
        steps_per_flex=4,
    )
    grad_jit, H_jit, GI_jit, Ty_jit, TI_jit = p2c9f._make_exact_functions(residual_fun=residual_jit, task_fun=task_jit)
    _ = residual_jit(jnp.asarray(y_oracle), jnp.asarray(I0)).block_until_ready()
    _ = H_jit(jnp.asarray(y_oracle), jnp.asarray(I0)).block_until_ready()
    base = p2c7._solve_branch(
        y0=y_oracle,
        I=I0,
        residual_jit=residual_jit,
        output_jit=output_jit,
        Jy_jit=Jy_jit,
        max_iter=max_iter,
    )
    return {
        "I0": I0,
        "target_p": np.asarray(target_p, dtype=float),
        "target_q": np.asarray(target_q, dtype=float),
        "y_oracle": y_oracle,
        "base": base,
        "residual_jit": residual_jit,
        "task_jit": task_jit,
        "output_jit": output_jit,
        "continuity_jit": continuity_jit,
        "Jy_jit": Jy_jit,
        "grad_jit": grad_jit,
        "H_jit": H_jit,
        "GI_jit": GI_jit,
        "Ty_jit": Ty_jit,
        "TI_jit": TI_jit,
    }


def _branch_spread(bundle: dict[str, Any], max_iter: int) -> dict[str, Any]:
    init_map = {
        "oracle": bundle["y_oracle"],
        "zero": np.zeros_like(bundle["y_oracle"]),
        "half_oracle": 0.5 * bundle["y_oracle"],
    }
    solves = {}
    tasks = {}
    for name, y0 in init_map.items():
        solve = p2c7._solve_branch(
            y0=y0,
            I=bundle["I0"],
            residual_jit=bundle["residual_jit"],
            output_jit=bundle["output_jit"],
            Jy_jit=bundle["Jy_jit"],
            max_iter=max_iter,
        )
        solves[name] = solve
        tasks[name] = np.asarray(jax.device_get(bundle["task_jit"](jnp.asarray(solve.y), jnp.asarray(bundle["I0"]))), dtype=float)
    names = list(init_map)
    task_spread = max(float(np.linalg.norm(tasks[a] - tasks[b])) for a in names for b in names)
    state_spread = max(float(np.linalg.norm(solves[a].y - solves[b].y)) for a in names for b in names)
    success_count = sum(1 for solve in solves.values() if solve.success)
    return {
        "branch_task_spread": task_spread,
        "branch_state_spread": state_spread,
        "branch_success_count": success_count,
        "branch_init_count": len(init_map),
        "branch_consistent": bool(success_count == len(init_map) and task_spread <= 1.0e-3 and state_spread <= 5.0),
    }


def _audit_direction(
    *,
    bundle: dict[str, Any],
    sensitivity: np.ndarray,
    direction_name: str,
    direction: np.ndarray,
    step: float,
    max_iter: int,
) -> dict[str, Any]:
    base = bundle["base"]
    I0 = bundle["I0"]
    task0 = np.asarray(jax.device_get(bundle["task_jit"](jnp.asarray(base.y), jnp.asarray(I0))), dtype=float)
    delta_I = float(step) * np.asarray(direction, dtype=float)
    predicted_plus = sensitivity @ delta_I
    plus = p2c7._solve_branch(
        y0=base.y,
        I=I0 + delta_I,
        residual_jit=bundle["residual_jit"],
        output_jit=bundle["output_jit"],
        Jy_jit=bundle["Jy_jit"],
        max_iter=max_iter,
    )
    minus = p2c7._solve_branch(
        y0=base.y,
        I=I0 - delta_I,
        residual_jit=bundle["residual_jit"],
        output_jit=bundle["output_jit"],
        Jy_jit=bundle["Jy_jit"],
        max_iter=max_iter,
    )
    task_plus = np.asarray(jax.device_get(bundle["task_jit"](jnp.asarray(plus.y), jnp.asarray(I0 + delta_I))), dtype=float)
    task_minus = np.asarray(jax.device_get(bundle["task_jit"](jnp.asarray(minus.y), jnp.asarray(I0 - delta_I))), dtype=float)
    actual_plus = task_plus - task0
    central_delta = 0.5 * (task_plus - task_minus)
    rel_err_plus, cosine_plus, actual_norm = _relative_error_and_cosine(predicted_plus, actual_plus)
    rel_err_central, cosine_central, central_norm = _relative_error_and_cosine(predicted_plus, central_delta)
    lin_err = float(np.linalg.norm(task_plus - task0 - predicted_plus) / max(float(np.linalg.norm(task0)), 1.0e-12))
    branch_delta_plus = float(np.linalg.norm(plus.y - base.y))
    branch_delta_minus = float(np.linalg.norm(minus.y - base.y))
    branch_jump = bool(max(branch_delta_plus, branch_delta_minus) > 5.0)
    return {
        "direction": direction_name,
        "step_A": float(step),
        "rel_err_plus": rel_err_plus,
        "cosine_plus": cosine_plus,
        "actual_plus_norm": actual_norm,
        "rel_err_central": rel_err_central,
        "cosine_central": cosine_central,
        "central_delta_norm": central_norm,
        "linearization_error": lin_err,
        "predicted_norm": float(np.linalg.norm(predicted_plus)),
        "plus_success": plus.success,
        "minus_success": minus.success,
        "plus_residual_norm": plus.residual_norm,
        "minus_residual_norm": minus.residual_norm,
        "plus_stationarity_norm": plus.stationarity_norm,
        "minus_stationarity_norm": minus.stationarity_norm,
        "plus_boundary_norm": plus.boundary_norm,
        "minus_boundary_norm": minus.boundary_norm,
        "branch_delta_plus": branch_delta_plus,
        "branch_delta_minus": branch_delta_minus,
        "branch_jump": branch_jump,
        "delta_I": delta_I.tolist(),
    }


def _classify(row: dict[str, Any]) -> tuple[str, str]:
    if not row["base_success"]:
        return "诊断结果", "基线求解未通过"
    if row["base_residual_norm"] > 1.0e-2:
        return "诊断结果", "基线残差过大"
    if row["base_stationarity_norm"] > 1.0e-3:
        return "诊断结果", "一阶平稳性不足"
    if row["base_boundary_norm"] > 0.2:
        return "诊断结果", "边界残差过大"
    if row["hessian_condition"] > 1.0e10:
        return "诊断结果", "局部条件数过高"
    if not row["branch_consistent"]:
        return "诊断结果", "多初值分支不一致"
    if not row["plus_success"] or not row["minus_success"]:
        return "诊断结果", "扰动重求解失败"
    if row["branch_jump"]:
        return "诊断结果", "扰动后分支跳变"
    if row["rel_err_central"] > 0.3 or row["cosine_central"] <= 0.0:
        return "诊断结果", "重求解有限差分不一致"
    if row["linearization_error"] > 0.1:
        return "诊断结果", "局部线性近似误差过大"
    return "可发布候选", "通过严格门禁"


def _plot_metrics(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    labels = [f"{row['case']}\\ndir{i}" for i, row in enumerate(rows)]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(max(10, len(rows) * 1.5), 8))
    axes[0, 0].bar(x, [row["base_stationarity_norm"] for row in rows])
    axes[0, 0].axhline(1.0e-3, color="r", linestyle="--", linewidth=1.0)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("stationarity")
    axes[0, 1].bar(x, [row["hessian_condition"] for row in rows])
    axes[0, 1].axhline(1.0e10, color="r", linestyle="--", linewidth=1.0)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("local condition")
    axes[1, 0].bar(x, [row["rel_err_central"] for row in rows])
    axes[1, 0].axhline(0.3, color="r", linestyle="--", linewidth=1.0)
    axes[1, 0].set_title("FD relative error")
    axes[1, 1].bar(x, [row["linearization_error"] for row in rows])
    axes[1, 1].axhline(0.1, color="r", linestyle="--", linewidth=1.0)
    axes[1, 1].set_title("linearization error")
    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.suptitle("P2c10i boundary candidate strict gates")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "strict_gate_metrics.png", dpi=180)
    plt.close(fig)


def _plot_classification(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fail_counts: dict[str, int] = {}
    for row in rows:
        fail_counts[row["failure_class"]] = fail_counts.get(row["failure_class"], 0) + 1
    label_map = {
        "一阶平稳性不足": "stationarity",
        "局部条件数过高": "ill-conditioned",
    }
    labels = [label_map.get(key, f"F{i}") for i, key in enumerate(fail_counts)]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 4.8))
    ax.bar(np.arange(len(labels)), list(fail_counts.values()), color="#c5221f")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    ax.set_ylabel("count")
    ax.set_title("failure class counts")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "failure_class_counts.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="n2_z_pair_rand12:0.25,n3_mixed_rand20:0.25")
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
    rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    t_all = time.perf_counter()

    specs: list[tuple[str, float]] = []
    for raw in args.candidates.split(","):
        if not raw.strip():
            continue
        name, alpha_text = raw.split(":", 1)
        specs.append((name.strip(), float(alpha_text)))

    for case_name, alpha in specs:
        target_case = _case_by_name(case_name)
        case = _scaled_case(target_case, alpha)
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "build", "case": case.name})
        bundle = _build_case_functions(case, args.continuity_mu, args.reg_mu, args.max_iter)
        base = bundle["base"]
        continuity_norm = float(
            np.linalg.norm(
                np.asarray(
                    jax.device_get(bundle["continuity_jit"](jnp.asarray(base.y), jnp.asarray(bundle["I0"]))),
                    dtype=float,
                )
            )
        )
        branch = _branch_spread(bundle, args.max_iter)
        sensitivity, diag = _exact_sensitivity_matrix(
            y=base.y,
            I=bundle["I0"],
            grad_jit=bundle["grad_jit"],
            H_jit=bundle["H_jit"],
            GI_jit=bundle["GI_jit"],
            Ty_jit=bundle["Ty_jit"],
            TI_jit=bundle["TI_jit"],
        )
        baseline_row = {
            "case": case_name,
            "scaled_case": case.name,
            "alpha": alpha,
            "active_magnet_count": _active_magnet_count(target_case),
            "base_success": base.success,
            "base_residual_norm": base.residual_norm,
            "base_stationarity_norm": base.stationarity_norm,
            "base_boundary_norm": base.boundary_norm,
            "base_continuity_norm": continuity_norm,
            "base_steady_s": base.steady_s,
            **branch,
            **diag,
        }
        baseline_rows.append(baseline_row)

        for direction_name, direction in _directions(bundle["I0"], args.random_count):
            _append_jsonl(heartbeat, {"ts": time.time(), "phase": "audit", "case": case.name, "direction": direction_name})
            direction_row = _audit_direction(
                bundle=bundle,
                sensitivity=sensitivity,
                direction_name=direction_name,
                direction=direction,
                step=args.step,
                max_iter=args.max_iter,
            )
            row = {
                **baseline_row,
                **direction_row,
                "currents": bundle["I0"].tolist(),
                "m_body_list": target_case.m_body_list,
            }
            status, failure_class = _classify(row)
            row["status"] = status
            row["failure_class"] = failure_class
            rows.append(row)
            if status != "可发布候选":
                _append_jsonl(
                    failures,
                    {
                        "ts": time.time(),
                        "case": case.name,
                        "direction": direction_name,
                        "failure_class": failure_class,
                        "evidence": row,
                    },
                )

    _plot_metrics(rows)
    _plot_classification(rows)
    publishable = [row for row in rows if row["status"] == "可发布候选"]
    summary = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "total_wall_s": float(time.perf_counter() - t_all),
            "step_A": args.step,
            "continuity_mu": args.continuity_mu,
            "reg_mu": args.reg_mu,
            "max_iter": args.max_iter,
        },
        "classification": "存在可发布候选" if publishable else "仅诊断结果",
        "row_count": len(rows),
        "publishable_count": len(publishable),
        "diagnostic_count": len(rows) - len(publishable),
        "median_linearization_error": float(np.median([row["linearization_error"] for row in rows])) if rows else float("nan"),
        "median_rel_err_central": float(np.median([row["rel_err_central"] for row in rows])) if rows else float("nan"),
        "rows": rows,
        "baseline_rows": baseline_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "sensitivity_audit.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "summary.csv", rows)
    _write_csv(OUT_DIR / "baseline_summary.csv", baseline_rows)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
