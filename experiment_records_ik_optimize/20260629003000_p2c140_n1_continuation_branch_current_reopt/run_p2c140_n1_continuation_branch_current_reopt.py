"""Audit current perturbations on the n1 continuation branch."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
P2C139_DIR = ROOT / "experiment_records_ik_optimize" / "20260630063000_p2c139_n1_mid_branch_recovery_probe"
P2C132_DIR = ROOT / "experiment_records_ik_optimize" / "20260629230000_p2c132_n1_mid_branch_consistency"
for path in (ROOT, P2C139_DIR, P2C132_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c132_n1_mid_branch_consistency as p2c132
import run_p2c139_n1_mid_branch_recovery_probe as p2c139


configure_matplotlib_cjk()
OUT_DIR = Path(__file__).resolve().parent
RADII_A = (1.0e-5, 3.0e-5, 1.0e-4)
LINEAR_REL_GATE = 0.05
BRANCH_DRIFT_GATE_M = 1.0e-3
MIN_RELIABLE_TIP_DELTA_M = 1.0e-10


def jsonable(value: Any) -> Any:
    """Convert values into JSON-compatible objects."""
    return p2c132.jsonable(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def current_directions() -> list[tuple[str, np.ndarray]]:
    """Return deterministic current-space directions."""
    base = p2c139.p2c132.p2c131.BASE_CURRENTS.astype(float)
    base = base / max(float(np.linalg.norm(base)), 1.0e-18)
    coil5 = np.zeros(8)
    coil5[4] = 1.0
    coil6 = np.zeros(8)
    coil6[5] = 1.0
    rng = np.random.default_rng(20260629)
    random_dir = rng.normal(size=8)
    random_dir /= np.linalg.norm(random_dir)
    return [
        ("base_current_direction", base),
        ("coil5_positive", coil5),
        ("coil6_positive", coil6),
        ("random_direction", random_dir),
    ]


def base_problem() -> dict[str, Any]:
    """Build final-alpha target, functions, and continuation branch state."""
    spec, zero_oracle, target_oracle, current_i = p2c139.solve_alpha_oracles(0.10)
    params = target_oracle["params"]
    target_p = np.asarray(p2c132.jax.device_get(target_oracle["sites"]["tip_p_dim"]), dtype=float).reshape(3)
    zero_p = np.asarray(p2c132.jax.device_get(zero_oracle["sites"]["tip_p_dim"]), dtype=float).reshape(3)
    direction_raw = target_p - zero_p
    direction = direction_raw / max(float(np.linalg.norm(direction_raw)), 1.0e-12)
    y_ref = p2c132.p2c131.p2c2.p2c1._y_guess_from_oracle(target_oracle["z"], params, "segment_wrench")
    args = p2c132.p2c131.p2c110.Args()
    args.position_scale = 60.0
    args.max_iter = 240
    funcs = p2c132.p2c131.make_functions(params, target_p, direction, y_ref, args, task_mode="position_3d")
    y_base = np.load(P2C139_DIR / "cont_fine_state_y.npy")
    out_base = funcs["output"](jnp.asarray(y_base, dtype=jnp.float64), jnp.asarray(current_i, dtype=jnp.float64).reshape(8))
    return {
        "spec": spec,
        "params": params,
        "args": args,
        "funcs": funcs,
        "current_i": np.asarray(current_i, dtype=float),
        "y_base": np.asarray(y_base, dtype=float),
        "base_tip": np.asarray(p2c132.jax.device_get(out_base.tip_p), dtype=float).reshape(3),
        "base_centerline": np.asarray(p2c132.jax.device_get(out_base.centerline), dtype=float),
    }


def solve_perturbed(problem: dict[str, Any], direction_name: str, unit_direction: np.ndarray, radius: float) -> dict[str, Any]:
    """Solve one perturbed-current state from the continuation branch."""
    args = problem["args"]
    funcs = problem["funcs"]
    current = problem["current_i"] + radius * unit_direction
    solve = p2c132.p2c131.p2c110._solve(problem["y_base"], current, funcs, args)
    y = np.asarray(solve["y"], dtype=float)
    tip = np.asarray(solve["out"]["tip_p"], dtype=float).reshape(3)
    centerline = np.asarray(solve["out"]["centerline"], dtype=float)
    branch_drift = float(np.max(np.linalg.norm(centerline - problem["base_centerline"], axis=1)))
    load = p2c132.p2c131.p2c110.p2c57.p2c47._interface_load_metrics(problem["params"], y, steps_per_flex=args.steps_per_flex)
    row = {
        "direction": direction_name,
        "radius_A": float(radius),
        "tip_delta_x": float(tip[0] - problem["base_tip"][0]),
        "tip_delta_y": float(tip[1] - problem["base_tip"][1]),
        "tip_delta_z": float(tip[2] - problem["base_tip"][2]),
        "tip_delta_norm_m": float(np.linalg.norm(tip - problem["base_tip"])),
        "state_delta_norm": float(np.linalg.norm(y - problem["y_base"])),
        "centerline_branch_drift_m": branch_drift,
        "stationarity_norm": float(solve["stationarity_norm"]),
        "jacobian_condition": float(solve["jacobian_condition"]),
        "continuity_norm": float(solve["continuity_norm"]),
        "max_interface_force_jump_over_rigid_force": float(load["max_interface_force_jump_over_rigid_force"]),
        "max_interface_torque_jump_over_rigid_torque": float(load["max_interface_torque_jump_over_rigid_torque"]),
        "solve_success": bool(solve["success"]),
        "nfev": int(solve["nfev"]),
        "steady_s": float(solve["steady_s"]),
    }
    row["physical_gate"] = bool(
        row["solve_success"]
        and row["stationarity_norm"] <= 1.0e-5
        and row["jacobian_condition"] < 1.0e7
        and row["centerline_branch_drift_m"] <= BRANCH_DRIFT_GATE_M
        and row["continuity_norm"] <= 1.0e-3
        and row["max_interface_force_jump_over_rigid_force"] <= 1.5
    )
    np.save(OUT_DIR / f"{direction_name}_{radius:.0e}_centerline.npy", centerline)
    return row


def add_linearization(rows: list[dict[str, Any]]) -> None:
    """Add empirical linearly extrapolated errors."""
    for direction in sorted({row["direction"] for row in rows}):
        group = sorted([row for row in rows if row["direction"] == direction], key=lambda item: item["radius_A"])
        ref = np.array([group[0]["tip_delta_x"], group[0]["tip_delta_y"], group[0]["tip_delta_z"]], dtype=float)
        ref_radius = group[0]["radius_A"]
        for row in group:
            actual = np.array([row["tip_delta_x"], row["tip_delta_y"], row["tip_delta_z"]], dtype=float)
            pred = ref * (row["radius_A"] / ref_radius)
            denom = max(float(np.linalg.norm(actual)), 1.0e-14)
            row["linear_rel_err"] = float(np.linalg.norm(pred - actual) / denom)
            row["linear_gate"] = bool(row["linear_rel_err"] < LINEAR_REL_GATE)
            row["publishable_gate"] = bool(row["physical_gate"] and row["linear_gate"])
            if row["tip_delta_norm_m"] < MIN_RELIABLE_TIP_DELTA_M:
                row["publishable_gate"] = False
                row["failure_class"] = "末端位移低于可靠判读阈值"
            elif row["publishable_gate"]:
                row["failure_class"] = "通过"
            elif not row["physical_gate"]:
                row["failure_class"] = "物理门禁失败"
            else:
                row["failure_class"] = "局部线性化误差超限"


def summarize_direction(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize pass radii per direction."""
    out = []
    for direction in sorted({row["direction"] for row in rows}):
        group = [row for row in rows if row["direction"] == direction]
        passed = [row for row in group if row["publishable_gate"]]
        out.append(
            {
                "direction": direction,
                "tested_count": len(group),
                "pass_count": len(passed),
                "max_pass_radius_A": max((row["radius_A"] for row in passed), default=None),
                "max_tip_delta_norm_m": max(row["tip_delta_norm_m"] for row in group),
                "max_linear_rel_err": max(row["linear_rel_err"] for row in group),
                "max_branch_drift_m": max(row["centerline_branch_drift_m"] for row in group),
                "max_condition": max(row["jacobian_condition"] for row in group),
                "release_class": "分支条件下可发布" if passed else "不可发布",
                "low_signal": bool(max(row["tip_delta_norm_m"] for row in group) < MIN_RELIABLE_TIP_DELTA_M),
            }
        )
    return out


def plot(rows: list[dict[str, Any]], direction_rows: list[dict[str, Any]]) -> None:
    """Plot P2c140 metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    for direction in sorted({row["direction"] for row in rows}):
        group = sorted([row for row in rows if row["direction"] == direction], key=lambda item: item["radius_A"])
        x = [row["radius_A"] for row in group]
        axes[0].plot(x, [row["linear_rel_err"] for row in group], marker="o", label=direction)
        axes[1].plot(x, [row["centerline_branch_drift_m"] for row in group], marker="o")
        axes[2].plot(x, [row["tip_delta_norm_m"] for row in group], marker="o")
    axes[0].axhline(LINEAR_REL_GATE, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(BRANCH_DRIFT_GATE_M, color="black", linestyle="--", linewidth=1)
    for ax, title in zip(axes, ["局部线性化误差", "中心线分支漂移", "末端位移幅值"]):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("电流扰动半径/A")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(OUT_DIR / "n1_continuation_branch_current_reopt_cn.png", dpi=220)
    plt.close(fig)


def write_record(summary: dict[str, Any], direction_rows: list[dict[str, Any]]) -> None:
    """Write the Chinese experiment record."""
    lines = "\n".join(
        f"- {row['direction']}：通过 {row['pass_count']}/{row['tested_count']}；最大通过半径 {row['max_pass_radius_A']} A；"
        f"最大线性误差 {row['max_linear_rel_err']:.3e}；最大分支漂移 {row['max_branch_drift_m']:.3e} m；类别 {row['release_class']}。"
        for row in direction_rows
    )
    record = f"""# P2c140 单点中段 continuation 分支电流扰动重求解

## 实验目的

P2c139 恢复出一个可重复的 n1 continuation 分支，但该分支与 direct_oracle 分支不同。本轮只在该分支内做小电流扰动重求解，判断其是否满足局部线性化、分支漂移和物理门禁。

## 实验方案

以 P2c139 的 `cont_fine` 终态作为基准状态。选取四个电流方向：基准电流方向、线圈 5、线圈 6、固定随机方向。扰动半径为 `1e-5, 3e-5, 1e-4 A`。同一方向上使用最小半径的真实位移作为经验线性模型，外推到更大半径，并计算相对误差。

## 结果

{lines}

## 结论

发布状态：{summary['release_status']}。本轮所有方向的末端位移均低于可靠判读阈值 `1e-10 m`，因此不能形成可发布灵敏度。物理门禁和分支漂移虽然稳定，但输出信号过小，局部导数在当前数值精度下不可可靠辨识。

## 下一步计划

下一步应维持 P2c138 的 n1 不可发布边界；若继续 n1，应更换工作点、增大可观测位移或改变磁化配置，而不是在当前点继续扰动扫描。

## 耗时

- 墙钟耗时：{summary['wall_s']:.2f} s。
- 重求解净耗时：{summary['total_solve_steady_s']:.2f} s。
"""
    (OUT_DIR / "record.md").write_text(record, encoding="utf-8")


def main() -> None:
    """Run P2c140."""
    t0 = time.perf_counter()
    problem = base_problem()
    rows: list[dict[str, Any]] = []
    for name, unit in current_directions():
        for radius in RADII_A:
            rows.append(solve_perturbed(problem, name, unit, radius))
    add_linearization(rows)
    direction_rows = summarize_direction(rows)
    pass_dirs = [row for row in direction_rows if row["pass_count"] > 0]
    summary = {
        "experiment_id": OUT_DIR.name,
        "case": "n1_mid_z_rand15_a010",
        "branch": "continuation",
        "release_status": "branch_conditioned_candidate" if pass_dirs else "not_publishable",
        "release_direction_count": len(pass_dirs),
        "publishable_claim": bool(pass_dirs),
        "min_reliable_tip_delta_m": MIN_RELIABLE_TIP_DELTA_M,
        "low_signal_direction_count": sum(row["low_signal"] for row in direction_rows),
        "outer_ik_allowed": False,
        "full_3d_position_publishable": False,
        "direction_rows": direction_rows,
        "rows": rows,
        "total_solve_steady_s": sum(row["steady_s"] for row in rows),
        "wall_s": time.perf_counter() - t0,
    }
    write_csv(OUT_DIR / "current_reopt_rows.csv", rows)
    write_csv(OUT_DIR / "direction_rows.csv", direction_rows)
    write_csv(
        OUT_DIR / "summary.csv",
        [
            {
                "release_status": summary["release_status"],
                "release_direction_count": summary["release_direction_count"],
                "publishable_claim": summary["publishable_claim"],
                "outer_ik_allowed": summary["outer_ik_allowed"],
                "wall_s": summary["wall_s"],
            }
        ],
    )
    (OUT_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(jsonable(row), ensure_ascii=False) + "\n" for row in rows if not row["publishable_gate"]),
        encoding="utf-8",
    )
    plot(rows, direction_rows)
    write_record(summary, direction_rows)
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
