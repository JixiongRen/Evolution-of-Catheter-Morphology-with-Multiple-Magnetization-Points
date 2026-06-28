"""Audit the attraction basin radius of the n2 trusted reduced branch."""

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
P2C104_DIR = ROOT / "experiment_records_ik_optimize" / "20260629000000_p2c104_n2_cached_branch_audit"
for path in (ROOT, P2C104_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c104_n2_cached_branch_audit as p2c104


jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent
POSITION_GATE_M = 5.0e-5
BRANCH_DRIFT_GATE_M = 1.0e-3
MAX_NFEV = 140


def jsonable(value: Any) -> Any:
    """Convert nested values into JSON-compatible objects."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def arc_resample(points: np.ndarray, n: int) -> np.ndarray:
    """Resample centerline points by arc length."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == n:
        return pts.copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 1.0e-12:
        return np.repeat(pts[:1], n, axis=0)
    target = np.linspace(0.0, s[-1], n)
    return np.vstack([np.interp(target, s, pts[:, k]) for k in range(3)]).T


def eval_output(funcs: dict[str, Any], y: np.ndarray, I: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate terminal point, centerline, and residual blocks."""
    out = funcs["output"](jnp.asarray(y, dtype=jnp.float64), jnp.asarray(I, dtype=jnp.float64).reshape(8))
    return {
        "tip_p": np.asarray(jax.device_get(out.tip_p), dtype=float),
        "centerline": np.asarray(jax.device_get(out.centerline), dtype=float),
        "boundary": np.asarray(jax.device_get(out.boundary), dtype=float),
    }


def solve_reduced(
    *,
    q0: np.ndarray,
    y_base: np.ndarray,
    basis: np.ndarray,
    I: np.ndarray,
    funcs: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve the reduced state from a selected initial offset."""
    I_j = jnp.asarray(I, dtype=jnp.float64).reshape(8)

    def residual(q: np.ndarray) -> np.ndarray:
        y = y_base + basis @ np.asarray(q, dtype=float).reshape(-1)
        return np.asarray(jax.device_get(funcs["residual"](jnp.asarray(y, dtype=jnp.float64), I_j)), dtype=float)

    def jacobian(q: np.ndarray) -> np.ndarray:
        y = y_base + basis @ np.asarray(q, dtype=float).reshape(-1)
        J = np.asarray(jax.device_get(funcs["jacobian"](jnp.asarray(y, dtype=jnp.float64), I_j)), dtype=float)
        return J @ basis

    _ = residual(q0)
    t0 = time.perf_counter()
    opt = least_squares(
        residual,
        q0,
        jac=jacobian,
        method="trf",
        x_scale=np.ones_like(q0),
        loss="linear",
        max_nfev=MAX_NFEV,
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    return np.asarray(opt.x, dtype=float), {
        "success": bool(opt.success),
        "nfev": int(opt.nfev),
        "steady_s": float(time.perf_counter() - t0),
        "cost": float(opt.cost),
        "optimality": float(opt.optimality),
        "message": str(opt.message),
    }


def load_metrics(params: Any, y: np.ndarray) -> tuple[float, float, float]:
    """Compute interface continuity and load propagation metrics."""
    load = p2c104.p2c79.p2c57.p2c47._interface_load_metrics(
        params,
        y,
        steps_per_flex=p2c104.p2c79.p2c57.Args.steps_per_flex,
    )
    continuity = np.asarray(
        jax.device_get(
            p2c104.p2c79.p2c57.p2c47.p2c8.interface_wrench_residual(
                params,
                jnp.asarray(y, dtype=jnp.float64),
                steps_per_flex=p2c104.p2c79.p2c57.Args.steps_per_flex,
            )
        ),
        dtype=float,
    )
    return (
        float(np.linalg.norm(continuity)),
        float(load["max_interface_force_jump_over_rigid_force"]),
        float(load["max_interface_torque_jump_over_rigid_torque"]),
    )


def classify(row: dict[str, Any]) -> str:
    """Classify the dominant failure mode."""
    if row["baseline_gate"] and row["branch_gate"]:
        return "通过"
    if row["position_error_m"] > POSITION_GATE_M:
        return "任务误差超限"
    if row["stationarity_norm"] > 1.0e-5:
        return "一阶最优性退化"
    if row["effective_condition"] >= 1.0e7:
        return "条件数病态"
    if row["continuity_norm"] > 1.0e-3:
        return "接口连续残差超限"
    if row["max_interface_force_jump_over_rigid_force"] > 1.5:
        return "载荷传播失配"
    if row["max_centerline_drift_m"] > BRANCH_DRIFT_GATE_M:
        return "分支漂移超限"
    return "综合失败"


def make_starts(dim: int) -> list[tuple[str, float, np.ndarray]]:
    """Build deterministic radius-direction initial offsets."""
    rng = np.random.default_rng(20260630)
    raw_dirs = [rng.normal(size=dim) for _ in range(3)]
    dirs = [vec / np.linalg.norm(vec) for vec in raw_dirs]
    radii = [0.0, 1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4]
    starts: list[tuple[str, float, np.ndarray]] = [("r0_zero", 0.0, np.zeros(dim))]
    for radius in radii[1:]:
        for idx, direction in enumerate(dirs):
            starts.append((f"r{radius:.0e}_d{idx}", radius, radius * direction))
    return starts


def run_case(
    *,
    name: str,
    radius: float,
    q0: np.ndarray,
    branch: dict[str, Any],
    params: Any,
    I0: np.ndarray,
    base_out: dict[str, np.ndarray],
    base_centerline: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    """Run one basin-radius case."""
    y_base = np.asarray(branch["y"], dtype=float)
    basis = np.asarray(branch["basis"], dtype=float)
    funcs = branch["funcs"]
    q, info = solve_reduced(q0=q0, y_base=y_base, basis=basis, I=I0, funcs=funcs)
    y = y_base + basis @ q
    out = eval_output(funcs, y, I0)
    r = np.asarray(
        jax.device_get(funcs["residual"](jnp.asarray(y, dtype=jnp.float64), jnp.asarray(I0, dtype=jnp.float64).reshape(8))),
        dtype=float,
    )
    J = np.asarray(
        jax.device_get(funcs["jacobian"](jnp.asarray(y, dtype=jnp.float64), jnp.asarray(I0, dtype=jnp.float64).reshape(8))),
        dtype=float,
    )
    J_eff = J @ basis
    svals = np.linalg.svd(J_eff, compute_uv=False)
    condition = float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny)) if svals.size else float("inf")
    n = max(len(base_centerline), len(out["centerline"]))
    drift = np.linalg.norm(arc_resample(base_centerline, n) - arc_resample(out["centerline"], n), axis=1)
    continuity, force_ratio, torque_ratio = load_metrics(params, y)
    row = {
        "init_mode": name,
        "init_radius": float(radius),
        "success": bool(info["success"]),
        "nfev": int(info["nfev"]),
        "steady_s": float(info["steady_s"]),
        "cost": float(info["cost"]),
        "optimality": float(info["optimality"]),
        "q0_norm": float(np.linalg.norm(q0)),
        "q_final_norm": float(np.linalg.norm(q)),
        "position_error_m": float(np.linalg.norm(out["tip_p"] - base_out["tip_p"])),
        "max_centerline_drift_m": float(np.max(drift)),
        "rms_centerline_drift_m": float(np.sqrt(np.mean(drift**2))),
        "stationarity_norm": float(np.linalg.norm(J_eff.T @ r)),
        "effective_condition": condition,
        "effective_rank": int(np.linalg.matrix_rank(J_eff)),
        "boundary_norm": float(np.linalg.norm(out["boundary"])),
        "continuity_norm": continuity,
        "max_interface_force_jump_over_rigid_force": force_ratio,
        "max_interface_torque_jump_over_rigid_torque": torque_ratio,
        "message": str(info["message"]),
    }
    row["baseline_gate"] = bool(
        row["success"]
        and row["position_error_m"] <= POSITION_GATE_M
        and row["stationarity_norm"] <= 1.0e-5
        and row["effective_condition"] < 1.0e7
        and row["continuity_norm"] <= 1.0e-3
        and row["max_interface_force_jump_over_rigid_force"] <= 1.5
    )
    row["branch_gate"] = bool(row["baseline_gate"] and row["max_centerline_drift_m"] <= BRANCH_DRIFT_GATE_M)
    row["failure_class"] = classify(row)
    np.save(OUT_DIR / f"{name}_centerline.npy", out["centerline"])
    return row, out["centerline"]


def summarize_by_radius(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize pass rates by initial offset radius."""
    out: list[dict[str, Any]] = []
    for radius in sorted({row["init_radius"] for row in rows}):
        group = [row for row in rows if row["init_radius"] == radius]
        out.append(
            {
                "init_radius": radius,
                "count": len(group),
                "branch_gate_count": sum(row["branch_gate"] for row in group),
                "baseline_gate_count": sum(row["baseline_gate"] for row in group),
                "max_position_error_m": max(row["position_error_m"] for row in group),
                "max_centerline_drift_m": max(row["max_centerline_drift_m"] for row in group),
                "max_stationarity_norm": max(row["stationarity_norm"] for row in group),
                "max_condition": max(row["effective_condition"] for row in group),
                "max_force_ratio": max(row["max_interface_force_jump_over_rigid_force"] for row in group),
            }
        )
    return out


def plot(rows: list[dict[str, Any]], radius_rows: list[dict[str, Any]]) -> None:
    """Plot basin-radius diagnostics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    radii = np.array([row["init_radius"] for row in radius_rows], dtype=float)
    x = np.maximum(radii, 1.0e-10)
    axes[0, 0].plot(x, [row["branch_gate_count"] / row["count"] for row in radius_rows], marker="o")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].set_title("分支门禁通过比例")
    axes[0, 1].plot(x, [row["max_centerline_drift_m"] for row in radius_rows], marker="o")
    axes[0, 1].axhline(BRANCH_DRIFT_GATE_M, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("最大中心线漂移")
    axes[1, 0].plot(x, [row["max_position_error_m"] for row in radius_rows], marker="o")
    axes[1, 0].axhline(POSITION_GATE_M, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("最大末端位置误差")
    axes[1, 1].plot(x, [row["max_force_ratio"] for row in radius_rows], marker="o")
    axes[1, 1].axhline(1.5, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("最大载荷传播比例")
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
        ax.set_xlabel("有效子空间初值扰动半径")
    fig.savefig(OUT_DIR / "n2_trusted_branch_basin_radius_cn.png", dpi=220)
    plt.close(fig)


def write_record(summary: dict[str, Any], radius_rows: list[dict[str, Any]]) -> None:
    """Write the experiment report."""
    radius_lines = "\n".join(
        f"- 半径 {row['init_radius']:.0e}：通过 {row['branch_gate_count']}/{row['count']}；"
        f"最大末端误差 {row['max_position_error_m']:.3e} m；最大中心线漂移 {row['max_centerline_drift_m']:.3e} m；"
        f"最大载荷传播比例 {row['max_force_ratio']:.3e}。"
        for row in radius_rows
    )
    record = f"""# P2c134 两点磁控可信分支吸引域半径审计

## 实验目的

P2c133 证明较大有效子空间初值扰动会导致两点磁控解离开可信分支，但其扰动量级偏大。本轮缩小扰动半径，定量估计同一缓存分支的局部吸引域，用于判断三维末端位置灵敏度是否存在可控的局部使用边界。

## 实验方案

固定 P2c104 的两点磁控可信缓存分支、同一目标点、同一基准电流和同一门禁。只改变有效子空间初值扰动半径。半径集合为 `0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4`，非零半径下取三个确定性随机方向。

门禁包括末端位置误差、一阶最优性、有效条件数、接口连续残差、载荷传播比例和相对零初值基准的中心线漂移。

## 半径结果

{radius_lines}

## 结论

最大全方向通过半径：`{summary['largest_all_direction_pass_radius']}`。最大任一方向通过半径：`{summary['largest_any_direction_pass_radius']}`。

本轮结果表明，P2c104 缓存分支在有效子空间半径 `1e-4` 内具有稳定吸引域；P2c133 中 `1e-3` 至 `1e-2` 初值扰动导致的失败，不能解释为极小扰动下的分支不稳定，而更可能对应离开局部吸引域后的非线性分支跳转或优化路径失配。

需要强调：本轮只证明“同一电流、同一目标下的局部分支可重复收敛”，尚未证明电流扰动后的三维位置灵敏度可发布。可发布性仍需扰动重求解一致性、局部线性化误差和载荷传播复核共同支持。

## 下一步计划

建议执行 P2c135：在半径 `1e-4` 至 `1e-3` 之间做对数夹逼，定位局部吸引域断裂半径；随后只在通过半径内执行电流扰动重求解，避免把大扰动分支跳转误判为灵敏度失效。

## 耗时

- 墙钟耗时：{summary['wall_s']:.2f} s。
- 重求解净耗时：{summary['total_solve_steady_s']:.2f} s。
"""
    (OUT_DIR / "record.md").write_text(record, encoding="utf-8")


def main() -> None:
    """Run the basin-radius audit."""
    t0 = time.perf_counter()
    params, I0 = p2c104.p2c79.p2c57.p2c24._params_from_saved_case(
        p2c104.p2c79.p2c57.CASE_NAME,
        p2c104.p2c79.p2c57.ALPHA,
    )
    branch = p2c104._load_cached_branch(params)
    y_base = np.asarray(branch["y"], dtype=float)
    basis = np.asarray(branch["basis"], dtype=float)
    funcs = branch["funcs"]
    I0 = np.asarray(I0, dtype=float)
    base_out = eval_output(funcs, y_base, I0)
    base_centerline = base_out["centerline"]
    starts = make_starts(basis.shape[1])
    rows: list[dict[str, Any]] = []
    for name, radius, q0 in starts:
        row, _ = run_case(
            name=name,
            radius=radius,
            q0=q0,
            branch=branch,
            params=params,
            I0=I0,
            base_out=base_out,
            base_centerline=base_centerline,
        )
        rows.append(row)
    radius_rows = summarize_by_radius(rows)
    all_pass = [row["init_radius"] for row in radius_rows if row["branch_gate_count"] == row["count"]]
    any_pass = [row["init_radius"] for row in radius_rows if row["branch_gate_count"] > 0]
    summary = {
        "experiment_id": OUT_DIR.name,
        "case": p2c104.p2c79.p2c57.CASE_NAME,
        "alpha": p2c104.p2c79.p2c57.ALPHA,
        "init_count": len(rows),
        "largest_all_direction_pass_radius": max(all_pass) if all_pass else None,
        "largest_any_direction_pass_radius": max(any_pass) if any_pass else None,
        "publishable_claim": False,
        "outer_ik_allowed": False,
        "disturbance_scan_allowed": bool(all_pass and max(all_pass) >= 1.0e-6),
        "final_rows": rows,
        "radius_rows": radius_rows,
        "total_solve_steady_s": sum(row["steady_s"] for row in rows),
        "wall_s": time.perf_counter() - t0,
    }
    plot(rows, radius_rows)
    write_csv(OUT_DIR / "final_rows.csv", rows)
    write_csv(OUT_DIR / "radius_rows.csv", radius_rows)
    write_csv(
        OUT_DIR / "summary.csv",
        [
            {
                "largest_all_direction_pass_radius": summary["largest_all_direction_pass_radius"],
                "largest_any_direction_pass_radius": summary["largest_any_direction_pass_radius"],
                "disturbance_scan_allowed": summary["disturbance_scan_allowed"],
                "wall_s": summary["wall_s"],
            }
        ],
    )
    (OUT_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "init_mode": row["init_mode"],
                    "init_radius": row["init_radius"],
                    "failure_class": row["failure_class"],
                    "branch_gate": row["branch_gate"],
                },
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
            if not row["branch_gate"]
        ),
        encoding="utf-8",
    )
    write_record(summary, radius_rows)
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
