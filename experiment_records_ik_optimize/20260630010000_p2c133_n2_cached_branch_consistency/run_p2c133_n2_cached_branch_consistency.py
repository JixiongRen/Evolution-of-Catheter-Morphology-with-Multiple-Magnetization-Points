"""Check n2 cached-branch consistency from multiple reduced-space starts."""

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
P2C75_DIR = ROOT / "experiment_records_ik_optimize" / "20260628002500_p2c75_n2_state_increment_audit"
P2C104_DIR = ROOT / "experiment_records_ik_optimize" / "20260629000000_p2c104_n2_cached_branch_audit"
for path in (ROOT, P2C75_DIR, P2C104_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c75_n2_state_increment_audit as p2c75
import run_p2c104_n2_cached_branch_audit as p2c104


jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent
BRANCH_DRIFT_GATE_M = 1.0e-3
POSITION_GATE_M = 5.0e-5
MAX_NFEV = 180


def jsonable(value: Any) -> Any:
    """Convert values into JSON-compatible objects."""
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
    """Write CSV rows."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def arc_resample(points: np.ndarray, n: int) -> np.ndarray:
    """Resample a centerline by arc length."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == n:
        return pts.copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 1.0e-12:
        return np.repeat(pts[:1], n, axis=0)
    target = np.linspace(0.0, s[-1], n)
    return np.vstack([np.interp(target, s, pts[:, k]) for k in range(3)]).T


def output_arrays(funcs: dict[str, Any], y: np.ndarray, I: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate endpoint and centerline outputs."""
    out = funcs["output"](jnp.asarray(y, dtype=jnp.float64), jnp.asarray(I, dtype=jnp.float64).reshape(8))
    return {
        "tip_p": np.asarray(jax.device_get(out.tip_p), dtype=float),
        "centerline": np.asarray(jax.device_get(out.centerline), dtype=float),
        "boundary": np.asarray(jax.device_get(out.boundary), dtype=float),
    }


def solve_q_from(
    *,
    q0: np.ndarray,
    y_base: np.ndarray,
    basis: np.ndarray,
    I: np.ndarray,
    funcs: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve the reduced state from a supplied initial q."""
    I_j = jnp.asarray(I, dtype=jnp.float64).reshape(8)
    y_base = np.asarray(y_base, dtype=float).reshape(-1)
    basis = np.asarray(basis, dtype=float)
    q0 = np.asarray(q0, dtype=float).reshape(basis.shape[1])

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


def classify(row: dict[str, Any]) -> str:
    """Classify the dominant gate failure."""
    if row["baseline_gate"]:
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
    return "综合失败"


def build_q_starts(dim: int) -> dict[str, np.ndarray]:
    """Build deterministic reduced-space initial guesses."""
    rng = np.random.default_rng(20260630)
    starts = {
        "zero_q": np.zeros(dim),
        "small_random_pos": 1.0e-3 * rng.normal(size=dim),
        "small_random_neg": -1.0e-3 * rng.normal(size=dim),
        "medium_random": 1.0e-2 * rng.normal(size=dim),
    }
    if dim:
        e0 = np.zeros(dim)
        e0[0] = 1.0e-2
        starts["basis_axis_pos"] = e0
        starts["basis_axis_neg"] = -e0
    return starts


def run_start(name: str, q0: np.ndarray, branch: dict[str, Any], params: Any, I0: np.ndarray, base_out: dict[str, np.ndarray]) -> tuple[dict[str, Any], np.ndarray]:
    """Run one reduced-space start."""
    y_base = np.asarray(branch["y"], dtype=float)
    basis = np.asarray(branch["basis"], dtype=float)
    funcs = branch["funcs"]
    q, info = solve_q_from(q0=q0, y_base=y_base, basis=basis, I=I0, funcs=funcs)
    y = y_base + basis @ q
    out = output_arrays(funcs, y, I0)
    r = np.asarray(jax.device_get(funcs["residual"](jnp.asarray(y, dtype=jnp.float64), jnp.asarray(I0, dtype=jnp.float64).reshape(8))), dtype=float)
    J = np.asarray(jax.device_get(funcs["jacobian"](jnp.asarray(y, dtype=jnp.float64), jnp.asarray(I0, dtype=jnp.float64).reshape(8))), dtype=float)
    J_eff = J @ basis
    svals = np.linalg.svd(J_eff, compute_uv=False)
    condition = float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny)) if svals.size else float("inf")
    boundary_norm = float(np.linalg.norm(out["boundary"]))
    position_error = float(np.linalg.norm(out["tip_p"] - base_out["tip_p"]))
    stationarity = float(np.linalg.norm(J_eff.T @ r))
    load = p2c104.p2c79.p2c57.p2c47._interface_load_metrics(params, y, steps_per_flex=p2c104.p2c79.p2c57.Args.steps_per_flex)
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
    row = {
        "init_mode": name,
        "success": bool(info["success"]),
        "nfev": int(info["nfev"]),
        "steady_s": float(info["steady_s"]),
        "cost": float(info["cost"]),
        "optimality": float(info["optimality"]),
        "q_norm": float(np.linalg.norm(q)),
        "position_error_m": position_error,
        "stationarity_norm": stationarity,
        "effective_condition": condition,
        "effective_rank": int(np.linalg.matrix_rank(J_eff)),
        "boundary_norm": boundary_norm,
        "continuity_norm": float(np.linalg.norm(continuity)),
        "max_interface_force_jump_over_rigid_force": float(load["max_interface_force_jump_over_rigid_force"]),
        "max_interface_torque_jump_over_rigid_torque": float(load["max_interface_torque_jump_over_rigid_torque"]),
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
    row["failure_class"] = classify(row)
    np.save(OUT_DIR / f"{name}_centerline.npy", out["centerline"])
    np.save(OUT_DIR / f"{name}_q.npy", q)
    return row, out["centerline"]


def pairwise_rows(centerlines: dict[str, np.ndarray], final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute pairwise branch drift."""
    row_by_name = {row["init_mode"]: row for row in final_rows}
    names = list(centerlines)
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            n = max(len(centerlines[a]), len(centerlines[b]))
            ca = arc_resample(centerlines[a], n)
            cb = arc_resample(centerlines[b], n)
            d = np.linalg.norm(ca - cb, axis=1)
            both = bool(row_by_name[a]["baseline_gate"] and row_by_name[b]["baseline_gate"])
            rows.append(
                {
                    "init_a": a,
                    "init_b": b,
                    "both_baseline_gate": both,
                    "max_centerline_drift_m": float(np.max(d)),
                    "rms_centerline_drift_m": float(np.sqrt(np.mean(d**2))),
                    "tip_drift_m": float(np.linalg.norm(ca[-1] - cb[-1])),
                    "branch_gate": bool(both and np.max(d) <= BRANCH_DRIFT_GATE_M),
                }
            )
    return rows


def plot(final_rows: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> None:
    """Plot branch gates."""
    labels = [row["init_mode"] for row in final_rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes[0].bar(labels, [row["effective_condition"] for row in final_rows], color="#2a9d8f")
    axes[0].axhline(1.0e7, color="black", linestyle="--", linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_title("有效子空间条件数")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.25)
    labels2 = [f"{row['init_a']}\n{row['init_b']}" for row in pairs]
    axes[1].bar(labels2, [row["max_centerline_drift_m"] for row in pairs], color="#457b9d")
    axes[1].axhline(BRANCH_DRIFT_GATE_M, color="black", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_title("两两中心线最大漂移")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(OUT_DIR / "n2_cached_branch_consistency_cn.png", dpi=220)
    plt.close(fig)


def write_record(summary: dict[str, Any], rows: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> None:
    """Write the experiment record."""
    result_lines = "\n".join(
        f"- {row['init_mode']}：基线门禁 {row['baseline_gate']}；位置误差 {row['position_error_m']:.3e} m；"
        f"一阶最优性 {row['stationarity_norm']:.3e}；条件数 {row['effective_condition']:.3e}；失败分类 {row['failure_class']}。"
        for row in rows
    )
    pair_lines = "\n".join(
        f"- {row['init_a']} vs {row['init_b']}：两者均过基线 {row['both_baseline_gate']}；最大中心线漂移 {row['max_centerline_drift_m']:.3e} m；分支门禁 {row['branch_gate']}。"
        for row in pairs
    )
    record = f"""# P2c133 两点磁控缓存分支一致性复核

## 实验目的

P2c125 已将两点磁控定位为“同一可信分支下的局部位置诊断”。本轮检查 P2c104 缓存可信分支在有效子空间内是否对不同初值稳定收敛，为后续三维位置灵敏度发布边界提供分支一致性证据。

## 实验方案

固定 P2c104 的两点磁控可信缓存分支、同一目标点和同一基准电流。在有效子空间中设置零初值、小随机正负扰动、中等随机扰动、第一基向量正负扰动，共六个初值。每个初值重求解同一三维末端位置任务，并比较最终中心线。

通过标准：每条路径需通过位置误差、一阶最优性、有效条件数、接口连续残差和载荷传播门禁；至少一对均过基线路径的中心线最大漂移不超过 `1e-3 m`。

## 基线结果

{result_lines}

## 分支漂移

{pair_lines}

## 结论

分支一致性结论：{summary['branch_status']}。若通过，则 n2 可继续进入扰动重求解和局部线性化复核；若不通过，则维持“方向诊断可用、三维发布不足”的结论。

## 下一步计划

若本轮通过，执行 P2c134：围绕该缓存分支做三维随机方向小扰动重求解复核，并报告局部线性化误差分布。若未通过，继续维持 P2c125 的方向性工程接口，不升级三维位置发布。

## 耗时

- 墙钟耗时：{summary['wall_s']:.2f} s。
- 重求解净耗时：{summary['total_solve_steady_s']:.2f} s。
"""
    (OUT_DIR / "record.md").write_text(record, encoding="utf-8")


def main() -> None:
    """Run cached-branch consistency checks."""
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
    base_out = output_arrays(funcs, y_base, I0)
    starts = build_q_starts(basis.shape[1])
    rows: list[dict[str, Any]] = []
    centerlines: dict[str, np.ndarray] = {}
    for name, q0 in starts.items():
        row, centerline = run_start(name, q0, branch, params, I0, base_out)
        rows.append(row)
        centerlines[name] = centerline
    pairs = pairwise_rows(centerlines, rows)
    pass_pairs = [row for row in pairs if row["branch_gate"]]
    branch_status = "通过" if pass_pairs else "未通过"
    plot(rows, pairs)
    summary = {
        "experiment_id": OUT_DIR.name,
        "case": p2c104.p2c79.p2c57.CASE_NAME,
        "alpha": p2c104.p2c79.p2c57.ALPHA,
        "init_count": len(rows),
        "baseline_pass_count": sum(row["baseline_gate"] for row in rows),
        "branch_pair_pass_count": len(pass_pairs),
        "branch_status": branch_status,
        "publishable_claim": False,
        "outer_ik_allowed": False,
        "disturbance_scan_allowed": bool(pass_pairs),
        "final_rows": rows,
        "branch_rows": pairs,
        "total_solve_steady_s": sum(row["steady_s"] for row in rows),
        "wall_s": time.perf_counter() - t0,
    }
    write_csv(OUT_DIR / "final_rows.csv", rows)
    write_csv(OUT_DIR / "branch_pair_rows.csv", pairs)
    write_csv(OUT_DIR / "summary.csv", [{"branch_status": branch_status, "baseline_pass_count": summary["baseline_pass_count"], "branch_pair_pass_count": len(pass_pairs), "wall_s": summary["wall_s"]}])
    (OUT_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(
            json.dumps({"init_mode": row["init_mode"], "failure_class": row["failure_class"], "baseline_gate": row["baseline_gate"]}, ensure_ascii=False) + "\n"
            for row in rows
            if not row["baseline_gate"]
        ),
        encoding="utf-8",
    )
    write_record(summary, rows, pairs)
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
