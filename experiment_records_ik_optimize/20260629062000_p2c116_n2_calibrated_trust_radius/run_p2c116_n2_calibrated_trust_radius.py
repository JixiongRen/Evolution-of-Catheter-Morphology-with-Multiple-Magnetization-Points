"""Calibrate the local current trust radius for the guarded n2 branch."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
P2C75_DIR = ROOT / "experiment_records_ik_optimize/20260628002500_p2c75_n2_state_increment_audit"
P2C98_DIR = ROOT / "experiment_records_ik_optimize/20260628213500_p2c98_n2_eps0125_random_current_batch"
P2C104_DIR = ROOT / "experiment_records_ik_optimize/20260629000000_p2c104_n2_cached_branch_audit"
for path in (ROOT, P2C75_DIR, P2C98_DIR, P2C104_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c75_n2_state_increment_audit as p2c75
import run_p2c98_n2_eps0125_random_current_batch as p2c98
import run_p2c104_n2_cached_branch_audit as p2c104


jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

RADII_A = (0.0025, 0.005, 0.0125)


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write CSV rows."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def jsonable(value):
    """Convert arrays and scalars into JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def output_arrays(funcs: dict, y: np.ndarray, I: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate output arrays for one branch state."""
    out = funcs["output"](jnp.asarray(y, dtype=jnp.float64), jnp.asarray(I, dtype=jnp.float64).reshape(8))
    return {
        "tip_p": np.asarray(jax.device_get(out.tip_p), dtype=float),
        "centerline": np.asarray(jax.device_get(out.centerline), dtype=float),
        "boundary": np.asarray(jax.device_get(out.boundary), dtype=float),
    }


def relative_error(pred: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Return relative error and cosine similarity."""
    pred = np.asarray(pred, dtype=float).reshape(-1)
    actual = np.asarray(actual, dtype=float).reshape(-1)
    rel = float(np.linalg.norm(pred - actual) / max(float(np.linalg.norm(actual)), 1.0e-14))
    denom = max(float(np.linalg.norm(pred) * np.linalg.norm(actual)), 1.0e-14)
    return rel, float(np.dot(pred, actual) / denom)


def build_directions(S: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Build principal and random current directions."""
    _u, _s, vt = np.linalg.svd(S, full_matrices=False)
    directions: list[tuple[str, np.ndarray]] = [(f"mode_{i + 1}", vt[i]) for i in range(3)]
    for name, direction in list(p2c98._directions())[:3]:
        directions.append((name, np.asarray(direction, dtype=float)))
    return directions


def plot_rows(rows: list[dict]) -> None:
    """Plot calibrated trust-radius errors."""
    labels = [f"{row['direction']}_{row['radius_A']}" for row in rows]
    rel = [row["actual_rel_err"] for row in rows]
    colors = ["#207a3c" if row["status"] == "pass" else "#b33a3a" for row in rows]
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    ax.bar(labels, rel, color=colors)
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1.0, label="线性误差门限 0.05")
    ax.set_yscale("log")
    ax.set_ylabel("实际重求解相对误差")
    ax.set_title("n2 已审计小电流半径下的局部线性闭环")
    ax.tick_params(axis="x", rotation=55)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.savefig(OUT_DIR / "n2_calibrated_trust_radius_cn.png", dpi=220)
    plt.close(fig)


def main() -> None:
    """Run calibrated trust-radius closure checks."""
    t0 = time.perf_counter()
    params, I0 = p2c104.p2c79.p2c57.p2c24._params_from_saved_case(
        p2c104.p2c79.p2c57.CASE_NAME,
        p2c104.p2c79.p2c57.ALPHA,
    )
    branch = p2c104._load_cached_branch(params)
    y_base = np.asarray(branch["y"], dtype=float)
    basis = np.asarray(branch["basis"], dtype=float)
    S = np.asarray(branch["S"], dtype=float)
    funcs = branch["funcs"]
    I0 = np.asarray(I0, dtype=float)
    base_out = output_arrays(funcs, y_base, I0)

    rows: list[dict] = []
    failures: list[dict] = []
    for name, direction in build_directions(S):
        direction = np.asarray(direction, dtype=float).reshape(8)
        direction /= np.linalg.norm(direction)
        for radius in RADII_A:
            dI = radius * direction
            pred = S @ dI
            I_new = I0 + dI
            q, info = p2c75._solve_q(y_base=y_base, basis=basis, I=I_new, funcs=funcs)
            y_new = y_base + basis @ q
            out_new = output_arrays(funcs, y_new, I_new)
            actual = out_new["tip_p"] - base_out["tip_p"]
            rel, cosine = relative_error(pred, actual)
            centerline_change = float(np.max(np.linalg.norm(out_new["centerline"] - base_out["centerline"], axis=1)))
            passed = bool(info["success"] and rel < 0.05 and centerline_change < 1.0e-3)
            if not passed:
                failures.append(
                    {
                        "direction": name,
                        "radius_A": radius,
                        "failure_class": "calibrated_radius_failed",
                        "rel_err": rel,
                        "centerline_change_m": centerline_change,
                        "solve_success": bool(info["success"]),
                    }
                )
            rows.append(
                {
                    "direction": name,
                    "radius_A": radius,
                    "status": "pass" if passed else "fail",
                    "predicted_norm_um": float(np.linalg.norm(pred) * 1.0e6),
                    "actual_norm_um": float(np.linalg.norm(actual) * 1.0e6),
                    "actual_rel_err": rel,
                    "actual_cosine": cosine,
                    "centerline_max_change_m": centerline_change,
                    "boundary_norm": float(np.linalg.norm(out_new["boundary"])),
                    "solve_success": bool(info["success"]),
                    "solve_nfev": int(info["nfev"]),
                    "solve_steady_s": float(info["steady_s"]),
                }
            )

    pass_count = sum(row["status"] == "pass" for row in rows)
    by_radius = {}
    for radius in RADII_A:
        part = [row for row in rows if row["radius_A"] == radius]
        by_radius[str(radius)] = {
            "passed": sum(row["status"] == "pass" for row in part),
            "total": len(part),
            "max_rel_err": max(row["actual_rel_err"] for row in part),
            "max_centerline_change_m": max(row["centerline_max_change_m"] for row in part),
        }
    summary = {
        "experiment_id": OUT_DIR.name,
        "status": "calibrated_radius_pass" if pass_count == len(rows) else "calibrated_radius_partial",
        "tested_cases": len(rows),
        "pass_count": pass_count,
        "radii_A": list(RADII_A),
        "by_radius": by_radius,
        "wall_s": time.perf_counter() - t0,
        "total_solve_steady_s": sum(row["solve_steady_s"] for row in rows),
        "interpretation": "该实验检验 P2c107 已审计扰动尺度附近的真实重求解闭环，用于估计当前 n2 分支可线性化电流半径。",
    }
    write_csv(OUT_DIR / "radius_rows.csv", rows)
    write_csv(OUT_DIR / "summary.csv", [{"pass_count": pass_count, "tested_cases": len(rows), "wall_s": summary["wall_s"]}])
    (OUT_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(jsonable(row), ensure_ascii=False) + "\n" for row in failures),
        encoding="utf-8",
    )
    plot_rows(rows)

    radius_lines = "\n".join(
        f"- 半径 {radius} A：{item['passed']}/{item['total']} 通过，最大相对误差 {item['max_rel_err']:.3e}，最大中心线变化 {item['max_centerline_change_m']:.3e} m。"
        for radius, item in by_radius.items()
    )
    record = f"""# P2c116 n2 已审计小电流半径闭环验证

## 实验目的

P2c115 证明 5 A 不是当前可信分支的有效线性信赖域。本轮将电流半径回退到 P2c107/P2c104 已审计的扰动尺度附近，检验 `0.0025 A`、`0.005 A` 和 `0.0125 A` 下的真实重求解闭环。

## 方法

测试方向包括三条主灵敏方向和三条随机电流方向。对每个方向和半径，计算线性预测 \\(S\\Delta I\\)，再在同一可信分支有效子空间内重求解状态，比较真实末端位移与线性预测。通过标准为：状态重求解成功、相对误差小于 0.05、中心线最大变化小于 \\(10^{{-3}}\\,\\mathrm{{m}}\\)。

## 结果

- 总样本数：{len(rows)}。
- 通过数：{pass_count}。
{radius_lines}
- 累计重求解耗时：{summary["total_solve_steady_s"]:.2f} s。
- 墙钟耗时：{summary["wall_s"]:.2f} s。

## 结论

该轮用于界定 n2 分支的可线性化电流半径。若 `0.0125 A` 全部通过，则此前随机方向审计尺度可以作为初始保守信赖域；若只在更小半径通过，则外层 IK 必须采用更小电流步长并逐步夹逼。无论哪种情况，P2c115 已否定 5 A 级别的单步线性外推。

## 下一步计划

若本轮存在通过半径，下一步应将最大通过半径写入只读接口契约，并把 P2c114 的 5 A 工程预算拆分为“物理电流限幅”和“局部线性信赖域半径”两个不同概念。
"""
    (OUT_DIR / "record.md").write_text(record, encoding="utf-8")
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
