"""Reoptimize trusted n2 branch after trust-region step proposals."""

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
P2C104_DIR = ROOT / "experiment_records_ik_optimize/20260629000000_p2c104_n2_cached_branch_audit"
P2C114_DIR = ROOT / "experiment_records_ik_optimize/20260629050000_p2c114_n2_trust_region_step_filter"
for path in (ROOT, P2C75_DIR, P2C104_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c75_n2_state_increment_audit as p2c75
import run_p2c104_n2_cached_branch_audit as p2c104


jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV rows."""
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def norm_rel(pred: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Return relative error and cosine similarity."""
    pred = np.asarray(pred, dtype=float).reshape(-1)
    actual = np.asarray(actual, dtype=float).reshape(-1)
    rel = float(np.linalg.norm(pred - actual) / max(float(np.linalg.norm(actual)), 1.0e-14))
    denom = max(float(np.linalg.norm(pred) * np.linalg.norm(actual)), 1.0e-14)
    cos = float(np.dot(pred, actual) / denom)
    return rel, cos


def plot_rows(rows: list[dict]) -> None:
    """Plot closure errors after branch reoptimization."""
    names = [row["case"] for row in rows]
    rel = [row["actual_rel_err"] for row in rows]
    center = [row["centerline_max_change_m"] for row in rows]
    colors = ["#207a3c" if row["closure_status"] == "pass" else "#b33a3a" for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axes[0].bar(names, rel, color=colors)
    axes[0].axhline(0.05, color="black", linestyle="--", linewidth=1.0, label="线性误差门限 0.05")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("实际重求解相对误差")
    axes[0].set_title("P2c115 n2 信赖域步长真实重求解闭环")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(names, center, color=colors)
    axes[1].axhline(1.0e-3, color="black", linestyle="--", linewidth=1.0, label="中心线变化参考 1e-3 m")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("中心线最大变化 (m)")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    fig.savefig(OUT_DIR / "n2_trust_region_reopt_closure_cn.png", dpi=220)
    plt.close(fig)


def main() -> None:
    """Run the reoptimization closure audit."""
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

    cases = [row for row in read_csv(P2C114_DIR / "trust_region_rows.csv") if row["action"] in {"完整执行", "缩放执行"}]
    rows: list[dict] = []
    failures: list[dict] = []
    for case in cases:
        dI = np.asarray(json.loads(case["dI_A"]), dtype=float)
        predicted = np.asarray(json.loads(case["suggested_dx_m"]), dtype=float)
        I_new = I0 + dI
        q, solve_info = p2c75._solve_q(y_base=y_base, basis=basis, I=I_new, funcs=funcs)
        y_new = y_base + basis @ q
        out_new = output_arrays(funcs, y_new, I_new)
        actual = out_new["tip_p"] - base_out["tip_p"]
        rel, cosine = norm_rel(predicted, actual)
        centerline_change = float(np.max(np.linalg.norm(out_new["centerline"] - base_out["centerline"], axis=1)))
        boundary_norm = float(np.linalg.norm(out_new["boundary"]))
        closure_pass = bool(solve_info["success"] and rel < 0.05 and centerline_change < 1.0e-3)
        if not closure_pass:
            reasons = []
            if not solve_info["success"]:
                reasons.append("状态重求解失败")
            if rel >= 0.05:
                reasons.append("实际位移与线性预测不一致")
            if centerline_change >= 1.0e-3:
                reasons.append("中心线变化超过局部分支参考门限")
            failures.append({"case": case["case"], "failure_class": "closure_failed", "reasons": reasons})
        rows.append(
            {
                "case": case["case"],
                "action": case["action"],
                "requested_norm_um": float(case["requested_norm_um"]),
                "predicted_step_norm_um": float(case["predicted_step_norm_um"]),
                "actual_step_norm_um": float(np.linalg.norm(actual) * 1.0e6),
                "remaining_norm_um": float(case["remaining_norm_um"]),
                "suggested_current_A": float(case["suggested_current_A"]),
                "actual_rel_err": rel,
                "actual_cosine": cosine,
                "centerline_max_change_m": centerline_change,
                "boundary_norm": boundary_norm,
                "solve_success": bool(solve_info["success"]),
                "solve_nfev": int(solve_info["nfev"]),
                "solve_steady_s": float(solve_info["steady_s"]),
                "closure_status": "pass" if closure_pass else "fail",
                "predicted_dx_m": json.dumps(predicted.tolist()),
                "actual_dx_m": json.dumps(actual.tolist()),
                "dI_A": json.dumps(dI.tolist()),
            }
        )

    pass_count = sum(row["closure_status"] == "pass" for row in rows)
    max_rel = max(row["actual_rel_err"] for row in rows)
    max_centerline = max(row["centerline_max_change_m"] for row in rows)
    summary = {
        "experiment_id": OUT_DIR.name,
        "status": "reopt_closure_pass" if pass_count == len(rows) else "reopt_closure_partial",
        "source_experiment": P2C114_DIR.name,
        "tested_cases": len(rows),
        "pass_count": pass_count,
        "max_actual_rel_err": max_rel,
        "max_centerline_change_m": max_centerline,
        "total_solve_steady_s": sum(row["solve_steady_s"] for row in rows),
        "wall_s": time.perf_counter() - t0,
        "interpretation": (
            "该实验对 P2c114 中可执行或可缩放的建议步长进行真实分支重求解。"
            "若通过，说明该步长在当前可信分支内具有局部闭环一致性；仍不代表全局 IK 可用。"
        ),
    }
    write_csv(OUT_DIR / "closure_rows.csv", rows)
    write_csv(OUT_DIR / "summary.csv", [summary])
    (OUT_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(jsonable(item), ensure_ascii=False) + "\n" for item in failures),
        encoding="utf-8",
    )
    plot_rows(rows)
    if pass_count == len(rows):
        conclusion = (
            "P2c114 的可执行步长在真实分支重求解后仍保持局部一致。"
            "该结论仍限定于同一模型、同一基准电流、同一可信分支和三维末端位置任务。"
        )
        next_plan = (
            "下一步可将该闭环筛选封装为只读候选接口，并在每次执行后重新审计分支漂移。"
        )
    else:
        conclusion = (
            "P2c114 的 5 A 电流步长预算过大，真实重求解后未保持局部一致。"
            "失败主因不是接口契约失效，而是建议步长远超 P2c107/P2c104 已验证的 0.0125 A 扰动尺度。"
            "因此当前不能把 5 A 作为外层 IK 的局部线性信赖域。"
        )
        next_plan = (
            "下一轮应把信赖域半径回退到已审计扰动尺度附近，例如 0.0125 A 或更小，"
            "先验证微小电流步长的真实重求解闭环，再逐步夹逼最大可线性化半径。"
        )

    record = f"""# P2c115 n2 信赖域步长真实重求解闭环验证

## 实验目的

P2c114 建立了外层请求进入局部灵敏度模型前的信赖域筛选规则。本轮进一步验证：对被标记为“完整执行”或“缩放执行”的请求，按照建议电流步长改变电流后，可信分支状态重求解得到的真实末端位移是否与线性预测一致。

## 方法

从 P2c104 可信分支缓存恢复基准状态、有效子空间和残差函数。对 P2c114 中 9 个可执行或可缩放请求，施加建议电流步长 \\(\\Delta I\\)，并在原有效子空间内重求解状态增量。随后比较

\\[
\\Delta p_\\mathrm{{pred}} = S\\Delta I, \\qquad
\\Delta p_\\mathrm{{actual}} = p(I_0 + \\Delta I, y_0 + \\Delta y) - p(I_0, y_0).
\\]

通过门限为：状态重求解成功、实际相对误差小于 0.05、中心线最大变化小于 \\(10^{{-3}}\\,\\mathrm{{m}}\\)。

## 结果

- 测试样本数：{len(rows)}。
- 通过数：{pass_count}。
- 最大实际相对误差：{max_rel:.3e}。
- 最大中心线变化：{max_centerline:.3e} m。
- 状态重求解累计耗时：{summary["total_solve_steady_s"]:.2f} s。
- 墙钟耗时：{summary["wall_s"]:.2f} s。

## 结论

{conclusion}

## 下一步计划

{next_plan}
"""
    (OUT_DIR / "record.md").write_text(record, encoding="utf-8")
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
