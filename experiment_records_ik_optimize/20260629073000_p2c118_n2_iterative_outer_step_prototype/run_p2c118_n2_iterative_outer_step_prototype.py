"""Prototype iterative outer steps using the n2 trust-radius interface."""

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
P2C117_DIR = ROOT / "experiment_records_ik_optimize/20260629070000_p2c117_n2_trust_radius_interface_contract"
for path in (ROOT, P2C75_DIR, P2C104_DIR, P2C117_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c75_n2_state_increment_audit as p2c75
import run_p2c104_n2_cached_branch_audit as p2c104
from n2_trust_radius_sensitivity_interface import N2TrustRadiusSensitivityInterface


jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

MAX_STEPS = 8
STOP_ERROR_M = 1.0e-8


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


def minimum_norm_current_step(S: np.ndarray, residual: np.ndarray, radius: float) -> np.ndarray:
    """Compute a trust-radius limited minimum-norm current step."""
    dI = np.linalg.pinv(S, rcond=1.0e-12) @ residual
    norm = float(np.linalg.norm(dI))
    if norm > radius:
        dI = dI * (radius / norm)
    return np.asarray(dI, dtype=float)


def build_targets(S: np.ndarray, radius: float) -> list[tuple[str, np.ndarray]]:
    """Build local target offsets with different controllability."""
    U, singular_values, _vt = np.linalg.svd(S, full_matrices=False)
    return [
        ("mode1_three_steps", U[:, 0] * singular_values[0] * radius * 3.0),
        ("mode2_three_steps", U[:, 1] * singular_values[1] * radius * 3.0),
        ("cart_x_small", np.array([singular_values[0] * radius * 2.0, 0.0, 0.0])),
        ("cart_y_weak", np.array([0.0, singular_values[0] * radius * 2.0, 0.0])),
    ]


def plot_rows(rows: list[dict]) -> None:
    """Plot iterative error reduction."""
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    for target in sorted({row["target"] for row in rows}):
        part = [row for row in rows if row["target"] == target]
        ax.plot([row["step"] for row in part], [row["error_norm_um"] for row in part], marker="o", label=target)
    ax.set_yscale("log")
    ax.set_xlabel("迭代步")
    ax.set_ylabel("末端位置误差范数 (微米)")
    ax.set_title("n2 小步迭代外层 IK 原型")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(OUT_DIR / "n2_iterative_outer_step_prototype_cn.png", dpi=220)
    plt.close(fig)


def main() -> None:
    """Run the iterative outer-step prototype."""
    t0 = time.perf_counter()
    interface = N2TrustRadiusSensitivityInterface(ROOT)
    radius = interface.max_linear_current_step_A()
    base_request = interface.as_dict()["required_request"]
    params, I0 = p2c104.p2c79.p2c57.p2c24._params_from_saved_case(
        p2c104.p2c79.p2c57.CASE_NAME,
        p2c104.p2c79.p2c57.ALPHA,
    )
    branch = p2c104._load_cached_branch(params)
    y0 = np.asarray(branch["y"], dtype=float)
    basis = np.asarray(branch["basis"], dtype=float)
    S = np.asarray(branch["S"], dtype=float)
    funcs = branch["funcs"]
    I0 = np.asarray(I0, dtype=float)
    base_out = output_arrays(funcs, y0, I0)
    base_tip = base_out["tip_p"]
    base_centerline = base_out["centerline"]

    rows: list[dict] = []
    failures: list[dict] = []
    target_summaries: list[dict] = []
    for target_name, target_offset in build_targets(S, radius):
        target_tip = base_tip + target_offset
        current_I = I0.copy()
        current_y = y0.copy()
        current_out = base_out
        initial_error = float(np.linalg.norm(target_tip - current_out["tip_p"]))
        for step in range(MAX_STEPS + 1):
            error_vec = target_tip - current_out["tip_p"]
            error_norm = float(np.linalg.norm(error_vec))
            rows.append(
                {
                    "target": target_name,
                    "step": step,
                    "error_norm_um": error_norm * 1.0e6,
                    "current_step_norm_A": 0.0,
                    "cumulative_current_norm_A": float(np.linalg.norm(current_I - I0)),
                    "centerline_drift_m": float(np.max(np.linalg.norm(current_out["centerline"] - base_centerline, axis=1))),
                    "solve_success": True,
                    "solve_nfev": 0,
                    "linear_step_accepted": True,
                }
            )
            if step == MAX_STEPS or error_norm <= STOP_ERROR_M:
                break
            dI = minimum_norm_current_step(S, error_vec, radius)
            request = {
                **base_request,
                "step_type": "current_increment",
                "dI_A": dI.tolist(),
                "uses_physical_budget_as_linear_radius": False,
            }
            decision = interface.validate_step_request(request)
            if not decision.accepted:
                failures.append({"target": target_name, "step": step, "failure_class": "interface_rejected_step", "reasons": decision.reasons})
                break
            next_I = current_I + dI
            q, info = p2c75._solve_q(y_base=y0, basis=basis, I=next_I, funcs=funcs)
            next_y = y0 + basis @ q
            next_out = output_arrays(funcs, next_y, next_I)
            centerline_drift = float(np.max(np.linalg.norm(next_out["centerline"] - base_centerline, axis=1)))
            if not info["success"] or centerline_drift >= 1.0e-3:
                failures.append(
                    {
                        "target": target_name,
                        "step": step + 1,
                        "failure_class": "branch_reopt_failed",
                        "solve_success": bool(info["success"]),
                        "centerline_drift_m": centerline_drift,
                    }
                )
                break
            current_I = next_I
            current_y = next_y
            current_out = next_out
            rows[-1]["current_step_norm_A"] = float(np.linalg.norm(dI))
            rows[-1]["solve_success"] = bool(info["success"])
            rows[-1]["solve_nfev"] = int(info["nfev"])
        final_error = float(np.linalg.norm(target_tip - current_out["tip_p"]))
        target_summaries.append(
            {
                "target": target_name,
                "initial_error_um": initial_error * 1.0e6,
                "final_error_um": final_error * 1.0e6,
                "reduction_ratio": 1.0 - final_error / max(initial_error, 1.0e-18),
                "steps_recorded": max(row["step"] for row in rows if row["target"] == target_name),
                "status": "improved" if final_error < initial_error else "not_improved",
            }
        )

    improved = sum(row["status"] == "improved" for row in target_summaries)
    summary = {
        "experiment_id": OUT_DIR.name,
        "status": "iterative_probe_completed",
        "max_linear_current_step_A": radius,
        "targets": target_summaries,
        "improved_count": improved,
        "total_targets": len(target_summaries),
        "failure_count": len(failures),
        "wall_s": time.perf_counter() - t0,
        "interpretation": "该原型只验证小步迭代外层 IK 的局部趋势，不声明全局 IK 可用。",
    }
    write_csv(OUT_DIR / "iteration_rows.csv", rows)
    write_csv(OUT_DIR / "target_summary.csv", target_summaries)
    write_csv(OUT_DIR / "summary.csv", [{"improved_count": improved, "total_targets": len(target_summaries), "failure_count": len(failures), "wall_s": summary["wall_s"]}])
    (OUT_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(jsonable(row), ensure_ascii=False) + "\n" for row in failures),
        encoding="utf-8",
    )
    plot_rows(rows)
    table = "\n".join(
        f"- {row['target']}：初始误差 {row['initial_error_um']:.3e} 微米，最终误差 {row['final_error_um']:.3e} 微米，下降比例 {row['reduction_ratio']:.3f}。"
        for row in target_summaries
    )
    record = f"""# P2c118 n2 小步迭代外层 IK 原型

## 实验目的

P2c117 已将 n2 分支条件三维位置灵敏度的单步线性信赖域限定为 `0.0125 A`。本轮验证一个最小外层迭代原型：每步只请求半径内电流增量，执行后重求解可信分支状态，并观察末端位置误差是否沿目标方向下降。

## 方法

固定使用 P2c104 的可信分支和 P2c117 的接口契约。对每个小目标位移，采用最小范数电流步长：

\\[
\\Delta I = S^+ (p_\\mathrm{{target}} - p_k),
\\]

若 \\(\\|\\Delta I\\| > 0.0125\\,\\mathrm{{A}}\\)，则缩放到信赖域半径。每步后在同一有效子空间内重求解状态，并检查中心线漂移是否小于 \\(10^{{-3}}\\,\\mathrm{{m}}\\)。

## 结果

- 最大线性电流步长：`{radius:.6g} A`。
- 目标数：{len(target_summaries)}。
- 误差下降目标数：{improved}。
- 失败事件数：{len(failures)}。
- 墙钟耗时：{summary["wall_s"]:.2f} s。

{table}

## 结论

该实验是外层 IK 的局部原型验证，不是全局求解器。若强方向误差能稳定下降，而弱方向或笛卡尔混合方向下降很慢，说明当前 n2 灵敏度适合作为小步局部校正器，但外层任务规划必须考虑可控方向和步长上限。

## 下一步计划

下一步应把该原型整理为“诊断模式外层 IK”：默认只接受小目标位移，逐步执行并在每步后复查分支漂移；对弱方向目标应输出不可快速收敛或需重新规划的工程提示。
"""
    (OUT_DIR / "record.md").write_text(record, encoding="utf-8")
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
