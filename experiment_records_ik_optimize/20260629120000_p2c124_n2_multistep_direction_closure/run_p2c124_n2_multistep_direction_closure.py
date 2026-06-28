"""Run multistep closure checks for trusted n2 directional sensitivity."""

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
from n2_trust_radius_sensitivity_interface import N2TrustRadiusSensitivityInterface

import run_p2c75_n2_state_increment_audit as p2c75
import run_p2c104_n2_cached_branch_audit as p2c104


jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

MAX_STEPS = 12
STOP_ERROR_M = 1.0e-9
CENTERLINE_GATE_M = 1.0e-3
TARGETS = {
    "强方向_50nm": ("strong", 50.0e-9),
    "强方向_10nm": ("strong", 10.0e-9),
    "弱方向_5nm": ("weak", 5.0e-9),
    "弱方向_1nm": ("weak", 1.0e-9),
}


def jsonable(value: Any) -> Any:
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV rows."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def output_arrays(funcs: dict[str, Any], y: np.ndarray, I: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate endpoint and centerline outputs."""
    out = funcs["output"](jnp.asarray(y, dtype=jnp.float64), jnp.asarray(I, dtype=jnp.float64).reshape(8))
    return {
        "tip_p": np.asarray(jax.device_get(out.tip_p), dtype=float),
        "centerline": np.asarray(jax.device_get(out.centerline), dtype=float),
        "boundary": np.asarray(jax.device_get(out.boundary), dtype=float),
    }


def minimum_norm_step(S: np.ndarray, residual: np.ndarray, radius_A: float) -> np.ndarray:
    """Compute a trust-radius limited current step."""
    dI = np.linalg.pinv(S, rcond=1.0e-12) @ np.asarray(residual, dtype=float).reshape(3)
    norm = float(np.linalg.norm(dI))
    if norm > radius_A:
        dI = dI * (radius_A / norm)
    return np.asarray(dI, dtype=float)


def build_targets(S: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Build strong and weak task-space target offsets."""
    U, _singular_values, _Vt = np.linalg.svd(S, full_matrices=False)
    targets = []
    for name, (kind, magnitude_m) in TARGETS.items():
        direction = U[:, 0] if kind == "strong" else U[:, -1]
        targets.append((name, direction * magnitude_m))
    return targets


def run_target(
    *,
    target_name: str,
    target_offset: np.ndarray,
    radius_A: float,
    base_request: dict[str, Any],
    interface: N2TrustRadiusSensitivityInterface,
    S: np.ndarray,
    I0: np.ndarray,
    y0: np.ndarray,
    basis: np.ndarray,
    funcs: dict[str, Any],
    base_out: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run one multistep closure target."""
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    target_tip = base_out["tip_p"] + target_offset
    current_I = I0.copy()
    current_out = base_out
    initial_error = float(np.linalg.norm(target_tip - current_out["tip_p"]))
    total_solve_s = 0.0
    for step in range(MAX_STEPS + 1):
        error_vec = target_tip - current_out["tip_p"]
        error_norm = float(np.linalg.norm(error_vec))
        centerline_drift = float(np.max(np.linalg.norm(current_out["centerline"] - base_out["centerline"], axis=1)))
        rows.append(
            {
                "target": target_name,
                "step": step,
                "error_nm": error_norm * 1.0e9,
                "cumulative_current_norm_A": float(np.linalg.norm(current_I - I0)),
                "centerline_drift_m": centerline_drift,
                "boundary_norm": float(np.linalg.norm(current_out["boundary"])),
                "current_step_norm_A": 0.0,
                "solve_nfev": 0,
                "solve_steady_s": 0.0,
                "solve_success": True,
            }
        )
        if error_norm <= STOP_ERROR_M or step == MAX_STEPS:
            break
        dI = minimum_norm_step(S, error_vec, radius_A)
        decision = interface.validate_step_request(
            {
                **base_request,
                "step_type": "current_increment",
                "dI_A": dI.tolist(),
                "uses_physical_budget_as_linear_radius": False,
            }
        )
        if not decision.accepted:
            failures.append(
                {
                    "target": target_name,
                    "step": step,
                    "failure_class": "interface_rejected_step",
                    "reasons": decision.reasons,
                }
            )
            break
        next_I = current_I + dI
        q, info = p2c75._solve_q(y_base=y0, basis=basis, I=next_I, funcs=funcs)
        next_out = output_arrays(funcs, y0 + basis @ q, next_I)
        next_drift = float(np.max(np.linalg.norm(next_out["centerline"] - base_out["centerline"], axis=1)))
        rows[-1]["current_step_norm_A"] = float(np.linalg.norm(dI))
        rows[-1]["solve_nfev"] = int(info["nfev"])
        rows[-1]["solve_steady_s"] = float(info["steady_s"])
        rows[-1]["solve_success"] = bool(info["success"])
        total_solve_s += float(info["steady_s"])
        if (not info["success"]) or next_drift >= CENTERLINE_GATE_M:
            failures.append(
                {
                    "target": target_name,
                    "step": step + 1,
                    "failure_class": "branch_reopt_failed",
                    "solve_success": bool(info["success"]),
                    "centerline_drift_m": next_drift,
                }
            )
            break
        current_I = next_I
        current_out = next_out

    final_error = float(np.linalg.norm(target_tip - current_out["tip_p"]))
    summary = {
        "target": target_name,
        "initial_error_nm": initial_error * 1.0e9,
        "final_error_nm": final_error * 1.0e9,
        "reduction_ratio": 1.0 - final_error / max(initial_error, 1.0e-18),
        "steps_recorded": max(row["step"] for row in rows),
        "cumulative_current_norm_A": float(np.linalg.norm(current_I - I0)),
        "centerline_drift_m": float(np.max(np.linalg.norm(current_out["centerline"] - base_out["centerline"], axis=1))),
        "total_solve_steady_s": total_solve_s,
        "status": "closed" if final_error <= STOP_ERROR_M else "improved" if final_error < initial_error else "failed",
    }
    return rows, failures, summary


def plot_rows(rows: list[dict[str, Any]], target_rows: list[dict[str, Any]]) -> None:
    """Plot multistep error and current histories."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    for target in sorted({row["target"] for row in rows}):
        part = [row for row in rows if row["target"] == target]
        axes[0].plot([row["step"] for row in part], [row["error_nm"] for row in part], marker="o", label=target)
        axes[1].plot([row["step"] for row in part], [row["cumulative_current_norm_A"] for row in part], marker="o", label=target)
    axes[0].axhline(STOP_ERROR_M * 1.0e9, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("末端位置误差（纳米）")
    axes[0].set_title("n2 强弱方向多步闭环")
    axes[1].set_xlabel("迭代步")
    axes[1].set_ylabel("累计电流变化范数（A）")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(OUT_DIR / "n2_multistep_direction_closure_cn.png", dpi=220)
    plt.close(fig)

    labels = [row["target"] for row in target_rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax.bar(labels, [row["reduction_ratio"] for row in target_rows], color="#356f8f", alpha=0.88)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("误差下降比例")
    ax.set_title("多步闭环目标完成度")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUT_DIR / "n2_multistep_completion_cn.png", dpi=220)
    plt.close(fig)


def write_record(summary: dict[str, Any], target_rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    """Write the Chinese experiment record."""
    target_lines = "\n".join(
        f"- {row['target']}：初始 {row['initial_error_nm']:.3g} nm，最终 {row['final_error_nm']:.3g} nm，下降 {row['reduction_ratio']:.3f}，累计电流 {row['cumulative_current_norm_A']:.3g} A，状态 {row['status']}。"
        for row in target_rows
    )
    failure_lines = "\n".join(
        f"- {row.get('target')} 第 {row.get('step')} 步：{row.get('failure_class')}。"
        for row in failures
    ) or "- 无。"
    record = f"""# P2c124 两点磁控强弱方向多步闭环报告

## 实验目的

P2c123 已证明强方向和弱方向在单步小半径内均能通过真实重求解闭环。本轮进一步测试多步局部推进能力：强方向尝试 50 nm 与 10 nm 目标，弱方向尝试 5 nm 与 1 nm 目标，以区分“灵敏度可信”和“工程推进效率可用”。

## 实验方案

固定使用 P2c104 的可信分支、P2c117 的线性信赖域半径和 P2c123 的最小范数电流步长。每步计算当前末端误差 \(e_k=p_{{target}}-p_k\)，并使用：

\\[
\\Delta I_k = S^+ e_k,
\\]

若 \\(\\|\\Delta I_k\\|_2>0.0125\\,\\mathrm{{A}}\\)，则缩放到信赖域半径。每步后进行真实状态重求解，检查中心线分支漂移是否小于 \(10^{{-3}}\\,\\mathrm{{m}}\)。停止条件为末端误差小于 1 nm 或达到 12 步上限。

## 结果

- 最大单步电流半径：{summary['max_linear_current_step_A']:.6g} A。
- 目标数：{summary['total_targets']}。
- 闭合目标数：{summary['closed_count']}。
- 改善目标数：{summary['improved_count']}。
- 失败事件数：{summary['failure_count']}。
- 墙钟耗时：{summary['wall_s']:.2f} s。
- 累计重求解耗时：{summary['total_solve_steady_s']:.2f} s。

{target_lines}

## 失败归因

{failure_lines}

## 结论

若强方向目标能够在少量步内闭合，而弱方向只能缓慢下降，则当前两点磁控灵敏度应表述为“局部导数可信，但工作空间方向性限制显著”。这类灵敏度可用于诊断与小范围修正，但不能直接替代全局逆运动学规划。

## 下一步计划

下一步应将强弱方向的最大可接受目标尺度写入接口策略：强方向可开放小目标闭环，弱方向默认只给诊断建议。若需要工程级位移，应更换工作点、增加驱动冗余或引入上层路径规划，而不是放大单步线性外推。
"""
    (OUT_DIR / "record.md").write_text(record, encoding="utf-8")


def main() -> None:
    """Run multistep directional closure checks."""
    t0 = time.perf_counter()
    interface = N2TrustRadiusSensitivityInterface(ROOT)
    radius_A = interface.max_linear_current_step_A()
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

    all_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    for target_name, target_offset in build_targets(S):
        rows, target_failures, target_summary = run_target(
            target_name=target_name,
            target_offset=target_offset,
            radius_A=radius_A,
            base_request=base_request,
            interface=interface,
            S=S,
            I0=I0,
            y0=y0,
            basis=basis,
            funcs=funcs,
            base_out=base_out,
        )
        all_rows.extend(rows)
        failures.extend(target_failures)
        target_rows.append(target_summary)

    closed_count = sum(row["status"] == "closed" for row in target_rows)
    improved_count = sum(row["status"] in {"closed", "improved"} for row in target_rows)
    summary = {
        "experiment_id": OUT_DIR.name,
        "status": "multistep_direction_closure_done",
        "max_linear_current_step_A": radius_A,
        "stop_error_nm": STOP_ERROR_M * 1.0e9,
        "max_steps": MAX_STEPS,
        "total_targets": len(target_rows),
        "closed_count": closed_count,
        "improved_count": improved_count,
        "failure_count": len(failures),
        "target_summaries": target_rows,
        "total_solve_steady_s": sum(row["total_solve_steady_s"] for row in target_rows),
        "wall_s": time.perf_counter() - t0,
        "interpretation": "该实验评估可信局部灵敏度在多步小范围推进中的工程可用性，不声明全局 IK 成功。",
    }
    write_csv(OUT_DIR / "iteration_rows.csv", all_rows)
    write_csv(OUT_DIR / "target_summary.csv", target_rows)
    write_csv(
        OUT_DIR / "summary.csv",
        [
            {
                "status": summary["status"],
                "closed_count": closed_count,
                "improved_count": improved_count,
                "failure_count": len(failures),
                "wall_s": summary["wall_s"],
            }
        ],
    )
    (OUT_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(jsonable(row), ensure_ascii=False) + "\n" for row in failures),
        encoding="utf-8",
    )
    plot_rows(all_rows, target_rows)
    write_record(summary, target_rows, failures)
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
