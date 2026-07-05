from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
P2C165_DIR = ROOT / "experiment_records_ik_optimize/20260630193000_p2c165_n2_torque_weight_continuation"
for path in (ROOT, P2C165_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk
from inverse_kinematics import ik_terminal_shooting_jax as shooting

import run_p2c165_n2_torque_weight_continuation as p2c165


REFERENCE_VARIANTS = (
    ("proximal_native", "近端原口径", 0.0),
    ("center_native", "磁体中心口径", 0.5),
    ("distal_native", "远端口径", 1.0),
    ("center_inverse", "中心反向力臂", -0.5),
    ("distal_inverse", "远端反向力臂", -1.0),
)


def jsonable(value: Any) -> Any:
    """Convert arrays and scalars to JSON-compatible values."""
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
    """Write dictionaries to CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def norm(value: Any) -> float:
    """Return a host-side Euclidean norm."""
    return float(np.linalg.norm(np.asarray(jax.device_get(value), dtype=float)))


def current_aware_params(branch: dict[str, Any], current_i: np.ndarray) -> Any:
    """Return branch parameters with updated coil currents."""
    p2c47 = p2c165.p2c162.p2c148.p2c134.p2c104.p2c79.p2c57.p2c47
    return p2c47.with_coil_currents(branch["params"], jnp.asarray(current_i, dtype=jnp.float64).reshape(8))


def rigid_wrench(params: Any, x_prox: jnp.ndarray, index: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return total rigid external wrench in world units."""
    rigid = params.rigid[index]
    f_total, tau_total = shooting.compute_external_wrench_total_rigid(
        x_proximal=x_prox,
        rigid_length=float(rigid.length),
        gravity=params.gravity_rigid_list[index],
        magnetic_model=params.magnetic_model,
        magnet_params=params.magnet_params_list[index],
        coil_currents=params.coil_currents,
    )
    f_total = f_total + jnp.asarray(params.f_ext_list[index], dtype=jnp.float64).reshape(3)
    tau_total = tau_total + jnp.asarray(params.tau_ext_list[index], dtype=jnp.float64).reshape(3)
    return f_total, tau_total


def propagate_with_tau(params: Any, x_prox: jnp.ndarray, index: int, f_total: jnp.ndarray, tau_total: jnp.ndarray) -> jnp.ndarray:
    """Propagate one rigid segment with a specified external wrench."""
    return shooting.normalize_state(
        shooting.rigid_state_along_dim(
            x_prox_dim=x_prox,
            sigma=float(params.rigid[index].length),
            rigid=params.rigid[index],
            f_ext_total_dim=f_total,
            tau_ext_total_dim=tau_total,
        )
    )


def trace_reference_rows(params: Any, y: np.ndarray, *, steps_per_flex: int) -> list[dict[str, Any]]:
    """Trace interface residuals under alternative rigid-moment reference assumptions."""
    blocks = jnp.asarray(y, dtype=jnp.float64).reshape((len(params.flex), 6))
    x = shooting.initial_state(params, blocks[0])
    rows: list[dict[str, Any]] = []
    f_ref = max(float(params.scales.F_ref), 1.0e-12)
    m_ref = max(float(params.scales.M_ref), 1.0e-12)

    for i in range(len(params.flex)):
        x = x.at[7:10].set(blocks[i, 0:3] * float(params.scales.F_ref))
        x = x.at[10:13].set(blocks[i, 3:6] * float(params.scales.M_ref))
        x = shooting.normalize_state(x)
        x_flex_dist, _ = shooting._integrate_flex(params, x, i, steps_per_flex)
        x_prox = shooting.normalize_state(x_flex_dist)
        f_total, tau_total = rigid_wrench(params, x_prox, i)
        axis_world = shooting.quat_to_rotmat(x_prox[3:7]) @ params.rigid[i].v_star
        rigid_length = float(params.rigid[i].length)

        if i + 1 >= len(params.flex):
            x = propagate_with_tau(params, x_prox, i, f_total, tau_total)
            continue

        next_force = blocks[i + 1, 0:3] * f_ref
        next_torque = blocks[i + 1, 3:6] * m_ref

        for variant_name, variant_label, lever_fraction in REFERENCE_VARIANTS:
            lever = lever_fraction * rigid_length * axis_world
            tau_variant = tau_total + jnp.cross(lever, f_total)
            x_dist = propagate_with_tau(params, x_prox, i, f_total, tau_variant)
            force_jump = next_force - x_dist[7:10]
            torque_jump = next_torque - x_dist[10:13]
            rows.append(
                {
                    "interface_index": i,
                    "variant": variant_name,
                    "variant_label": variant_label,
                    "lever_fraction": float(lever_fraction),
                    "force_jump_norm_dim": norm(force_jump),
                    "torque_jump_norm_dim": norm(torque_jump),
                    "block_jump_norm_dim": norm(jnp.concatenate([force_jump / f_ref, torque_jump / m_ref])),
                    "normalized_force_jump": norm(force_jump / f_ref),
                    "normalized_torque_jump": norm(torque_jump / m_ref),
                    "rigid_force_norm_dim": norm(f_total),
                    "rigid_torque_norm_dim": norm(tau_total),
                    "lever_moment_norm_dim": norm(jnp.cross(lever, f_total)),
                    "next_force_norm_dim": norm(next_force),
                    "next_torque_norm_dim": norm(next_torque),
                    "dist_force_norm_dim": norm(x_dist[7:10]),
                    "dist_torque_norm_dim": norm(x_dist[10:13]),
                }
            )
        x = propagate_with_tau(params, x_prox, i, f_total, tau_total)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize reference-point audit rows."""
    by_variant: dict[str, dict[str, Any]] = {}
    for name, label, _fraction in REFERENCE_VARIANTS:
        selected = [row for row in rows if row["variant"] == name]
        torque_norm = float(np.linalg.norm([row["normalized_torque_jump"] for row in selected]))
        force_norm = float(np.linalg.norm([row["normalized_force_jump"] for row in selected]))
        block_norm = float(np.linalg.norm([row["block_jump_norm_dim"] for row in selected]))
        by_variant[name] = {
            "variant": name,
            "variant_label": label,
            "normalized_force_jump_total": force_norm,
            "normalized_torque_jump_total": torque_norm,
            "normalized_block_jump_total": block_norm,
            "max_lever_moment_norm_dim": max((row["lever_moment_norm_dim"] for row in selected), default=0.0),
        }
    baseline = by_variant["proximal_native"]["normalized_torque_jump_total"]
    best_name = min(by_variant, key=lambda name: by_variant[name]["normalized_torque_jump_total"])
    best = by_variant[best_name]
    reduction = baseline / max(float(best["normalized_torque_jump_total"]), 1.0e-18)
    if reduction >= 10.0:
        decision = "存在强参考点失配嫌疑；下一步应在该参考点上重做物理闭合残差。"
        status = "moment_reference_mismatch_suspected"
    elif reduction >= 2.0:
        decision = "参考点影响存在但不足以解释全部失败；需同时审计刚段传播和任务输出。"
        status = "moment_reference_partial_effect"
    else:
        decision = "参考点切换不能显著降低矩残差；应转向刚段载荷传播方程或磁外载模型审计。"
        status = "moment_reference_not_primary"
    return {
        "status": status,
        "baseline_variant": "proximal_native",
        "best_variant": best_name,
        "baseline_normalized_torque_jump_total": baseline,
        "best_normalized_torque_jump_total": best["normalized_torque_jump_total"],
        "torque_jump_reduction": float(reduction),
        "decision": decision,
        "variant_summary": list(by_variant.values()),
    }


def plot(summary: dict[str, Any]) -> None:
    """Plot torque and force jump totals for each reference variant."""
    configure_matplotlib_cjk()
    rows = summary["variant_summary"]
    labels = [row["variant_label"] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.5), sharex=True)
    axes[0].bar(x, [row["normalized_torque_jump_total"] for row in rows], color="#d95f02")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("归一化矩跳变")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, [row["normalized_force_jump_total"] for row in rows], color="#1b9e77")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("归一化力跳变")
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("n2 刚段矩参考点只读审计")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "n2_moment_reference_audit_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any], rows: list[dict[str, Any]], wall_s: float) -> None:
    """Write Chinese experiment record."""
    table = [
        "| 参考点假设 | 归一化矩跳变 | 归一化力跳变 | 力臂矩最大值 |",
        "|---|---:|---:|---:|",
    ]
    for row in summary["variant_summary"]:
        table.append(
            f"| {row['variant_label']} | {row['normalized_torque_jump_total']:.3e} | "
            f"{row['normalized_force_jump_total']:.3e} | {row['max_lever_moment_norm_dim']:.3e} |"
        )
    text = f"""# P2c167 两点磁控 n2 矩参考点与刚段力臂只读审计

## 实验目的

P2c165 表明，继续增加接口矩连续权重不能得到可发布灵敏度。本轮在同一工作点、同一路径、同一第二步候选上，只读比较不同刚段外矩参考点假设，判断线性化失败是否可能由力矩参考点或力臂矩定义不一致导致。

## 实验方案

复用 `rand_05` 两点磁控工作点。第一步沿局部第一主方向；第二步复用 P2c165 的零额外矩权重候选。对刚段外矩传播比较五种假设：近端原口径、磁体中心口径、远端口径、中心反向力臂、远端反向力臂。每种假设均计算接口处归一化力跳变和归一化矩跳变。

## 结果

{chr(10).join(table)}

## 数据分析

原口径归一化矩跳变为 {summary['baseline_normalized_torque_jump_total']:.3e}。最优候选为 `{summary['best_variant']}`，归一化矩跳变为 {summary['best_normalized_torque_jump_total']:.3e}，下降倍数为 {summary['torque_jump_reduction']:.3f}。

若参考点切换能带来超过 10 倍下降，可判定存在强参考点失配嫌疑；若下降不足 2 倍，则说明当前失败更可能来自刚段传播方程、磁外载模型或任务输出耦合，而不是单一参考点选择。

## 实验结论

{summary['decision']}

## 下一步计划

若本轮没有发现强参考点失配，应停止继续扫矩权重，转向刚段载荷传播方程和磁外载模型一致性审计。若发现强参考点失配，则在该参考点上实现最小物理闭合残差，并复测 P2c162/P2c165 的第二步线性化门禁。

## 耗时

- 墙钟耗时：{wall_s:.2f} s。
- 是否修改生产 IK：否。
"""
    (EXP_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run the moment-reference audit."""
    t0 = time.perf_counter()
    base = p2c165.p2c162.p2c148.p2c145.p2c144.p2c89._recover_trusted()
    direction_name, direction = next(iter(p2c165.p2c162.p2c148.p2c145.p2c144.candidate_directions()))
    branch0 = p2c165.p2c162.p2c148.build_candidate_branch(base, direction_name, direction)
    first_row, branch1 = p2c165.p2c162.run_step(branch0, step_index=1, mode_index=0)
    q, solve_info, current_new = p2c165.solve_with_torque_weight(branch1, p2c165.STEP_RADIUS_A, 0.0)
    y_new = np.asarray(branch1["y"] + branch1["basis"] @ q, dtype=float)
    params_i = current_aware_params(branch1, current_new)
    args = p2c165.p2c162.p2c148.p2c134.p2c104.p2c79.p2c57.Args
    rows = trace_reference_rows(params_i, y_new, steps_per_flex=args.steps_per_flex)
    summary = summarize(rows)
    summary.update(
        {
            "experiment_id": EXP_DIR.name,
            "case": direction_name,
            "path": "first_principal_then_zero_torque_weight_second_step",
            "outer_ik_allowed": False,
            "publishable_sensitivity": False,
            "first_step": first_row,
            "second_step_solve": solve_info,
            "wall_s": float(time.perf_counter() - t0),
            "jax_devices": [str(device) for device in jax.devices()],
        }
    )
    write_csv(EXP_DIR / "moment_reference_rows.csv", rows)
    (EXP_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (EXP_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "failure_class": "模型耦合缺口" if summary["status"] != "moment_reference_mismatch_suspected" else "矩参考点不一致嫌疑",
                    "evidence": {
                        "status": summary["status"],
                        "best_variant": summary["best_variant"],
                        "torque_jump_reduction": summary["torque_jump_reduction"],
                        "decision": summary["decision"],
                    },
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    plot(summary)
    write_record(summary, rows, summary["wall_s"])
    commands = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/20260704223000_p2c167_n2_moment_reference_audit/run_p2c167_n2_moment_reference_audit.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
