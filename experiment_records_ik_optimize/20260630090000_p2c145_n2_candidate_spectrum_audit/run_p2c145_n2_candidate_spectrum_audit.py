"""Audit task-space sensitivity spectra for selected n2 candidates."""

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
P2C79_DIR = ROOT / "experiment_records_ik_optimize/20260628020500_p2c79_branch_random_validation"
P2C144_DIR = ROOT / "experiment_records_ik_optimize/20260630083000_p2c144_n2_candidate_baseline_gate"
for path in (ROOT, P2C79_DIR, P2C144_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import matplotlib.pyplot as plt
import numpy as np

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c79_branch_random_validation as p2c79
import run_p2c144_n2_candidate_baseline_gate as p2c144


configure_matplotlib_cjk()

TASK_CONDITION_GATE = 1.0e7
FORCE_RATIO_GATE = 1.5
STATIONARITY_GATE = 1.0e-5
CONTINUITY_GATE = 1.0e-3
MIN_SIGMA_GATE = 1.0e-10


def jsonable(value: Any) -> Any:
    """Convert arrays and scalars to JSON-compatible objects."""
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
    """Write rows to CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def classify(row: dict[str, Any]) -> tuple[str, str]:
    """Classify publishability from baseline and spectrum gates."""
    if not row["baseline_ok"]:
        return "不可发布", "基线物理门禁未通过"
    if row["task_condition"] >= TASK_CONDITION_GATE:
        return "谨慎诊断", "任务灵敏度条件数处于病态区"
    if row["effective_rank"] < 3:
        return "有效子空间候选", "完整三维任务秩不足"
    if row["sigma_3"] < MIN_SIGMA_GATE:
        return "有效子空间候选", "最弱方向信号低于可靠阈值"
    return "三维位置候选", "通过谱门禁，仍需扰动重求解"


def recover_candidate_rows() -> tuple[list[dict[str, Any]], float]:
    """Recover candidate branches and compute task sensitivity spectra."""
    base = p2c144.p2c89._recover_trusted()
    params = base["params"]
    I0 = base["current_i"]
    target0 = base["target_p"]
    y_ref = base["y_ref"]
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for name, direction in p2c144.candidate_directions():
        y = np.asarray(base["y"], dtype=float).copy()
        final = None
        target = None
        for fraction in (0.5, 1.0):
            target = target0 + fraction * p2c144.TARGET_STEP_M * direction
            out = p2c144.p2c89._solve_target_step(
                params=params,
                current_i=I0,
                target_p=target,
                y_ref=y_ref,
                y0=y,
            )
            y = np.asarray(out["solve"]["y"], dtype=float)
            final = dict(out["row"])
        assert final is not None and target is not None
        _args, funcs = p2c79.p2c69._make_funcs(params, target, y_ref, p2c79.p2c69.STAGES[-1][0])
        J = np.asarray(
            jax.device_get(
                funcs["jacobian"](
                    p2c79.jnp.asarray(y, dtype=p2c79.jnp.float64),
                    p2c79.jnp.asarray(I0, dtype=p2c79.jnp.float64).reshape(8),
                )
            ),
            dtype=float,
        )
        _u, _s, vt = np.linalg.svd(J, full_matrices=False)
        basis = p2c79.p2c72.p2c71._orthogonal_complement(vt[-p2c79.p2c72.REMOVE_WEAK_DIRECTIONS :], J.shape[1])
        system = p2c79.p2c77._build_optimality_system(y_base=y, I0=I0, funcs=funcs, basis=basis)
        S = p2c79.p2c78._task_sensitivity(system)
        singular = np.linalg.svd(S, compute_uv=False)
        condition = float(singular[0] / max(float(singular[-1]), np.finfo(float).tiny))
        out = p2c79.p2c57.p2c47.output_to_numpy(
            funcs["output"](
                p2c79.jnp.asarray(y, dtype=p2c79.jnp.float64),
                p2c79.jnp.asarray(I0, dtype=p2c79.jnp.float64).reshape(8),
            )
        )
        baseline_ok = bool(
            final["success"]
            and final["stationarity_norm"] <= STATIONARITY_GATE
            and final["continuity_norm"] <= CONTINUITY_GATE
            and final["max_interface_force_jump_over_rigid_force"] <= FORCE_RATIO_GATE
        )
        row = {
            "direction": name,
            "baseline_ok": baseline_ok,
            "target_offset_m": p2c144.TARGET_STEP_M,
            "stationarity_norm": final["stationarity_norm"],
            "continuity_norm": final["continuity_norm"],
            "force_ratio": final["max_interface_force_jump_over_rigid_force"],
            "position_error_m": final["position_error_m"],
            "task_condition": condition,
            "sigma_1": float(singular[0]),
            "sigma_2": float(singular[1]),
            "sigma_3": float(singular[2]),
            "sigma_2_over_sigma_1": float(singular[1] / max(singular[0], np.finfo(float).tiny)),
            "sigma_3_over_sigma_1": float(singular[2] / max(singular[0], np.finfo(float).tiny)),
            "effective_rank": int(np.sum(singular >= MIN_SIGMA_GATE)),
            "tip_x": float(out["tip_p"][0]),
            "tip_y": float(out["tip_p"][1]),
            "tip_z": float(out["tip_p"][2]),
        }
        release_class, reason = classify(row)
        row["release_class"] = release_class
        row["reason"] = reason
        rows.append(row)
        write_csv(OUT_DIR / "spectrum_rows.partial.csv", rows)
    return rows, float(time.perf_counter() - t0)


def plot(rows: list[dict[str, Any]]) -> None:
    """Plot candidate singular spectra."""
    labels = [row["direction"] for row in rows]
    x = np.arange(len(labels))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(x - width, [row["sigma_1"] for row in rows], width, label="第一奇异值")
    axes[0].bar(x, [row["sigma_2"] for row in rows], width, label="第二奇异值")
    axes[0].bar(x + width, [row["sigma_3"] for row in rows], width, label="第三奇异值")
    axes[0].axhline(MIN_SIGMA_GATE, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].set_title("末端位置灵敏度奇异值")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].legend()

    axes[1].bar(labels, [row["task_condition"] for row in rows], color="#457b9d")
    axes[1].axhline(TASK_CONDITION_GATE, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_yscale("log")
    axes[1].set_title("任务灵敏度条件数")
    axes[1].tick_params(axis="x", rotation=25)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("P2c145 n2 候选工作点灵敏度谱审计")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_candidate_spectrum_audit_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write the Chinese experiment record."""
    table = [
        "| 方向 | 基线门禁 | 条件数 | 有效秩 | 第一奇异值 | 第二奇异值 | 第三奇异值 | 结论 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        table.append(
            f"| {row['direction']} | {row['baseline_ok']} | {row['task_condition']:.3e} | "
            f"{row['effective_rank']} | {row['sigma_1']:.3e} | {row['sigma_2']:.3e} | "
            f"{row['sigma_3']:.3e} | {row['release_class']} |"
        )
    text = f"""# P2c145 两点磁控候选工作点灵敏度谱审计

## 实验目的

P2c144 已证明 5 个 n2 候选工作点均能通过真实基线闭环。本轮进一步计算每个候选的末端位置灵敏度矩阵奇异值谱，用于判断其是否具备完整三维末端位置灵敏度发布资格。

## 方法

对每个候选目标点重新恢复物理解分支，构造局部末端位置灵敏度矩阵 \(S = \\partial p_{{tip}}/\\partial I\)，并计算三个奇异值。若基线物理门禁不通过，则禁止后续扰动扫描；若条件数高于 `1e7` 或有效秩不足，则不得发布完整三维灵敏度。

## 结果

{chr(10).join(table)}

## 结论

- 基线通过候选数：{summary['baseline_pass_count']} / {summary['candidate_count']}。
- 三维位置候选数：{summary['full_3d_candidate_count']}。
- 有效子空间候选数：{summary['effective_subspace_candidate_count']}。
- 谨慎诊断候选数：{summary['diagnostic_count']}。

本轮只完成谱门禁，尚未完成扰动重求解一致性审计。即使某些候选满足谱门禁，也只能进入下一轮第一主方向和弱方向扰动重求解，不能直接接入外层 IK。

## 下一步计划

1. 对谱门禁最好的候选做第一主方向扰动重求解。
2. 若候选存在三维谱门禁通过样本，再补第二、第三主方向验证。
3. 若所有候选仍高度各向异性，则输出两点磁控工作空间分区建议。

## 耗时

- 墙钟耗时：{summary['wall_s']:.2f} s。
"""
    (OUT_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run the spectrum audit."""
    rows, wall_s = recover_candidate_rows()
    summary = {
        "experiment_id": OUT_DIR.name,
        "purpose": "audit_n2_candidate_task_sensitivity_spectra",
        "candidate_count": len(rows),
        "baseline_pass_count": sum(row["baseline_ok"] for row in rows),
        "full_3d_candidate_count": sum(row["release_class"] == "三维位置候选" for row in rows),
        "effective_subspace_candidate_count": sum(row["release_class"] == "有效子空间候选" for row in rows),
        "diagnostic_count": sum(row["release_class"] == "谨慎诊断" for row in rows),
        "publishable_now": False,
        "outer_ik_allowed": False,
        "wall_s": wall_s,
        "jax_devices": [str(device) for device in jax.devices()],
        "rows": rows,
    }
    failure_events = [
        {"direction": row["direction"], "failure_class": row["reason"], "evidence": row}
        for row in rows
        if row["release_class"] != "三维位置候选"
    ]
    write_csv(OUT_DIR / "spectrum_rows.csv", rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    with (OUT_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for event in failure_events:
            f.write(json.dumps(jsonable(event), ensure_ascii=False) + "\n")
    plot(rows)
    write_record(summary, rows)
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
