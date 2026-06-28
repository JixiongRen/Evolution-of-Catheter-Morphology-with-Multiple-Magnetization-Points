"""Audit n2 candidate principal-direction current reoptimization."""

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
P2C123_DIR = ROOT / "experiment_records_ik_optimize/20260629110000_p2c123_n2_direction_reopt_closure"
P2C134_DIR = ROOT / "experiment_records_ik_optimize/20260630023000_p2c134_n2_trusted_branch_basin_radius"
P2C145_DIR = ROOT / "experiment_records_ik_optimize/20260630090000_p2c145_n2_candidate_spectrum_audit"
for path in (ROOT, P2C123_DIR, P2C134_DIR, P2C145_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import matplotlib.pyplot as plt
import numpy as np

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c123_n2_direction_reopt_closure as p2c123
import run_p2c134_n2_trusted_branch_basin_radius as p2c134
import run_p2c145_n2_candidate_spectrum_audit as p2c145


configure_matplotlib_cjk()

CURRENT_RADII_A = (1.0e-4, 3.0e-4, 1.0e-3, 2.0e-3)
LINEAR_REL_GATE = 0.05
CENTERLINE_GATE_M = 1.0e-3
FORCE_RATIO_GATE = 1.5
CONTINUITY_GATE = 1.0e-3
STATE_RADIUS_GATE = 1.0e-3


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
    """Write rows to CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def build_candidate_branch(base: dict[str, Any], name: str, direction: np.ndarray) -> dict[str, Any]:
    """Recover a candidate branch and build its local sensitivity matrix."""
    params = base["params"]
    I0 = base["current_i"]
    target0 = base["target_p"]
    y_ref = base["y_ref"]
    y = np.asarray(base["y"], dtype=float).copy()
    target = target0 + p2c145.p2c144.TARGET_STEP_M * direction
    final = None
    for fraction in (0.5, 1.0):
        step_target = target0 + fraction * p2c145.p2c144.TARGET_STEP_M * direction
        out = p2c145.p2c144.p2c89._solve_target_step(
            params=params,
            current_i=I0,
            target_p=step_target,
            y_ref=y_ref,
            y0=y,
        )
        y = np.asarray(out["solve"]["y"], dtype=float)
        final = dict(out["row"])
    assert final is not None
    _args, funcs = p2c145.p2c79.p2c69._make_funcs(params, target, y_ref, p2c145.p2c79.p2c69.STAGES[-1][0])
    I_j = p2c145.p2c79.jnp.asarray(I0, dtype=p2c145.p2c79.jnp.float64).reshape(8)
    y_j = p2c145.p2c79.jnp.asarray(y, dtype=p2c145.p2c79.jnp.float64)
    J = np.asarray(jax.device_get(funcs["jacobian"](y_j, I_j)), dtype=float)
    _u_j, _s_j, vt = np.linalg.svd(J, full_matrices=False)
    basis = p2c145.p2c79.p2c72.p2c71._orthogonal_complement(
        vt[-p2c145.p2c79.p2c72.REMOVE_WEAK_DIRECTIONS :],
        J.shape[1],
    )
    system = p2c145.p2c79.p2c77._build_optimality_system(y_base=y, I0=I0, funcs=funcs, basis=basis)
    S = p2c145.p2c79.p2c78._task_sensitivity(system)
    U, singular_values, Vt = np.linalg.svd(S, full_matrices=False)
    out = p2c134.eval_output(funcs, y, I0)
    return {
        "name": name,
        "params": params,
        "I0": np.asarray(I0, dtype=float),
        "y": y,
        "basis": basis,
        "funcs": funcs,
        "S": S,
        "U": U,
        "Vt": Vt,
        "singular_values": singular_values,
        "final": final,
        "base_out": out,
    }


def classify(row: dict[str, Any]) -> tuple[str, str]:
    """Classify one perturbation row."""
    if not row["solve_success"]:
        return "不可发布", "扰动重求解未收敛"
    if row["linear_rel_err"] >= LINEAR_REL_GATE:
        return "不可发布", "局部线性化误差超限"
    if row["centerline_max_change_m"] >= CENTERLINE_GATE_M:
        return "不可发布", "中心线分支漂移超限"
    if row["continuity_norm"] >= CONTINUITY_GATE:
        return "不可发布", "接口连续残差超限"
    if row["force_ratio"] >= FORCE_RATIO_GATE:
        return "不可发布", "载荷传播比例超限"
    if row["state_radius"] >= STATE_RADIUS_GATE:
        return "谨慎诊断", "有效状态增量偏大"
    return "通过", "第一主方向扰动重求解通过"


def run_reopt_case(branch: dict[str, Any], radius: float) -> dict[str, Any]:
    """Run one current perturbation reoptimization case."""
    task_direction = np.asarray(branch["U"][:, 0], dtype=float)
    dI = p2c123.current_step_for_task_direction(branch["S"], task_direction, radius)
    pred = branch["S"] @ dI
    I_new = branch["I0"] + dI
    q, info = p2c134.solve_reduced(
        q0=np.zeros(branch["basis"].shape[1], dtype=float),
        y_base=branch["y"],
        basis=branch["basis"],
        I=I_new,
        funcs=branch["funcs"],
    )
    y_new = branch["y"] + branch["basis"] @ q
    out = p2c134.eval_output(branch["funcs"], y_new, I_new)
    actual = out["tip_p"] - branch["base_out"]["tip_p"]
    rel, cosine = p2c123.relative_error(pred, actual)
    centerline_change = float(np.max(np.linalg.norm(out["centerline"] - branch["base_out"]["centerline"], axis=1)))
    continuity, force_ratio, torque_ratio = p2c134.load_metrics(branch["params"], y_new)
    row = {
        "direction": branch["name"],
        "radius_A": radius,
        "singular_value_1": float(branch["singular_values"][0]),
        "singular_value_2": float(branch["singular_values"][1]),
        "singular_value_3": float(branch["singular_values"][2]),
        "condition": float(branch["singular_values"][0] / max(float(branch["singular_values"][-1]), np.finfo(float).tiny)),
        "linear_rel_err": rel,
        "pred_actual_cosine": cosine,
        "predicted_norm_um": float(np.linalg.norm(pred) * 1.0e6),
        "actual_norm_um": float(np.linalg.norm(actual) * 1.0e6),
        "centerline_max_change_m": centerline_change,
        "continuity_norm": continuity,
        "force_ratio": force_ratio,
        "torque_ratio": torque_ratio,
        "state_radius": float(np.linalg.norm(q)),
        "solve_success": bool(info["success"]),
        "solve_nfev": int(info["nfev"]),
        "solve_steady_s": float(info["steady_s"]),
    }
    release_class, reason = classify(row)
    row["release_class"] = release_class
    row["reason"] = reason
    return row


def plot(rows: list[dict[str, Any]]) -> None:
    """Plot reoptimization audit metrics."""
    labels = [f"{row['direction']}\n{row['radius_A']:.3f}A" for row in rows]
    x = np.arange(len(labels))
    colors = ["#2a9d8f" if row["release_class"] == "通过" else "#e76f51" for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    axes[0].bar(x, [row["linear_rel_err"] for row in rows], color=colors)
    axes[0].axhline(LINEAR_REL_GATE, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].set_title("局部线性化误差")
    axes[1].bar(x, [row["centerline_max_change_m"] for row in rows], color=colors)
    axes[1].axhline(CENTERLINE_GATE_M, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_yscale("log")
    axes[1].set_title("中心线分支漂移")
    axes[2].bar(x, [row["force_ratio"] for row in rows], color=colors)
    axes[2].axhline(FORCE_RATIO_GATE, color="black", linestyle="--", linewidth=1.0)
    axes[2].set_title("载荷传播比例")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("P2c147 n2 第一主方向安全半径夹逼")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_principal_radius_bracket_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write the Chinese experiment report."""
    table = [
        "| 候选 | 半径/A | 线性误差 | 实际位移/um | 中心线漂移/m | 载荷比 | 结论 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        table.append(
            f"| {row['direction']} | {row['radius_A']:.3f} | {row['linear_rel_err']:.3e} | "
            f"{row['actual_norm_um']:.3e} | {row['centerline_max_change_m']:.3e} | "
            f"{row['force_ratio']:.3f} | {row['release_class']} |"
        )
    text = f"""# P2c147 两点磁控第一主方向安全半径夹逼审计

## 实验目的

P2c146 发现第一主方向在 0.005 A 和 0.010 A 下均因接口连续残差超限而不可发布。本轮缩小电流扰动半径，对每个候选夹逼可满足硬门禁的安全半径。

## 方法

对每个候选构造局部灵敏度矩阵 \(S\)，取第一左奇异向量作为任务方向。使用最小范数电流方向，并在 `1e-4, 3e-4, 1e-3, 2e-3 A` 四个半径下重求解状态。通过门禁包括：重求解成功、线性化误差 `<0.05`、中心线漂移 `<1e-3 m`、接口连续残差 `<1e-3`、载荷传播比例 `<1.5`。

## 结果

{chr(10).join(table)}

## 结论

- 测试样本数：{summary['tested_count']}。
- 通过样本数：{summary['pass_count']}。
- 两个半径均通过的候选数：{summary['candidate_full_pass_count']}。
- 当前是否可发布完整三维灵敏度：否。原因是本轮只验证第一主方向，尚未验证第二、第三主方向。

若存在共同通过半径，则该半径可作为后续第一主方向扰动重求解的安全半径；若所有半径仍失败，则应优先修复接口连续模型或采用连续残差约束增强的重求解。

## 耗时

- 墙钟耗时：{summary['wall_s']:.2f} s。
- 重求解净耗时：{summary['total_solve_steady_s']:.2f} s。
"""
    (OUT_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run the principal-direction reoptimization audit."""
    base = p2c145.p2c144.p2c89._recover_trusted()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for name, direction in p2c145.p2c144.candidate_directions():
        branch = build_candidate_branch(base, name, direction)
        for radius in CURRENT_RADII_A:
            row = run_reopt_case(branch, radius)
            rows.append(row)
            if row["release_class"] != "通过":
                failures.append({"direction": name, "radius_A": radius, "failure_class": row["reason"], "evidence": row})
            write_csv(OUT_DIR / "radius_bracket_rows.partial.csv", rows)
    candidate_full_pass = 0
    for name in sorted({row["direction"] for row in rows}):
        group = [row for row in rows if row["direction"] == name]
        if len(group) == len(CURRENT_RADII_A) and all(row["release_class"] == "通过" for row in group):
            candidate_full_pass += 1
    summary = {
        "experiment_id": OUT_DIR.name,
        "purpose": "bracket_n2_candidate_first_principal_direction_safe_radius",
        "tested_count": len(rows),
        "pass_count": sum(row["release_class"] == "通过" for row in rows),
        "candidate_count": len({row["direction"] for row in rows}),
        "candidate_full_pass_count": candidate_full_pass,
        "publishable_now": False,
        "outer_ik_allowed": False,
        "wall_s": float(time.perf_counter() - t0),
        "total_solve_steady_s": float(sum(row["solve_steady_s"] for row in rows)),
        "jax_devices": [str(device) for device in jax.devices()],
        "rows": rows,
    }
    write_csv(OUT_DIR / "radius_bracket_rows.csv", rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    with (OUT_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for failure in failures:
            f.write(json.dumps(jsonable(failure), ensure_ascii=False) + "\n")
    plot(rows)
    write_record(summary, rows)
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
