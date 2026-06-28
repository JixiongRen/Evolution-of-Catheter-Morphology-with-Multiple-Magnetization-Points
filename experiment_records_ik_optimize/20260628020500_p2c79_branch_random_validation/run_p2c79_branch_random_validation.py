"""Audit n2 branch reproducibility and random task-space directions."""

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
P2C57_DIR = ROOT / "experiment_records_ik_optimize" / "20260607200500_p2c57_force_continuity_weight_repair"
P2C69_DIR = ROOT / "experiment_records_ik_optimize" / "20260627230500_p2c69_n2_condition_precondition_audit"
P2C72_DIR = ROOT / "experiment_records_ik_optimize" / "20260627234500_p2c72_n2_subspace_perturb_reopt"
P2C75_DIR = ROOT / "experiment_records_ik_optimize" / "20260628002500_p2c75_n2_state_increment_audit"
P2C77_DIR = ROOT / "experiment_records_ik_optimize" / "20260628010500_p2c77_optimality_stabilization_audit"
P2C78_DIR = ROOT / "experiment_records_ik_optimize" / "20260628013500_p2c78_task_space_dimensionality_audit"
for path in (ROOT, P2C57_DIR, P2C69_DIR, P2C72_DIR, P2C75_DIR, P2C77_DIR, P2C78_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c57_force_continuity_weight_repair as p2c57
import run_p2c69_n2_condition_precondition_audit as p2c69
import run_p2c72_n2_subspace_perturb_reopt as p2c72
import run_p2c75_n2_state_increment_audit as p2c75
import run_p2c77_optimality_stabilization_audit as p2c77
import run_p2c78_task_space_dimensionality_audit as p2c78

jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent
EPS = 0.005
RANDOM_DIRS = 6
TASK_GATE = 0.05
BRANCH_TIP_GATE_M = 1.0e-5
BRANCH_SENS_GATE = 0.05


def _jsonable(value: Any) -> Any:
    return p2c72._jsonable(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _safe_rel(pred: np.ndarray, fd: np.ndarray) -> tuple[float, float, float]:
    return p2c75._safe_rel(np.asarray(pred, dtype=float), np.asarray(fd, dtype=float))


def _seed_map(params: Any, y_seed: np.ndarray) -> dict[str, np.ndarray]:
    y_seed = np.asarray(y_seed, dtype=float).reshape(-1)
    return {
        "trusted_seed": y_seed,
        "zero_seed": np.zeros_like(y_seed),
        "half_seed": 0.5 * y_seed,
    }


def _recover_branch(params: Any, I0: np.ndarray, target_p: np.ndarray, y0: np.ndarray, label: str) -> dict[str, Any]:
    rec = p2c69._recover_near_gate(params, I0, target_p, y0)
    y = np.asarray(rec["solve"]["y"], dtype=float)
    y_ref = rec["y_ref"]
    _args, funcs = p2c69._make_funcs(params, target_p, y_ref, p2c69.STAGES[-1][0])
    J = np.asarray(jax.device_get(funcs["jacobian"](jnp.asarray(y, dtype=jnp.float64), jnp.asarray(I0, dtype=jnp.float64).reshape(8))), dtype=float)
    _u, _s, vt = np.linalg.svd(J, full_matrices=False)
    basis = p2c72.p2c71._orthogonal_complement(vt[-p2c72.REMOVE_WEAK_DIRECTIONS :], J.shape[1])
    system = p2c77._build_optimality_system(y_base=y, I0=I0, funcs=funcs, basis=basis)
    S = p2c78._task_sensitivity(system)
    U, singular_values, Vt = np.linalg.svd(S, full_matrices=False)
    final_row = max(rec["rows"], key=lambda row: row["stage"])
    out = p2c57.p2c47.output_to_numpy(funcs["output"](jnp.asarray(y, dtype=jnp.float64), jnp.asarray(I0, dtype=jnp.float64).reshape(8)))
    return {
        "label": label,
        "recovered": rec,
        "final_row": final_row,
        "y": y,
        "y_ref": y_ref,
        "funcs": funcs,
        "basis": basis,
        "system": system,
        "S": S,
        "U": U,
        "singular_values": singular_values,
        "Vt": Vt,
        "tip": np.asarray(out["tip_p"], dtype=float),
        "task_condition": float(singular_values[0] / max(float(singular_values[-1]), np.finfo(float).tiny)),
    }


def _branch_row(branch: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    S = branch["S"]
    S_ref = ref["S"]
    sens_rel, sens_cos, _ = _safe_rel(S.reshape(-1), S_ref.reshape(-1))
    tip_diff = float(np.linalg.norm(branch["tip"] - ref["tip"]))
    y_diff = float(np.linalg.norm(branch["y"] - ref["y"]))
    final = branch["final_row"]
    baseline_ok = bool(
        final["success"]
        and final["stationarity_norm"] <= 1.0e-5
        and final["continuity_norm"] <= 1.0e-3
        and final["max_interface_force_jump_over_rigid_force"] <= 1.5
    )
    branch_ok = baseline_ok and tip_diff <= BRANCH_TIP_GATE_M and sens_rel <= BRANCH_SENS_GATE
    return {
        "branch": branch["label"],
        "baseline_ok": baseline_ok,
        "branch_ok": branch_ok,
        "tip_diff_m": tip_diff,
        "state_diff_norm": y_diff,
        "sensitivity_rel_err": sens_rel,
        "sensitivity_cosine": sens_cos,
        "task_condition": branch["task_condition"],
        "sigma_1": float(branch["singular_values"][0]),
        "sigma_2": float(branch["singular_values"][1]),
        "sigma_3": float(branch["singular_values"][2]),
        "stationarity_norm": final["stationarity_norm"],
        "continuity_norm": final["continuity_norm"],
        "force_ratio": final["max_interface_force_jump_over_rigid_force"],
        "position_error_m": final["position_error_m"],
    }


def _direction_set(branch: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    dirs: list[tuple[str, np.ndarray]] = []
    for i in range(3):
        dirs.append((f"task_mode_{i+1}", np.asarray(branch["Vt"][i], dtype=float)))
    rng = np.random.default_rng(79)
    for i in range(RANDOM_DIRS):
        v = rng.normal(size=8)
        dirs.append((f"random_{i+1}", v / np.linalg.norm(v)))
    return dirs


def _audit_direction(branch: dict[str, Any], I0: np.ndarray, name: str, direction: np.ndarray) -> dict[str, Any]:
    funcs = branch["funcs"]
    basis = branch["basis"]
    y = branch["y"]
    q_plus, plus_info = p2c75._solve_q(y_base=y, basis=basis, I=I0 + EPS * direction, funcs=funcs)
    q_minus, minus_info = p2c75._solve_q(y_base=y, basis=basis, I=I0 - EPS * direction, funcs=funcs)
    y_plus = y + basis @ q_plus
    y_minus = y + basis @ q_minus
    out_plus = funcs["output"](jnp.asarray(y_plus, dtype=jnp.float64), jnp.asarray(I0 + EPS * direction, dtype=jnp.float64))
    out_minus = funcs["output"](jnp.asarray(y_minus, dtype=jnp.float64), jnp.asarray(I0 - EPS * direction, dtype=jnp.float64))
    fd = (np.asarray(jax.device_get(out_plus.tip_p), dtype=float) - np.asarray(jax.device_get(out_minus.tip_p), dtype=float)) / (2.0 * EPS)
    pred = branch["S"] @ direction
    rel, cosine, fd_norm = _safe_rel(pred, fd)
    status = "publishable" if rel < TASK_GATE and plus_info["success"] and minus_info["success"] else "diagnostic_only"
    return {
        "branch": branch["label"],
        "direction": name,
        "eps": EPS,
        "status": status,
        "failure_class": "PASS" if status == "publishable" else "随机方向有限差分不一致",
        "vector_rel_err": rel,
        "vector_cosine": cosine,
        "fd_norm": fd_norm,
        "pred_norm": float(np.linalg.norm(pred)),
        "plus_nfev": plus_info["nfev"],
        "minus_nfev": minus_info["nfev"],
        "plus_success": plus_info["success"],
        "minus_success": minus_info["success"],
    }


def _plot(branch_rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    labels = [row["branch"] for row in branch_rows]
    x = np.arange(len(labels))
    axes[0].bar(x - 0.18, [row["tip_diff_m"] for row in branch_rows], width=0.36, label="末端差异")
    axes[0].bar(x + 0.18, [row["sensitivity_rel_err"] for row in branch_rows], width=0.36, label="灵敏度差异")
    axes[0].axhline(BRANCH_TIP_GATE_M, color="gray", linestyle=":", linewidth=1)
    axes[0].axhline(BRANCH_SENS_GATE, color="black", linestyle="--", linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, labels)
    axes[0].legend()
    dirs = list(dict.fromkeys(row["direction"] for row in audit_rows))
    max_err = [max(row["vector_rel_err"] for row in audit_rows if row["direction"] == d) for d in dirs]
    axes[1].bar(np.arange(len(dirs)), max_err)
    axes[1].axhline(TASK_GATE, color="black", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_xticks(np.arange(len(dirs)), dirs, rotation=35, ha="right")
    axes[1].set_ylabel("最大随机方向误差")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("P2c79 分支复核与随机方向验证")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_branch_random_validation_cn.png", dpi=180)
    plt.close(fig)


def _record(branch_rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]], wall_s: float) -> str:
    branch_table = [
        "| 分支 | 基线门禁 | 分支一致 | 末端差异 | 灵敏度差异 | 任务条件数 | 一阶最优性 | 载荷比 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in branch_rows:
        branch_table.append(
            f"| {row['branch']} | {row['baseline_ok']} | {row['branch_ok']} | {row['tip_diff_m']:.3e} | "
            f"{row['sensitivity_rel_err']:.3e} | {row['task_condition']:.3e} | {row['stationarity_norm']:.3e} | {row['force_ratio']:.3e} |"
        )
    audit_table = [
        "| 分支 | 方向数 | 通过数 | 最大误差 | 结论 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for branch in sorted({row["branch"] for row in audit_rows}):
        part = [row for row in audit_rows if row["branch"] == branch]
        passed = sum(row["status"] == "publishable" for row in part)
        max_err = max(row["vector_rel_err"] for row in part) if part else float("nan")
        audit_table.append(f"| {branch} | {len(part)} | {passed} | {max_err:.3e} | {'通过' if passed == len(part) else '失败'} |")
    all_branches_ok = all(row["branch_ok"] for row in branch_rows)
    all_audit_ok = all(row["status"] == "publishable" for row in audit_rows)
    return "\n".join(
        [
            "# P2c79 n2 多初值分支复核与随机方向验证记录",
            "",
            "## 实验目的",
            "",
            "P2c78 已显示 n2 工况末端三维位置任务空间通过模态扰动验证。本轮加入多初值分支复核和随机输入方向验证，判断该候选是否可进入发布证据链。",
            "",
            "## 分支复核",
            "",
            *branch_table,
            "",
            "## 随机方向验证",
            "",
            *audit_table,
            "",
            "## 结论",
            "",
            f"分支一致性整体结果：`{all_branches_ok}`。随机方向整体结果：`{all_audit_ok}`。只有两者同时为真，才能把该工况列为两点磁控末端三维位置灵敏度发布候选。",
            "受信任分支上的随机方向若全部通过，说明该局部分支内的三维末端位置灵敏度稳定。若其他初值未回到同一物理解分支，则不能发布为初始化无关的稳健灵敏度，只能表述为分支条件下的局部导数。",
            "分支失败的主要归因应优先归为初始化敏感和多平衡分支风险，而不是任务空间线性化失败。",
            "",
            f"- 总耗时：`{wall_s:.2f} s`",
            "",
            "## 下一步计划",
            "",
            "1. 建立 continuation 分支复核，从零电流或低电流逐步加载到目标电流，检查是否能回到 trusted 分支。",
            "2. 若 continuation 通过，可将发布口径限定为“continuation 可达分支”。",
            "3. 若 continuation 仍失败，则将 n2 该工况标记为初始化敏感，只发布诊断结果或局部分支条件导数。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    params, I0 = p2c57.p2c24._params_from_saved_case(p2c57.CASE_NAME, p2c57.ALPHA)
    target_p, y_slack = p2c57.p2c24._load_saved_target_and_seed(p2c57.CASE_NAME, p2c57.ALPHA)
    y_seed = p2c57.p2c47._base_slack_to_segment_y(params, y_slack, steps_per_flex=p2c57.Args.steps_per_flex)
    branches = []
    for label, seed in _seed_map(params, y_seed).items():
        branches.append(_recover_branch(params, I0, target_p, seed, label))
    ref = branches[0]
    branch_rows = [_branch_row(branch, ref) for branch in branches]
    audit_rows = []
    for branch, brow in zip(branches, branch_rows):
        if not brow["baseline_ok"]:
            continue
        for name, direction in _direction_set(branch):
            audit_rows.append(_audit_direction(branch, I0, name, direction))
    wall_s = float(time.perf_counter() - t0)
    _plot(branch_rows, audit_rows)
    _write_csv(OUT_DIR / "branch_rows.csv", branch_rows)
    _write_csv(OUT_DIR / "random_direction_rows.csv", audit_rows)
    _write_csv(OUT_DIR / "summary.csv", branch_rows + audit_rows)
    failures = []
    for row in branch_rows:
        if not row["branch_ok"]:
            failures.append({"failure": "分支不一致或基线门禁失败", "evidence": row})
    for row in audit_rows:
        if row["status"] != "publishable":
            failures.append({"failure": row["failure_class"], "evidence": row})
    (OUT_DIR / "failure_events.jsonl").write_text("".join(json.dumps(_jsonable(item), ensure_ascii=False) + "\n" for item in failures), encoding="utf-8")
    payload = {
        "experiment": OUT_DIR.name,
        "eps": EPS,
        "random_dirs": RANDOM_DIRS,
        "wall_s": wall_s,
        "branch_rows": branch_rows,
        "audit_rows": audit_rows,
        "jax_devices": [str(device) for device in jax.devices()],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "record.md").write_text(_record(branch_rows, audit_rows, wall_s), encoding="utf-8")
    print(json.dumps(_jsonable({"wall_s": wall_s, "failures": failures}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
