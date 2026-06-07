"""Audit n1 combined load gates together with effective subspace evidence."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
P2C47_DIR = ROOT / "experiment_records_ik_optimize" / "20260607172000_p2c47_explicit_continuity_position_gate"
P2C48_DIR = ROOT / "experiment_records_ik_optimize" / "20260607174500_p2c48_n1_effective_subspace_diagnostic"
for path in (ROOT, P2C47_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk
from inverse_kinematics.ik_terminal_shooting_jax import _advance_rigid, _integrate_flex, initial_state, normalize_state

import run_p2c47_explicit_continuity_position_gate as p2c47

jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent
CASE_NAME = "n1_tip_z_rand15"
ALPHA = 0.10


class Args:
    """Settings matching the P2c47 n1 best baseline."""

    boundary_scale = 0.25
    reg_scale = 1.0e-4
    max_iter = 120
    steps_per_flex = 4
    position_scale = 60.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(ROOT), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc!r}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _solve_baseline() -> tuple[Any, dict[str, Any], dict[str, Any]]:
    params, current_i = p2c47.p2c24._params_from_saved_case(CASE_NAME, ALPHA)
    target_p, y_slack = p2c47.p2c24._load_saved_target_and_seed(CASE_NAME, ALPHA)
    y0 = p2c47._base_slack_to_segment_y(params, y_slack, steps_per_flex=Args.steps_per_flex)
    args = Args()
    funcs = p2c47._make_functions(params, target_p, y0, args, continuity_scale=1.0, boundary_mode="full")
    solve = p2c47._solve(y0, current_i, funcs, args)
    boundary = np.asarray(solve["out"]["boundary"], dtype=float).reshape(6)
    load = p2c47._interface_load_metrics(params, solve["y"], steps_per_flex=Args.steps_per_flex)
    row = {
        "case": CASE_NAME,
        "alpha": ALPHA,
        "position_error_m": float(np.linalg.norm(np.asarray(solve["out"]["tip_p"], dtype=float) - target_p)),
        "stationarity_norm": solve["stationarity_norm"],
        "jacobian_condition": solve["jacobian_condition"],
        "jacobian_rank": solve["jacobian_rank"],
        "continuity_norm": solve["continuity_norm"],
        "terminal_force_norm": float(np.linalg.norm(boundary[:3])),
        "terminal_torque_norm": float(np.linalg.norm(boundary[3:])),
        "nfev": solve["nfev"],
        "steady_s": solve["steady_s"],
        "success": solve["success"],
        **load,
    }
    row["baseline_gate"] = bool(
        row["success"]
        and row["stationarity_norm"] <= 1.0e-5
        and row["jacobian_condition"] < 1.0e7
        and row["continuity_norm"] <= 1.0e-3
        and row["max_interface_force_jump_over_rigid_force"] <= 1.5
    )
    return params, solve, row


def _interface_metrics(params: Any, y: np.ndarray) -> list[dict[str, Any]]:
    blocks = jnp.asarray(y, dtype=jnp.float64).reshape((len(params.flex), 6))
    x = initial_state(params, blocks[0])
    f_ref = float(params.scales.F_ref)
    m_ref = float(params.scales.M_ref)
    rows: list[dict[str, Any]] = []
    for i in range(len(params.flex) - 1):
        x = x.at[7:10].set(blocks[i, 0:3] * f_ref)
        x = x.at[10:13].set(blocks[i, 3:6] * m_ref)
        x = normalize_state(x)
        x, _flex_points = _integrate_flex(params, x, i, Args.steps_per_flex)
        pre_force = x[7:10]
        pre_torque = x[10:13]
        rigid_force, rigid_torque = p2c47._rigid_wrench(params, x, i)
        x_post, _mid_p, _mid_q = _advance_rigid(params, x, i)
        post_force = x_post[7:10]
        post_torque = x_post[10:13]
        next_force = blocks[i + 1, 0:3] * f_ref
        next_torque = blocks[i + 1, 3:6] * m_ref
        jump_force = next_force - post_force
        jump_torque = next_torque - post_torque
        internal_force_scale = jnp.maximum(jnp.maximum(jnp.linalg.norm(pre_force), jnp.linalg.norm(post_force)), jnp.linalg.norm(next_force))
        internal_torque_scale = jnp.maximum(jnp.maximum(jnp.linalg.norm(pre_torque), jnp.linalg.norm(post_torque)), jnp.linalg.norm(next_torque))
        rows.append(
            {
                "interface": int(i),
                "jump_force_norm": float(jnp.linalg.norm(jump_force)),
                "jump_torque_norm": float(jnp.linalg.norm(jump_torque)),
                "rigid_force_norm": float(jnp.linalg.norm(rigid_force)),
                "rigid_torque_norm": float(jnp.linalg.norm(rigid_torque)),
                "force_ratio_to_rigid_load": float(jnp.linalg.norm(jump_force) / jnp.maximum(jnp.linalg.norm(rigid_force), 1.0e-30)),
                "force_ratio_to_internal_force": float(jnp.linalg.norm(jump_force) / jnp.maximum(internal_force_scale, 1.0e-30)),
                "torque_ratio_to_internal_torque": float(jnp.linalg.norm(jump_torque) / jnp.maximum(internal_torque_scale, 1.0e-30)),
                "normalized_jump_norm": float(jnp.linalg.norm(jnp.concatenate([jump_force / f_ref, jump_torque / m_ref], axis=0))),
            }
        )
        x = x_post
    return rows


def _load_subspace() -> dict[str, Any]:
    payload = json.loads((P2C48_DIR / "summary.json").read_text(encoding="utf-8"))
    return {
        "implicit_singular_values": payload["implicit_position_response"]["singular_values"],
        "implicit_effective_rank_rel_0p1": payload["implicit_position_response"]["effective_rank_rel_0p1"],
        "direct_effective_rank_rel_0p1": payload["direct_position_response"]["effective_rank_rel_0p1"],
    }


def _summary(base: dict[str, Any], rows: list[dict[str, Any]], subspace: dict[str, Any]) -> dict[str, Any]:
    return {
        "publish_gate": False,
        "subspace_publishable": False,
        "diagnostic_subspace_rank": subspace["implicit_effective_rank_rel_0p1"],
        "stationarity_norm": base["stationarity_norm"],
        "jacobian_condition": base["jacobian_condition"],
        "continuity_norm": base["continuity_norm"],
        "max_force_ratio_to_rigid_load": max(row["force_ratio_to_rigid_load"] for row in rows),
        "max_force_ratio_to_internal_force": max(row["force_ratio_to_internal_force"] for row in rows),
        "max_abs_force_jump_N": max(row["jump_force_norm"] for row in rows),
        "max_torque_ratio_to_internal_torque": max(row["torque_ratio_to_internal_torque"] for row in rows),
    }


def _plot(rows: list[dict[str, Any]]) -> None:
    labels = [f"接口 {row['interface']}" for row in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].bar(x, [row["jump_force_norm"] for row in rows])
    axes[1].bar(x, [row["force_ratio_to_rigid_load"] for row in rows], label="相对刚段外力")
    axes[1].bar(x, [row["force_ratio_to_internal_force"] for row in rows], alpha=0.65, label="相对接口内力")
    axes[2].bar(x, [row["torque_ratio_to_internal_torque"] for row in rows])
    axes[0].set_ylabel("绝对力跳变/N")
    axes[1].set_ylabel("力跳变比例")
    axes[2].set_ylabel("相对内矩跳变")
    axes[2].set_xticks(x, labels)
    axes[1].legend()
    for ax in axes:
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("单点工况组合门禁与有效子空间审计")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n1_combined_gate_subspace_cn.png", dpi=180)
    plt.close(fig)


def _record(base: dict[str, Any], rows: list[dict[str, Any]], subspace: dict[str, Any], summary: dict[str, Any], wall_s: float) -> str:
    table = [
        "| 接口 | 绝对力跳变/N | 相对刚段外力 | 相对接口内力 | 相对接口内矩 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        table.append(
            f"| {row['interface']} | {row['jump_force_norm']:.3e} | {row['force_ratio_to_rigid_load']:.3e} | "
            f"{row['force_ratio_to_internal_force']:.3e} | {row['torque_ratio_to_internal_torque']:.3e} |"
        )
    return "\n".join(
        [
            "# P2c56 单点组合门禁与有效子空间审计记录",
            "",
            "## 实验目的",
            "",
            "P2c48 显示单点磁控在近似隐式响应下具有二维有效子空间，但物理基线未通过。本轮复算 n1 显式连续基线，并叠加组合载荷门禁，判断该二维子空间是否具备发布或诊断价值。",
            "",
            "## 基线结果",
            "",
            f"- 一阶最优性残差：`{base['stationarity_norm']:.3e}`",
            f"- 条件数：`{base['jacobian_condition']:.3e}`",
            f"- 接口连续残差：`{base['continuity_norm']:.3e}`",
            f"- 末端位置误差：`{base['position_error_m']:.3e} m`",
            f"- 隐式响应有效秩：`{subspace['implicit_effective_rank_rel_0p1']}`",
            f"- 隐式响应奇异值：`{subspace['implicit_singular_values']}`",
            "",
            "## 组合载荷门禁",
            "",
            *table,
            "",
            "## 判定",
            "",
            f"- 发布级门禁：`{summary['publish_gate']}`",
            f"- 有效子空间是否可发布：`{summary['subspace_publishable']}`",
            f"- 诊断有效秩：`{summary['diagnostic_subspace_rank']}`",
            "",
            "## 结论",
            "",
            "单点工况当前不能发布有效子空间灵敏度。虽然近似隐式响应呈二维结构，但基线一阶最优性、条件数和接口力传播均严重失败。该二维结构只能作为模型诊断线索，不能作为外层 IK 的可信灵敏度。",
            "",
            f"- 总耗时：`{wall_s:.2f} s`",
            "",
            "## 下一步计划",
            "",
            "1. 单点方向停止完整三维灵敏度目标，保留二维有效子空间作为诊断对象。",
            "2. 若继续推进单点，应先修复一阶最优性和条件数，而不是扩大扰动扫描。",
            "3. 与 n2 结果合并，形成当前单点/两点不可发布边界说明。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    params, solve, base = _solve_baseline()
    rows = _interface_metrics(params, solve["y"])
    subspace = _load_subspace()
    summary = _summary(base, rows, subspace)
    wall_s = float(time.perf_counter() - t0)
    _plot(rows)
    _write_csv(OUT_DIR / "combined_gate_rows.csv", rows)
    _write_csv(OUT_DIR / "summary.csv", [summary])
    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": wall_s,
        "case": CASE_NAME,
        "alpha": ALPHA,
        "base": base,
        "subspace": subspace,
        "summary": summary,
        "rows": rows,
        "git": {
            "branch": _run(["git", "branch", "--show-current"]),
            "commit": _run(["git", "rev-parse", "HEAD"]),
            "status_short": _run(["git", "status", "--short"]),
        },
        "jax_devices": [str(device) for device in jax.devices()],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "failure_events.jsonl").write_text(
        json.dumps(_jsonable({"failure": "单点有效子空间不可发布", "summary": summary}), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "record.md").write_text(_record(base, rows, subspace, summary, wall_s), encoding="utf-8")
    print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
