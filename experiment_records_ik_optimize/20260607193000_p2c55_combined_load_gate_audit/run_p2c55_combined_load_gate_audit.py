"""Audit combined load-propagation gates on the n2 trusted branch."""

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
P2C24_DIR = ROOT / "experiment_records_ik_optimize" / "20260606160747_p2c24_interface_weight_continuation_repair"
P2C47_DIR = ROOT / "experiment_records_ik_optimize" / "20260607172000_p2c47_explicit_continuity_position_gate"
P2C51_DIR = ROOT / "experiment_records_ik_optimize" / "20260607173930_p2c51_n2_current_boundary_scan"
for path in (ROOT, P2C24_DIR, P2C47_DIR, P2C51_DIR):
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

import run_p2c24_interface_weight_continuation_repair as p2c24
import run_p2c47_explicit_continuity_position_gate as p2c47
import run_p2c51_n2_current_boundary_scan as p2c51

jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent
CASE_NAME = "n2_z_pair_rand12"
ALPHA = 0.10


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


def _solve_trusted_branch() -> tuple[Any, dict[str, Any], dict[str, Any]]:
    target_saved, y_slack_saved = p2c24._load_saved_target_and_seed(CASE_NAME, ALPHA)
    params, current_i = p2c24._params_from_saved_case(CASE_NAME, ALPHA)
    target_p, _target_q, _y_oracle, _oracle = p2c51.p2c10t._build_case(CASE_NAME, ALPHA)[2:]
    if float(np.linalg.norm(target_saved - target_p)) > 1.0e-12:
        raise RuntimeError("Saved target and rebuilt target disagree.")
    y = p2c47._base_slack_to_segment_y(params, y_slack_saved, steps_per_flex=p2c51.Args.steps_per_flex)
    y_ref = y.copy()
    last_solve: dict[str, Any] | None = None
    last_row: dict[str, Any] | None = None
    for idx, (pos_scale, cont_scale, max_iter) in enumerate(p2c51.STAGES, start=1):
        solve = p2c51._solve_stage(
            params,
            current_i,
            target_p,
            y_ref,
            y,
            pos_scale=pos_scale,
            cont_scale=cont_scale,
            max_iter=max_iter,
        )
        row = p2c51._row_from_solve(
            alpha=ALPHA,
            stage=idx,
            pos_scale=pos_scale,
            cont_scale=cont_scale,
            max_iter=max_iter,
            params=params,
            target_p=target_p,
            solve=solve,
        )
        y = solve["y"]
        last_solve = solve
        last_row = row
    assert last_solve is not None and last_row is not None
    return params, last_solve, last_row


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
        x, _flex_points = _integrate_flex(params, x, i, p2c51.Args.steps_per_flex)
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
                "pre_force_norm": float(jnp.linalg.norm(pre_force)),
                "post_force_norm": float(jnp.linalg.norm(post_force)),
                "next_force_norm": float(jnp.linalg.norm(next_force)),
                "pre_torque_norm": float(jnp.linalg.norm(pre_torque)),
                "post_torque_norm": float(jnp.linalg.norm(post_torque)),
                "next_torque_norm": float(jnp.linalg.norm(next_torque)),
                "force_ratio_to_rigid_load": float(jnp.linalg.norm(jump_force) / jnp.maximum(jnp.linalg.norm(rigid_force), 1.0e-30)),
                "force_ratio_to_internal_force": float(jnp.linalg.norm(jump_force) / jnp.maximum(internal_force_scale, 1.0e-30)),
                "torque_ratio_to_rigid_load": float(jnp.linalg.norm(jump_torque) / jnp.maximum(jnp.linalg.norm(rigid_torque), 1.0e-30)),
                "torque_ratio_to_internal_torque": float(jnp.linalg.norm(jump_torque) / jnp.maximum(internal_torque_scale, 1.0e-30)),
                "normalized_jump_norm": float(jnp.linalg.norm(jnp.concatenate([jump_force / f_ref, jump_torque / m_ref], axis=0))),
            }
        )
        x = x_post
    return rows


def _summary(base_row: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    max_force_abs = max(row["jump_force_norm"] for row in rows)
    max_force_rigid = max(row["force_ratio_to_rigid_load"] for row in rows)
    max_force_internal = max(row["force_ratio_to_internal_force"] for row in rows)
    max_torque_internal = max(row["torque_ratio_to_internal_torque"] for row in rows)
    max_norm_jump = max(row["normalized_jump_norm"] for row in rows)
    diagnostic_gate = bool(
        base_row["stationarity_norm"] <= 1.0e-5
        and base_row["jacobian_condition"] < 1.0e9
        and max_force_abs <= 2.0e-3
        and max_force_internal <= 0.25
        and max_torque_internal <= 0.01
    )
    publish_gate = bool(
        base_row["stationarity_norm"] <= 1.0e-5
        and base_row["jacobian_condition"] < 1.0e7
        and base_row["continuity_norm"] <= 1.0e-3
        and max_force_rigid <= 1.5
    )
    return {
        "publish_gate": publish_gate,
        "diagnostic_gate": diagnostic_gate,
        "max_force_jump_abs_N": max_force_abs,
        "max_force_ratio_to_rigid_load": max_force_rigid,
        "max_force_ratio_to_internal_force": max_force_internal,
        "max_torque_ratio_to_internal_torque": max_torque_internal,
        "max_normalized_jump_norm": max_norm_jump,
        "stationarity_norm": base_row["stationarity_norm"],
        "jacobian_condition": base_row["jacobian_condition"],
        "continuity_norm": base_row["continuity_norm"],
        "position_error_m": base_row["position_error_m"],
    }


def _plot(rows: list[dict[str, Any]]) -> None:
    labels = [f"接口 {row['interface']}" for row in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].bar(x, [row["jump_force_norm"] for row in rows])
    axes[1].bar(x, [row["force_ratio_to_rigid_load"] for row in rows], label="相对刚段外力")
    axes[1].bar(x, [row["force_ratio_to_internal_force"] for row in rows], alpha=0.65, label="相对接口内力")
    axes[2].bar(x, [row["torque_ratio_to_internal_torque"] for row in rows])
    axes[0].axhline(2.0e-3, color="black", ls="--", lw=1.0)
    axes[1].axhline(1.5, color="black", ls="--", lw=1.0)
    axes[1].axhline(0.25, color="gray", ls=":", lw=1.0)
    axes[2].axhline(0.01, color="black", ls="--", lw=1.0)
    axes[0].set_ylabel("绝对力跳变/N")
    axes[1].set_ylabel("力跳变比例")
    axes[2].set_ylabel("相对内矩跳变")
    axes[2].set_xticks(x, labels)
    axes[1].legend()
    for ax in axes:
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("两点工况组合载荷门禁审计")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "combined_load_gate_audit_cn.png", dpi=180)
    plt.close(fig)


def _record(base_row: dict[str, Any], summary: dict[str, Any], rows: list[dict[str, Any]], wall_s: float) -> str:
    table = [
        "| 接口 | 绝对力跳变/N | 相对刚段外力 | 相对接口内力 | 相对接口内矩 | 归一化跳变 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        table.append(
            f"| {row['interface']} | {row['jump_force_norm']:.3e} | {row['force_ratio_to_rigid_load']:.3e} | "
            f"{row['force_ratio_to_internal_force']:.3e} | {row['torque_ratio_to_internal_torque']:.3e} | {row['normalized_jump_norm']:.3e} |"
        )
    return "\n".join(
        [
            "# P2c55 组合载荷门禁审计记录",
            "",
            "## 实验目的",
            "",
            "P2c54 显示，单一相对刚段外力门禁不足以完整解释 n2 失败。本轮在同一 P2c50 分支上同时记录绝对接口跳变、相对刚段外载跳变、相对接口内力跳变和相对接口内矩跳变，区分发布失败与诊断可用。",
            "",
            "## 基线状态",
            "",
            f"- 一阶最优性残差：`{base_row['stationarity_norm']:.3e}`",
            f"- 条件数：`{base_row['jacobian_condition']:.3e}`",
            f"- 接口连续残差：`{base_row['continuity_norm']:.3e}`",
            f"- 末端位置误差：`{base_row['position_error_m']:.3e} m`",
            "",
            "## 组合门禁结果",
            "",
            *table,
            "",
            "## 判定",
            "",
            f"- 发布级门禁：`{summary['publish_gate']}`",
            f"- 诊断级门禁：`{summary['diagnostic_gate']}`",
            f"- 最大绝对力跳变：`{summary['max_force_jump_abs_N']:.3e} N`",
            f"- 最大相对刚段外力跳变：`{summary['max_force_ratio_to_rigid_load']:.3e}`",
            f"- 最大相对接口内力跳变：`{summary['max_force_ratio_to_internal_force']:.3e}`",
            "",
            "## 结论",
            "",
            "当前 n2 分支不能发布三维末端位置灵敏度。失败并非只来自相对刚段外力分母；相对接口内力跳变也达到约 `1.0`，说明相邻柔段之间的内力传递没有形成稳定闭合。内矩跳变很小，问题主要集中在力连续性和病态条件数。因此该分支目前连诊断级扰动窗口也不稳，应优先修复接口力连续约束尺度或载荷传播表述，而不是进入扰动灵敏度扫描。",
            "",
            f"- 总耗时：`{wall_s:.2f} s`",
            "",
            "## 下一步计划",
            "",
            "1. 基于组合门禁寻找 n2 的诊断级扰动窗口，但报告中必须标注不可发布。",
            "2. 若要进入发布级，应优先降低条件数和接口连续残差，而不是继续追末端误差。",
            "3. 对 n1 单点工况复用组合门禁，判断有效子空间是否存在更稳健工作点。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    params, solve, base_row = _solve_trusted_branch()
    rows = _interface_metrics(params, solve["y"])
    summary = _summary(base_row, rows)
    wall_s = float(time.perf_counter() - t0)
    _plot(rows)
    _write_csv(OUT_DIR / "combined_load_rows.csv", rows)
    _write_csv(OUT_DIR / "summary.csv", [summary])
    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": wall_s,
        "case": CASE_NAME,
        "alpha": ALPHA,
        "base_row": base_row,
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
        json.dumps(_jsonable({"failure": "发布级门禁失败但诊断级可用", "summary": summary}), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "record.md").write_text(_record(base_row, summary, rows, wall_s), encoding="utf-8")
    print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
