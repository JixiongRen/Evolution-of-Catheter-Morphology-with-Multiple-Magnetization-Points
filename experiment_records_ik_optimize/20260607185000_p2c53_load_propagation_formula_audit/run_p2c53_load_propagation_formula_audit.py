"""Audit interface load propagation formulas on the n2 trusted branch."""

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
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _solve_trusted_branch() -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
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
    return params, current_i, target_p, last_solve, last_row


def _interface_rows(params: Any, y: np.ndarray) -> list[dict[str, Any]]:
    blocks = jnp.asarray(y, dtype=jnp.float64).reshape((len(params.flex), 6))
    x = initial_state(params, blocks[0])
    f_ref = float(params.scales.F_ref)
    m_ref = float(params.scales.M_ref)
    rows = []
    for i in range(len(params.flex) - 1):
        x = x.at[7:10].set(blocks[i, 0:3] * f_ref)
        x = x.at[10:13].set(blocks[i, 3:6] * m_ref)
        x = normalize_state(x)
        x, _flex_points = _integrate_flex(params, x, i, p2c51.Args.steps_per_flex)
        x_pre = x
        f_total, tau_total = p2c47._rigid_wrench(params, x_pre, i)
        x_post, _mid_p, _mid_q = _advance_rigid(params, x_pre, i)
        next_force = blocks[i + 1, 0:3] * f_ref
        next_torque = blocks[i + 1, 3:6] * m_ref
        candidates = {
            "当前刚段传播": (x_post[7:10], x_post[10:13]),
            "忽略刚段外载": (x_pre[7:10], x_pre[10:13]),
            "刚段外载反号": (x_pre[7:10] + f_total, x_pre[10:13] + tau_total),
        }
        for name, (cand_f, cand_m) in candidates.items():
            jump_f = next_force - cand_f
            jump_m = next_torque - cand_m
            rows.append(
                {
                    "interface": i,
                    "formula": name,
                    "jump_force_norm": float(jnp.linalg.norm(jump_f)),
                    "jump_torque_norm": float(jnp.linalg.norm(jump_m)),
                    "rigid_force_norm": float(jnp.linalg.norm(f_total)),
                    "rigid_torque_norm": float(jnp.linalg.norm(tau_total)),
                    "force_ratio": float(jnp.linalg.norm(jump_f) / jnp.maximum(jnp.linalg.norm(f_total), 1.0e-30)),
                    "torque_ratio": float(jnp.linalg.norm(jump_m) / jnp.maximum(jnp.linalg.norm(tau_total), 1.0e-30)),
                    "normalized_jump_norm": float(jnp.linalg.norm(jnp.concatenate([jump_f / f_ref, jump_m / m_ref], axis=0))),
                }
            )
        x = x_post
    return rows


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for formula in sorted({row["formula"] for row in rows}):
        sub = [row for row in rows if row["formula"] == formula]
        out.append(
            {
                "formula": formula,
                "max_force_ratio": max(row["force_ratio"] for row in sub),
                "max_torque_ratio": max(row["torque_ratio"] for row in sub),
                "max_normalized_jump_norm": max(row["normalized_jump_norm"] for row in sub),
                "sum_normalized_jump_norm": sum(row["normalized_jump_norm"] for row in sub),
            }
        )
    return out


def _plot(summaries: list[dict[str, Any]]) -> None:
    labels = [row["formula"] for row in summaries]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].bar(x, [row["max_force_ratio"] for row in summaries])
    axes[1].bar(x, [row["max_torque_ratio"] for row in summaries])
    axes[2].bar(x, [row["sum_normalized_jump_norm"] for row in summaries])
    axes[0].axhline(1.5, color="black", ls="--", lw=1.0)
    axes[0].set_ylabel("最大力跳变比")
    axes[1].set_ylabel("最大矩跳变比")
    axes[2].set_ylabel("归一化跳变总量")
    axes[2].set_xticks(x, labels, rotation=15, ha="right")
    for ax in axes:
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("两点可信分支接口载荷传播公式审计")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "load_propagation_formula_audit_cn.png", dpi=180)
    plt.close(fig)


def _record(base_row: dict[str, Any], summaries: list[dict[str, Any]], wall_s: float) -> str:
    table = [
        "| 公式假设 | 最大力跳变比 | 最大矩跳变比 | 归一化跳变总量 |",
        "| --- | --- | --- | --- |",
    ]
    for row in summaries:
        table.append(
            f"| {row['formula']} | {row['max_force_ratio']:.3e} | {row['max_torque_ratio']:.3e} | {row['sum_normalized_jump_norm']:.3e} |"
        )
    best = min(summaries, key=lambda row: row["sum_normalized_jump_norm"])
    return "\n".join(
        [
            "# P2c53 刚性磁体段载荷传播公式审计记录",
            "",
            "## 实验目的",
            "",
            "P2c52 说明两点工况的主要失败不随电流幅值降低而消失。本轮在可复现 P2c50 的 alpha=0.10 同源分支上审计接口载荷传播公式，判断当前失败是否可能来自刚段外载传播的符号或尺度错误。",
            "",
            "## 实验方案",
            "",
            "先复算 P2c50 的 task-first 细化路径，得到同一分支末状态。随后在每个柔段-刚段-柔段接口处比较三种传播假设：当前刚段传播、忽略刚段外载、刚段外载反号。比较力跳变比、矩跳变比和归一化跳变总量。",
            "",
            "## 基线复现",
            "",
            f"- 一阶最优性残差：`{base_row['stationarity_norm']:.3e}`",
            f"- 条件数：`{base_row['jacobian_condition']:.3e}`",
            f"- 接口连续残差：`{base_row['continuity_norm']:.3e}`",
            f"- 接口力跳变比：`{base_row['max_interface_force_jump_over_rigid_force']:.3e}`",
            f"- 末端位置误差：`{base_row['position_error_m']:.3e} m`",
            "",
            "## 公式审计结果",
            "",
            *table,
            "",
            "## 结论",
            "",
            f"归一化跳变总量最小的假设为 `{best['formula']}`。若该项不是当前刚段传播，说明载荷传播公式存在符号或尺度风险；若仍为当前刚段传播，则失败更可能来自约束相容性、任务可辨识性或模型闭合不足，而不是简单公式符号错误。",
            "",
            f"- 总耗时：`{wall_s:.2f} s`",
            "",
            "## 下一步计划",
            "",
            "1. 若替代公式显著更优，进入最小模型修正实验。",
            "2. 若当前公式仍最优，转向接口约束尺度、有效秩和分支相容性审计。",
            "3. 当前基线仍不可发布，不进入扰动灵敏度扫描。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    params, current_i, target_p, solve, base_row = _solve_trusted_branch()
    rows = _interface_rows(params, solve["y"])
    summaries = _summaries(rows)
    wall_s = float(time.perf_counter() - t0)
    _plot(summaries)
    _write_csv(OUT_DIR / "interface_formula_rows.csv", rows)
    _write_csv(OUT_DIR / "summary.csv", summaries)
    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": wall_s,
        "case": CASE_NAME,
        "alpha": ALPHA,
        "base_row": base_row,
        "summaries": summaries,
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
        json.dumps(
            _jsonable(
                {
                    "ts": time.time(),
                    "failure": "刚性磁体段载荷传播失配",
                    "evidence": base_row,
                    "formula_audit": summaries,
                }
            ),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "record.md").write_text(_record(base_row, summaries, wall_s), encoding="utf-8")
    print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
