"""Probe split force/torque interface-continuity weights for n2 shooting."""

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
for path in (ROOT, P2C24_DIR, P2C47_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk
from inverse_kinematics.ik_terminal_shooting_jax import forward_segment_wrench

import run_p2c24_interface_weight_continuation_repair as p2c24
import run_p2c47_explicit_continuity_position_gate as p2c47

jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent
CASE_NAME = "n2_z_pair_rand12"
ALPHA = 0.10
FORCE_SCALES = (0.25, 1.0, 4.0, 16.0)
TORQUE_SCALE = 0.25
STAGES = (
    (10.0, 120),
    (20.0, 120),
    (40.0, 120),
    (60.0, 240),
)


class Args:
    """Settings for split-continuity least-squares solves."""

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


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def _make_split_functions(params_base: Any, target_p: np.ndarray, y_ref: np.ndarray, args: Args, *, force_scale: float, torque_scale: float):
    target_p_j = jnp.asarray(target_p, dtype=jnp.float64).reshape(3)
    y_ref_j = jnp.asarray(y_ref, dtype=jnp.float64).reshape(-1)
    mask = p2c47._mode_mask("full")

    def params_for_I(I: jnp.ndarray) -> Any:
        return p2c47.with_coil_currents(params_base, jnp.asarray(I, dtype=jnp.float64).reshape(8))

    def output_yI(y: jnp.ndarray, I: jnp.ndarray) -> Any:
        return forward_segment_wrench(params_for_I(I), y, steps_per_flex=args.steps_per_flex)

    def continuity_yI(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        return p2c47.p2c8.interface_wrench_residual(params_for_I(I), y, steps_per_flex=args.steps_per_flex)

    def split_continuity_yI(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        raw = continuity_yI(y, I).reshape((-1, 6))
        force = float(force_scale) * raw[:, 0:3]
        torque = float(torque_scale) * raw[:, 3:6]
        return jnp.concatenate([force, torque], axis=1).reshape(-1)

    def residual_yI(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        params = params_for_I(I)
        out = forward_segment_wrench(params, y, steps_per_flex=args.steps_per_flex)
        r_p = float(args.position_scale) * (out.tip_p - target_p_j)
        r_b = float(args.boundary_scale) * mask * out.boundary
        r_c = split_continuity_yI(y, I)
        r_reg = float(args.reg_scale) * (jnp.asarray(y, dtype=jnp.float64).reshape(-1) - y_ref_j)
        return jnp.concatenate([r_p.reshape(-1), r_b.reshape(-1), r_c.reshape(-1), r_reg.reshape(-1)])

    return {
        "residual": jax.jit(residual_yI),
        "jacobian": jax.jit(jax.jacfwd(residual_yI, argnums=0)),
        "output": jax.jit(output_yI),
        "continuity": jax.jit(continuity_yI),
    }


def _solve_stage(params: Any, current_i: np.ndarray, target_p: np.ndarray, y_ref: np.ndarray, y0: np.ndarray, *, pos_scale: float, force_scale: float, max_iter: int) -> dict[str, Any]:
    args = Args()
    args.position_scale = float(pos_scale)
    args.max_iter = int(max_iter)
    funcs = _make_split_functions(params, target_p, y_ref, args, force_scale=force_scale, torque_scale=TORQUE_SCALE)
    return p2c47._solve(y0, current_i, funcs, args)


def _combined_force_metrics(params: Any, y: np.ndarray) -> dict[str, float]:
    blocks = jnp.asarray(y, dtype=jnp.float64).reshape((len(params.flex), 6))
    x = p2c47.initial_state(params, blocks[0])
    f_ref = float(params.scales.F_ref)
    force_rigid = []
    force_internal = []
    force_abs = []
    for i in range(len(params.flex) - 1):
        x = x.at[7:10].set(blocks[i, 0:3] * f_ref)
        x = x.at[10:13].set(blocks[i, 3:6] * float(params.scales.M_ref))
        x = p2c47.normalize_state(x)
        x, _flex_points = p2c47._integrate_flex(params, x, i, Args.steps_per_flex)
        pre_force = x[7:10]
        rigid_force, _rigid_torque = p2c47._rigid_wrench(params, x, i)
        x, _mid_p, _mid_q = p2c47._advance_rigid(params, x, i)
        post_force = x[7:10]
        next_force = blocks[i + 1, 0:3] * f_ref
        jump_force = next_force - post_force
        internal_scale = jnp.maximum(jnp.maximum(jnp.linalg.norm(pre_force), jnp.linalg.norm(post_force)), jnp.linalg.norm(next_force))
        force_abs.append(float(jnp.linalg.norm(jump_force)))
        force_rigid.append(float(jnp.linalg.norm(jump_force) / jnp.maximum(jnp.linalg.norm(rigid_force), 1.0e-30)))
        force_internal.append(float(jnp.linalg.norm(jump_force) / jnp.maximum(internal_scale, 1.0e-30)))
    return {
        "max_force_jump_abs_N": max(force_abs, default=0.0),
        "max_force_ratio_to_rigid_load": max(force_rigid, default=0.0),
        "max_force_ratio_to_internal_force": max(force_internal, default=0.0),
    }


def _row(*, force_scale: float, stage: int, pos_scale: float, max_iter: int, params: Any, target_p: np.ndarray, solve: dict[str, Any]) -> dict[str, Any]:
    boundary = np.asarray(solve["out"]["boundary"], dtype=float).reshape(6)
    load = p2c47._interface_load_metrics(params, solve["y"], steps_per_flex=Args.steps_per_flex)
    combined = _combined_force_metrics(params, solve["y"])
    row = {
        "case": CASE_NAME,
        "alpha": ALPHA,
        "force_continuity_scale": force_scale,
        "torque_continuity_scale": TORQUE_SCALE,
        "stage": stage,
        "position_scale": pos_scale,
        "max_iter": max_iter,
        "position_error_m": float(np.linalg.norm(np.asarray(solve["out"]["tip_p"], dtype=float) - target_p)),
        "stationarity_norm": solve["stationarity_norm"],
        "jacobian_condition": solve["jacobian_condition"],
        "continuity_norm": solve["continuity_norm"],
        "terminal_force_norm": float(np.linalg.norm(boundary[:3])),
        "terminal_torque_norm": float(np.linalg.norm(boundary[3:])),
        "nfev": solve["nfev"],
        "steady_s": solve["steady_s"],
        "success": solve["success"],
        **load,
        **combined,
    }
    row["baseline_gate"] = bool(
        row["success"]
        and row["stationarity_norm"] <= 1.0e-5
        and row["jacobian_condition"] < 1.0e7
        and row["continuity_norm"] <= 1.0e-3
        and row["max_interface_force_jump_over_rigid_force"] <= 1.5
    )
    if row["baseline_gate"]:
        row["failure_class"] = "通过"
    elif row["stationarity_norm"] > 1.0e-5:
        row["failure_class"] = "一阶最优性退化"
    elif row["jacobian_condition"] >= 1.0e9:
        row["failure_class"] = "条件数严重病态"
    elif row["continuity_norm"] > 1.0e-3 or row["max_interface_force_jump_over_rigid_force"] > 1.5:
        row["failure_class"] = "接口力连续性失配"
    else:
        row["failure_class"] = "综合失败"
    return row


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for force_scale in FORCE_SCALES:
        sub = [row for row in rows if row["force_continuity_scale"] == force_scale]
        final = max(sub, key=lambda row: row["stage"])
        out.append(
            {
                "force_continuity_scale": force_scale,
                "any_gate_pass": any(row["baseline_gate"] for row in sub),
                "final_stationarity_norm": final["stationarity_norm"],
                "final_condition": final["jacobian_condition"],
                "final_continuity_norm": final["continuity_norm"],
                "final_force_ratio_to_rigid_load": final["max_interface_force_jump_over_rigid_force"],
                "final_force_ratio_to_internal_force": final["max_force_ratio_to_internal_force"],
                "final_position_error_m": final["position_error_m"],
                "dominant_failure": final["failure_class"],
            }
        )
    return out


def _plot(summaries: list[dict[str, Any]]) -> None:
    x = [row["force_continuity_scale"] for row in summaries]
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(x, [row["final_stationarity_norm"] for row in summaries], marker="o")
    axes[1].plot(x, [row["final_condition"] for row in summaries], marker="o")
    axes[2].plot(x, [row["final_force_ratio_to_rigid_load"] for row in summaries], marker="o", label="相对刚段外力")
    axes[2].plot(x, [row["final_force_ratio_to_internal_force"] for row in summaries], marker="s", label="相对接口内力")
    axes[3].plot(x, [row["final_continuity_norm"] for row in summaries], marker="o")
    axes[0].axhline(1.0e-5, color="black", ls="--", lw=1.0)
    axes[1].axhline(1.0e7, color="black", ls="--", lw=1.0)
    axes[2].axhline(1.5, color="black", ls="--", lw=1.0)
    axes[3].axhline(1.0e-3, color="black", ls="--", lw=1.0)
    axes[0].set_ylabel("一阶最优性")
    axes[1].set_ylabel("条件数")
    axes[2].set_ylabel("力跳变比")
    axes[3].set_ylabel("接口连续残差")
    axes[3].set_xlabel("力连续权重")
    axes[2].legend()
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
    fig.suptitle("n2 力/矩连续分权重修复探针")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "force_continuity_weight_repair_cn.png", dpi=180)
    plt.close(fig)


def _record(summaries: list[dict[str, Any]], wall_s: float) -> str:
    table = [
        "| 力连续权重 | 是否过门 | 一阶最优性 | 条件数 | 接口连续残差 | 相对刚段外力 | 相对接口内力 | 末端误差/m | 失败分类 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summaries:
        table.append(
            f"| {row['force_continuity_scale']:.3g} | {row['any_gate_pass']} | {row['final_stationarity_norm']:.3e} | "
            f"{row['final_condition']:.3e} | {row['final_continuity_norm']:.3e} | "
            f"{row['final_force_ratio_to_rigid_load']:.3e} | {row['final_force_ratio_to_internal_force']:.3e} | "
            f"{row['final_position_error_m']:.3e} | {row['dominant_failure']} |"
        )
    best = min(summaries, key=lambda row: (row["final_force_ratio_to_internal_force"], row["final_continuity_norm"]))
    return "\n".join(
        [
            "# P2c57 n2 力/矩连续分权重修复记录",
            "",
            "## 实验目的",
            "",
            "P2c55 表明 n2 的主要问题是接口力连续性，而接口内矩跳变很小。本轮将接口连续残差拆成力连续权重和矩连续权重，只扫描力连续权重，检验能否降低接口力跳变并保持一阶最优性。",
            "",
            "## 实验方案",
            "",
            "固定 `n2_z_pair_rand12, alpha=0.10`、position-only 任务和 P2c50 同源初值。矩连续权重保持 `0.25`，力连续权重扫描 `0.25,1,4,16`。每组使用相同任务权重延拓。",
            "",
            "## 结果",
            "",
            *table,
            "",
            "## 结论",
            "",
            f"相对接口内力跳变最小的设置为力连续权重 `{best['force_continuity_scale']:.3g}`。若该设置仍不能同时满足一阶最优性、条件数和载荷传播门禁，则说明简单提高力连续权重不足以修复当前模型，需要进入状态参数化或约束表述修复。",
            "",
            f"- 总耗时：`{wall_s:.2f} s`",
            "",
            "## 下一步计划",
            "",
            "1. 若存在过门设置，进入同分支扰动重求解审计。",
            "2. 若仅降低力跳变但一阶最优性恶化，说明约束相容性不足。",
            "3. 若全部失败，下一步应尝试低维力传递参数化，而不是继续增大权重。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    failures = OUT_DIR / "failure_events.jsonl"
    failures.write_text("", encoding="utf-8")
    params, current_i = p2c24._params_from_saved_case(CASE_NAME, ALPHA)
    target_p, y_slack = p2c24._load_saved_target_and_seed(CASE_NAME, ALPHA)
    y_seed = p2c47._base_slack_to_segment_y(params, y_slack, steps_per_flex=Args.steps_per_flex)

    rows: list[dict[str, Any]] = []
    for force_scale in FORCE_SCALES:
        y = y_seed.copy()
        y_ref = y.copy()
        for stage, (pos_scale, max_iter) in enumerate(STAGES, start=1):
            solve = _solve_stage(params, current_i, target_p, y_ref, y, pos_scale=pos_scale, force_scale=force_scale, max_iter=max_iter)
            row = _row(force_scale=force_scale, stage=stage, pos_scale=pos_scale, max_iter=max_iter, params=params, target_p=target_p, solve=solve)
            rows.append(row)
            y = solve["y"]
            if not row["baseline_gate"]:
                _append_jsonl(failures, {"ts": time.time(), "failure": row["failure_class"], "evidence": row})

    summaries = _summaries(rows)
    wall_s = float(time.perf_counter() - t0)
    _plot(summaries)
    _write_csv(OUT_DIR / "force_continuity_rows.csv", rows)
    _write_csv(OUT_DIR / "summary.csv", summaries)
    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": wall_s,
        "case": CASE_NAME,
        "alpha": ALPHA,
        "force_scales": list(FORCE_SCALES),
        "torque_scale": TORQUE_SCALE,
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
    (OUT_DIR / "record.md").write_text(_record(summaries, wall_s), encoding="utf-8")
    print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
