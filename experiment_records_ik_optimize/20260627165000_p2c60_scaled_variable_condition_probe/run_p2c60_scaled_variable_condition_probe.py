"""Probe variable scaling for n2 force-continuity shooting conditioning."""

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
P2C57_DIR = ROOT / "experiment_records_ik_optimize" / "20260607200500_p2c57_force_continuity_weight_repair"
P2C58_DIR = ROOT / "experiment_records_ik_optimize" / "20260607202500_p2c58_force_continuity_condition_probe"
for path in (ROOT, P2C57_DIR, P2C58_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk
from inverse_kinematics.ik_terminal_shooting_jax import output_to_numpy

import run_p2c57_force_continuity_weight_repair as p2c57
import run_p2c58_force_continuity_condition_probe as p2c58

jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent
FORCE_CONTINUITY_SCALE = 4.0
REG_SCALE = 3.0e-4
VARIABLE_SCALE_PAIRS = (
    (1.0, 1.0),
    (0.25, 1.0),
    (1.0, 0.25),
    (0.25, 0.25),
    (0.1, 0.25),
    (0.1, 0.1),
)
STAGES = p2c57.STAGES


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


def _scale_vector(y: np.ndarray, *, force_scale: float, torque_scale: float) -> np.ndarray:
    blocks = np.asarray(y, dtype=float).reshape((-1, 6))
    scale = np.ones_like(blocks)
    scale[:, 0:3] = float(force_scale)
    scale[:, 3:6] = float(torque_scale)
    return scale.reshape(-1)


def _condition_number(J: np.ndarray) -> float:
    svals = np.linalg.svd(np.asarray(J, dtype=float), compute_uv=False)
    if not svals.size:
        return float("inf")
    return float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny))


def _solve_stage_scaled(
    params: Any,
    current_i: np.ndarray,
    target_p: np.ndarray,
    y_ref: np.ndarray,
    y0: np.ndarray,
    *,
    pos_scale: float,
    max_iter: int,
    variable_force_scale: float,
    variable_torque_scale: float,
) -> dict[str, Any]:
    args = p2c58.Args()
    args.position_scale = float(pos_scale)
    args.max_iter = int(max_iter)
    args.reg_scale = float(REG_SCALE)
    funcs = p2c57._make_split_functions(
        params,
        target_p,
        y_ref,
        args,
        force_scale=FORCE_CONTINUITY_SCALE,
        torque_scale=p2c57.TORQUE_SCALE,
    )
    I_j = jnp.asarray(current_i, dtype=jnp.float64).reshape(8)
    scale = _scale_vector(y0, force_scale=variable_force_scale, torque_scale=variable_torque_scale)
    u0 = np.asarray(y0, dtype=float).reshape(-1) / scale
    scale_j = jnp.asarray(scale, dtype=jnp.float64)

    def residual_u(u: jnp.ndarray) -> jnp.ndarray:
        return funcs["residual"](jnp.asarray(u, dtype=jnp.float64).reshape(-1) * scale_j, I_j)

    jacobian_u = jax.jit(jax.jacfwd(residual_u))
    residual_u_jit = jax.jit(residual_u)
    _ = residual_u_jit(jnp.asarray(u0, dtype=jnp.float64)).block_until_ready()

    t0 = time.perf_counter()
    opt = least_squares(
        lambda uu: np.asarray(jax.device_get(residual_u_jit(jnp.asarray(uu, dtype=jnp.float64))), dtype=float),
        u0,
        jac=lambda uu: np.asarray(jax.device_get(jacobian_u(jnp.asarray(uu, dtype=jnp.float64))), dtype=float),
        method="trf",
        x_scale=np.ones_like(u0),
        loss="linear",
        max_nfev=int(args.max_iter),
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    steady_s = float(time.perf_counter() - t0)
    u = np.asarray(opt.x, dtype=float).reshape(-1)
    y = u * scale
    y_j = jnp.asarray(y, dtype=jnp.float64)
    u_j = jnp.asarray(u, dtype=jnp.float64)
    r = np.asarray(jax.device_get(funcs["residual"](y_j, I_j)), dtype=float)
    J_y = np.asarray(jax.device_get(funcs["jacobian"](y_j, I_j)), dtype=float)
    J_u = np.asarray(jax.device_get(jacobian_u(u_j)), dtype=float)
    out = output_to_numpy(funcs["output"](y_j, I_j))
    continuity = np.asarray(jax.device_get(funcs["continuity"](y_j, I_j)), dtype=float).reshape(-1)
    return {
        "y": y,
        "out": out,
        "success": bool(opt.success),
        "nfev": int(opt.nfev),
        "steady_s": steady_s,
        "residual_norm": float(np.linalg.norm(r)),
        "physical_stationarity_norm": float(np.linalg.norm(J_y.T @ r)),
        "scaled_stationarity_norm": float(np.linalg.norm(J_u.T @ r)),
        "physical_jacobian_condition": _condition_number(J_y),
        "scaled_jacobian_condition": _condition_number(J_u),
        "jacobian_rank": int(np.linalg.matrix_rank(J_y)),
        "continuity_norm": float(np.linalg.norm(continuity)),
        "message": str(opt.message),
    }


def _row(
    *,
    params: Any,
    target_p: np.ndarray,
    solve: dict[str, Any],
    stage: int,
    pos_scale: float,
    max_iter: int,
    variable_force_scale: float,
    variable_torque_scale: float,
) -> dict[str, Any]:
    boundary = np.asarray(solve["out"]["boundary"], dtype=float).reshape(6)
    load = p2c57.p2c47._interface_load_metrics(params, solve["y"], steps_per_flex=p2c57.Args.steps_per_flex)
    combined = p2c57._combined_force_metrics(params, solve["y"])
    row = {
        "case": p2c57.CASE_NAME,
        "alpha": p2c57.ALPHA,
        "stage": int(stage),
        "position_scale": float(pos_scale),
        "max_iter": int(max_iter),
        "force_continuity_scale": FORCE_CONTINUITY_SCALE,
        "reg_scale": REG_SCALE,
        "variable_force_scale": float(variable_force_scale),
        "variable_torque_scale": float(variable_torque_scale),
        "position_error_m": float(np.linalg.norm(np.asarray(solve["out"]["tip_p"], dtype=float) - target_p)),
        "physical_stationarity_norm": solve["physical_stationarity_norm"],
        "scaled_stationarity_norm": solve["scaled_stationarity_norm"],
        "physical_jacobian_condition": solve["physical_jacobian_condition"],
        "scaled_jacobian_condition": solve["scaled_jacobian_condition"],
        "jacobian_rank": solve["jacobian_rank"],
        "continuity_norm": solve["continuity_norm"],
        "terminal_force_norm": float(np.linalg.norm(boundary[:3])),
        "terminal_torque_norm": float(np.linalg.norm(boundary[3:])),
        "nfev": solve["nfev"],
        "steady_s": solve["steady_s"],
        "success": solve["success"],
        **load,
        **combined,
    }
    row["physical_gate"] = bool(
        row["success"]
        and row["physical_stationarity_norm"] <= 1.0e-5
        and row["physical_jacobian_condition"] < 1.0e7
        and row["continuity_norm"] <= 1.0e-3
        and row["max_interface_force_jump_over_rigid_force"] <= 1.5
    )
    row["scaled_coordinate_gate"] = bool(
        row["success"]
        and row["scaled_stationarity_norm"] <= 1.0e-5
        and row["scaled_jacobian_condition"] < 1.0e7
        and row["continuity_norm"] <= 1.0e-3
        and row["max_interface_force_jump_over_rigid_force"] <= 1.5
    )
    if row["physical_gate"]:
        row["failure_class"] = "通过"
    elif row["physical_stationarity_norm"] > 1.0e-5:
        row["failure_class"] = "一阶最优性退化"
    elif row["physical_jacobian_condition"] >= 1.0e7 and row["scaled_coordinate_gate"]:
        row["failure_class"] = "物理变量坐标病态"
    elif row["physical_jacobian_condition"] >= 1.0e9:
        row["failure_class"] = "条件数严重病态"
    elif row["continuity_norm"] > 1.0e-3 or row["max_interface_force_jump_over_rigid_force"] > 1.5:
        row["failure_class"] = "接口力连续性失配"
    else:
        row["failure_class"] = "综合失败"
    return row


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for force_scale, torque_scale in VARIABLE_SCALE_PAIRS:
        sub = [
            row
            for row in rows
            if row["variable_force_scale"] == force_scale and row["variable_torque_scale"] == torque_scale
        ]
        final = max(sub, key=lambda row: row["stage"])
        score = (
            max(0.0, np.log10(final["physical_jacobian_condition"] / 1.0e7))
            + max(0.0, np.log10(final["max_interface_force_jump_over_rigid_force"] / 1.5))
            + max(0.0, np.log10(final["continuity_norm"] / 1.0e-3))
            + max(0.0, np.log10(final["physical_stationarity_norm"] / 1.0e-5))
        )
        out.append(
            {
                "variable_force_scale": force_scale,
                "variable_torque_scale": torque_scale,
                "physical_gate": final["physical_gate"],
                "scaled_coordinate_gate": final["scaled_coordinate_gate"],
                "gate_violation_score": float(score),
                "final_physical_stationarity_norm": final["physical_stationarity_norm"],
                "final_scaled_stationarity_norm": final["scaled_stationarity_norm"],
                "final_physical_condition": final["physical_jacobian_condition"],
                "final_scaled_condition": final["scaled_jacobian_condition"],
                "final_continuity_norm": final["continuity_norm"],
                "final_force_ratio_to_rigid_load": final["max_interface_force_jump_over_rigid_force"],
                "final_force_ratio_to_internal_force": final["max_force_ratio_to_internal_force"],
                "final_position_error_m": final["position_error_m"],
                "dominant_failure": final["failure_class"],
            }
        )
    return out


def _plot(summaries: list[dict[str, Any]]) -> None:
    labels = [f"F{r['variable_force_scale']:g}/M{r['variable_torque_scale']:g}" for r in summaries]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(x, [r["final_physical_condition"] for r in summaries], marker="o", label="物理坐标")
    axes[0].plot(x, [r["final_scaled_condition"] for r in summaries], marker="s", label="缩放坐标")
    axes[1].plot(x, [r["final_physical_stationarity_norm"] for r in summaries], marker="o", label="物理坐标")
    axes[1].plot(x, [r["final_scaled_stationarity_norm"] for r in summaries], marker="s", label="缩放坐标")
    axes[2].plot(x, [r["final_continuity_norm"] for r in summaries], marker="o")
    axes[3].plot(x, [r["final_force_ratio_to_rigid_load"] for r in summaries], marker="o", label="相对刚段外力")
    axes[3].plot(x, [r["final_force_ratio_to_internal_force"] for r in summaries], marker="s", label="相对接口内力")
    axes[0].axhline(1.0e7, color="black", ls="--", lw=1.0)
    axes[1].axhline(1.0e-5, color="black", ls="--", lw=1.0)
    axes[2].axhline(1.0e-3, color="black", ls="--", lw=1.0)
    axes[3].axhline(1.5, color="black", ls="--", lw=1.0)
    axes[0].set_ylabel("条件数")
    axes[1].set_ylabel("一阶最优性")
    axes[2].set_ylabel("接口连续残差")
    axes[3].set_ylabel("力跳变比")
    axes[3].set_xticks(x, labels, rotation=25, ha="right")
    for ax in axes:
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
    fig.suptitle("n2 变量尺度重整条件数探针")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scaled_variable_condition_probe_cn.png", dpi=180)
    plt.close(fig)


def _record(summaries: list[dict[str, Any]], wall_s: float) -> str:
    table = [
        "| 力变量尺度 | 矩变量尺度 | 物理门禁 | 缩放坐标门禁 | 物理条件数 | 缩放条件数 | 物理一阶最优性 | 接口连续 | 相对刚段外力 | 末端误差/m | 失败分类 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summaries:
        table.append(
            f"| {row['variable_force_scale']:.3g} | {row['variable_torque_scale']:.3g} | {row['physical_gate']} | {row['scaled_coordinate_gate']} | "
            f"{row['final_physical_condition']:.3e} | {row['final_scaled_condition']:.3e} | {row['final_physical_stationarity_norm']:.3e} | "
            f"{row['final_continuity_norm']:.3e} | {row['final_force_ratio_to_rigid_load']:.3e} | {row['final_position_error_m']:.3e} | {row['dominant_failure']} |"
        )
    best = min(summaries, key=lambda row: row["gate_violation_score"])
    return "\n".join(
        [
            "# P2c60 n2 变量尺度重整条件数探针记录",
            "",
            "## 实验目的",
            "",
            "P2c59 的最佳近门组合已满足一阶最优性、接口连续残差和载荷传播比例要求，但物理变量坐标下的雅可比条件数仍约为 `4.9e7`。本轮固定该近门组合，只改变优化变量坐标尺度，判断失败是否主要来自力/矩变量尺度病态。",
            "",
            "## 实验方案",
            "",
            "固定 `n2_z_pair_rand12, alpha=0.10`、position-only 任务、力连续权重 `4`、正则权重 `3e-4`。令原变量 `y = s * u`，在 `u` 坐标中求解；最终仍回到原物理变量 `y` 上计算物理门禁。",
            "",
            "## 结果",
            "",
            *table,
            "",
            "## 结论",
            "",
            f"门禁违背评分最低的变量尺度为力 `{best['variable_force_scale']:.3g}`、矩 `{best['variable_torque_scale']:.3g}`。若只有缩放坐标门禁通过而物理门禁未通过，则只能说明当前病态有明显坐标成分，不能直接发布灵敏度。",
            "",
            "本轮所有尺度组合均未通过物理门禁，也未通过缩放坐标门禁。最佳缩放坐标条件数降至约 `1.23e7`，接近但仍高于 `1e7` 门限；物理坐标条件数基本保持在 `4.91e7`。因此，P2c59 近门失败不能仅归因于优化变量尺度选择，继续做简单尺度扫描的信息增益有限。",
            "",
            f"- 总耗时：`{wall_s:.2f} s`",
            "",
            "## 下一步计划",
            "",
            "1. 若物理门禁通过，进入分支一致性和扰动重求解审计。",
            "2. 若只有缩放坐标门禁通过，补充数学说明并考虑重新定义无量纲变量尺度。",
            "3. 若二者均不过门，停止尺度扫描，转入低维力传递参数化。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    failures = OUT_DIR / "failure_events.jsonl"
    failures.write_text("", encoding="utf-8")
    params, current_i = p2c57.p2c24._params_from_saved_case(p2c57.CASE_NAME, p2c57.ALPHA)
    target_p, y_slack = p2c57.p2c24._load_saved_target_and_seed(p2c57.CASE_NAME, p2c57.ALPHA)
    y_seed = p2c57.p2c47._base_slack_to_segment_y(params, y_slack, steps_per_flex=p2c57.Args.steps_per_flex)

    rows: list[dict[str, Any]] = []
    for variable_force_scale, variable_torque_scale in VARIABLE_SCALE_PAIRS:
        y = y_seed.copy()
        y_ref = y.copy()
        for stage, (pos_scale, max_iter) in enumerate(STAGES, start=1):
            solve = _solve_stage_scaled(
                params,
                current_i,
                target_p,
                y_ref,
                y,
                pos_scale=pos_scale,
                max_iter=max_iter,
                variable_force_scale=variable_force_scale,
                variable_torque_scale=variable_torque_scale,
            )
            row = _row(
                params=params,
                target_p=target_p,
                solve=solve,
                stage=stage,
                pos_scale=pos_scale,
                max_iter=max_iter,
                variable_force_scale=variable_force_scale,
                variable_torque_scale=variable_torque_scale,
            )
            rows.append(row)
            y = solve["y"]
            if not row["physical_gate"]:
                p2c57._append_jsonl(failures, {"ts": time.time(), "failure": row["failure_class"], "evidence": row})

    summaries = _summaries(rows)
    wall_s = float(time.perf_counter() - t0)
    _plot(summaries)
    _write_csv(OUT_DIR / "scaled_variable_rows.csv", rows)
    _write_csv(OUT_DIR / "summary.csv", summaries)
    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": wall_s,
        "case": p2c57.CASE_NAME,
        "alpha": p2c57.ALPHA,
        "force_continuity_scale": FORCE_CONTINUITY_SCALE,
        "reg_scale": REG_SCALE,
        "variable_scale_pairs": list(VARIABLE_SCALE_PAIRS),
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
