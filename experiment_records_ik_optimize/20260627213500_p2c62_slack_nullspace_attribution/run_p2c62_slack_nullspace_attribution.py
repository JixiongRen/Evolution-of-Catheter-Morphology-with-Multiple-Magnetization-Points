"""Attribute rank loss in low-dimensional load-transfer shooting."""

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
P2C61_DIR = ROOT / "experiment_records_ik_optimize" / "20260627210200_p2c61_lowdim_load_transfer_parameterization"
for path in (ROOT, P2C61_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c61_lowdim_load_transfer_parameterization as p2c61

jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent
CASES = (
    {"label": "task_fit_rank_deficient", "position_scale": 60.0, "slack_reg_scale": 0.0},
    {"label": "regularized_condition_probe", "position_scale": 60.0, "slack_reg_scale": 1.0e-2},
)


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


def _condition_number(svals: np.ndarray) -> float:
    svals = np.asarray(svals, dtype=float)
    if not svals.size:
        return float("inf")
    if svals[-1] <= np.finfo(float).tiny:
        return float("inf")
    return float(svals[0] / svals[-1])


def _make_block_functions(params: Any, target_p: np.ndarray, *, position_scale: float, slack_reg_scale: float):
    target_p_j = jnp.asarray(target_p, dtype=jnp.float64).reshape(3)

    def output_u(u: jnp.ndarray):
        return p2c61._forward_lowdim(params, u, mode="load_scaled")

    def position_block(u: jnp.ndarray) -> jnp.ndarray:
        out = output_u(u)
        return float(position_scale) * (out["tip_p"] - target_p_j)

    def boundary_block(u: jnp.ndarray) -> jnp.ndarray:
        out = output_u(u)
        return float(p2c61.BOUNDARY_SCALE) * out["boundary"]

    def slack_block(u: jnp.ndarray) -> jnp.ndarray:
        _base, force_lam, torque_lam = p2c61._split_u(u, max(len(params.flex) - 1, 0), "load_scaled")
        return float(slack_reg_scale) * jnp.concatenate([force_lam.reshape(-1), torque_lam.reshape(-1)], axis=0)

    def full_residual(u: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([position_block(u).reshape(-1), boundary_block(u).reshape(-1), slack_block(u).reshape(-1)], axis=0)

    return {
        "output": jax.jit(output_u),
        "position": jax.jit(position_block),
        "boundary": jax.jit(boundary_block),
        "slack": jax.jit(slack_block),
        "full": jax.jit(full_residual),
        "J_position": jax.jit(jax.jacfwd(position_block)),
        "J_boundary": jax.jit(jax.jacfwd(boundary_block)),
        "J_slack": jax.jit(jax.jacfwd(slack_block)),
        "J_full": jax.jit(jax.jacfwd(full_residual)),
    }


def _analyze_case(params: Any, target_p: np.ndarray, y_seed: np.ndarray, cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    n_interfaces = max(len(params.flex) - 1, 0)
    u0 = np.concatenate([np.zeros(6, dtype=float), np.zeros(2 * n_interfaces, dtype=float)])
    funcs_solve = p2c61._make_functions(
        params,
        target_p,
        mode="load_scaled",
        position_scale=float(cfg["position_scale"]),
        slack_reg_scale=float(cfg["slack_reg_scale"]),
    )
    solve = p2c61._solve(u0, funcs_solve)
    row = p2c61._row(
        mode="load_scaled",
        init_name="zero",
        position_scale=float(cfg["position_scale"]),
        slack_reg_scale=float(cfg["slack_reg_scale"]),
        params=params,
        target_p=target_p,
        solve=solve,
    )

    funcs = _make_block_functions(
        params,
        target_p,
        position_scale=float(cfg["position_scale"]),
        slack_reg_scale=float(cfg["slack_reg_scale"]),
    )
    u = np.asarray(solve["u"], dtype=float)
    u_j = jnp.asarray(u, dtype=jnp.float64)
    r_p = np.asarray(jax.device_get(funcs["position"](u_j)), dtype=float)
    r_b = np.asarray(jax.device_get(funcs["boundary"](u_j)), dtype=float)
    r_s = np.asarray(jax.device_get(funcs["slack"](u_j)), dtype=float)
    Jp = np.asarray(jax.device_get(funcs["J_position"](u_j)), dtype=float)
    Jb = np.asarray(jax.device_get(funcs["J_boundary"](u_j)), dtype=float)
    Js = np.asarray(jax.device_get(funcs["J_slack"](u_j)), dtype=float)
    J = np.vstack([Jp, Jb, Js])
    r = np.concatenate([r_p, r_b, r_s])
    U, svals, Vt = np.linalg.svd(J, full_matrices=True)
    rank = int(np.linalg.matrix_rank(J))
    null_vec = Vt[-1]
    null_vec = null_vec / max(float(np.linalg.norm(null_vec)), 1.0e-30)
    base_fraction = float(np.linalg.norm(null_vec[:6]) / max(np.linalg.norm(null_vec), 1.0e-30))
    slack_fraction = float(np.linalg.norm(null_vec[6:]) / max(np.linalg.norm(null_vec), 1.0e-30))

    delta = 1.0e-3
    out0 = solve["out"]
    out_plus = jax.device_get(funcs["output"](jnp.asarray(u + delta * null_vec, dtype=jnp.float64)))
    out_minus = jax.device_get(funcs["output"](jnp.asarray(u - delta * null_vec, dtype=jnp.float64)))
    tip_change = max(
        float(np.linalg.norm(np.asarray(out_plus["tip_p"]) - out0["tip_p"])),
        float(np.linalg.norm(np.asarray(out_minus["tip_p"]) - out0["tip_p"])),
    )
    boundary_change = max(
        float(np.linalg.norm(np.asarray(out_plus["boundary"]) - out0["boundary"])),
        float(np.linalg.norm(np.asarray(out_minus["boundary"]) - out0["boundary"])),
    )
    slack_change = float(delta * np.linalg.norm(null_vec[6:]))

    block_rows = []
    for name, block, Jblock, rblock in [
        ("position", "末端位置", Jp, r_p),
        ("boundary", "终端边界", Jb, r_b),
        ("slack_regularization", "松弛正则", Js, r_s),
    ]:
        block_rows.append(
            {
                "label": cfg["label"],
                "block_name": name,
                "block_cn": block,
                "residual_norm": float(np.linalg.norm(rblock)),
                "jacobian_fro_norm": float(np.linalg.norm(Jblock)),
                "stationarity_contribution_norm": float(np.linalg.norm(Jblock.T @ rblock)),
                "null_projection_norm": float(np.linalg.norm(Jblock @ null_vec)),
            }
        )

    analysis = {
        **row,
        "label": cfg["label"],
        "full_residual_norm": float(np.linalg.norm(r)),
        "full_stationarity_norm_recomputed": float(np.linalg.norm(J.T @ r)),
        "singular_values": svals.tolist(),
        "rank_recomputed": rank,
        "condition_recomputed": _condition_number(svals),
        "null_base_fraction": base_fraction,
        "null_slack_fraction": slack_fraction,
        "null_tip_change_per_1e3": tip_change,
        "null_boundary_change_per_1e3": boundary_change,
        "null_slack_change_per_1e3": slack_change,
        "base_components": u[:6].tolist(),
        "slack_components": u[6:].tolist(),
        "oracle_seed_base": np.asarray(y_seed[:6], dtype=float).tolist(),
    }
    return analysis, {"block_rows": block_rows, "null_vec": null_vec.tolist()}


def _plot(analyses: list[dict[str, Any]], block_rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for row in analyses:
        svals = np.asarray(row["singular_values"], dtype=float)
        axes[0].plot(np.arange(1, len(svals) + 1), svals, marker="o", label=row["label"])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("序号")
    axes[0].set_ylabel("奇异值")
    axes[0].set_title("残差雅可比奇异值")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    labels = [row["label"] for row in analyses]
    x = np.arange(len(labels))
    axes[1].bar(x - 0.2, [row["null_base_fraction"] for row in analyses], width=0.4, label="基端载荷分量")
    axes[1].bar(x + 0.2, [row["null_slack_fraction"] for row in analyses], width=0.4, label="松弛分量")
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("零空间方向占比")
    axes[1].set_title("最弱方向组成")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "slack_nullspace_singular_values_cn.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    groups = sorted({row["label"] for row in block_rows})
    block_names = ["position", "boundary", "slack_regularization"]
    width = 0.25
    x = np.arange(len(groups))
    for i, block in enumerate(block_names):
        vals = [
            next(row["null_projection_norm"] for row in block_rows if row["label"] == label and row["block_name"] == block)
            for label in groups
        ]
        ax.bar(x + (i - 1) * width, vals, width=width, label=block)
    ax.set_yscale("log")
    ax.set_xticks(x, groups, rotation=20, ha="right")
    ax.set_ylabel("最弱方向在残差块中的投影")
    ax.set_title("零空间方向对各残差块的影响")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "slack_nullspace_block_projection_cn.png", dpi=180)
    plt.close(fig)


def _record(analyses: list[dict[str, Any]], block_rows: list[dict[str, Any]], wall_s: float) -> str:
    table = [
        "| 样本 | 末端误差/m | 一阶最优性 | 条件数 | 秩 | 相对刚段外力 | 零空间松弛占比 | 零空间末端变化/m | 失败分类 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in analyses:
        table.append(
            f"| {row['label']} | {row['position_error_m']:.3e} | {row['stationarity_norm']:.3e} | {row['jacobian_condition']:.3e} | "
            f"{row['jacobian_rank']} | {row['max_interface_force_jump_over_rigid_force']:.3e} | {row['null_slack_fraction']:.3f} | "
            f"{row['null_tip_change_per_1e3']:.3e} | {row['failure_class']} |"
        )
    block_table = [
        "| 样本 | 残差块 | 残差范数 | 雅可比范数 | 一阶贡献 | 最弱方向投影 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in block_rows:
        block_table.append(
            f"| {row['label']} | {row['block_cn']} | {row['residual_norm']:.3e} | {row['jacobian_fro_norm']:.3e} | "
            f"{row['stationarity_contribution_norm']:.3e} | {row['null_projection_norm']:.3e} |"
        )
    return "\n".join(
        [
            "# P2c62 松弛零空间归因审计记录",
            "",
            "## 实验目的",
            "",
            "P2c61 表明低维载荷定标模型可以把末端误差压得很小，但矩阵秩亏、载荷传播失败。本轮不继续扫权重，而是分析该模型的最弱方向和残差块耦合，判断末端拟合是否依赖不可辨识的松弛方向。",
            "",
            "## 实验方案",
            "",
            "固定 `n2_z_pair_rand12, alpha=0.10`。重求解两个代表样本：无松弛正则的最佳末端拟合样本，以及 `1e-2` 松弛正则样本。分别计算残差雅可比奇异值、零空间方向、零空间扰动对末端位置和边界残差的影响，并拆分末端位置、终端边界、松弛正则三个残差块。",
            "",
            "## 结果",
            "",
            *table,
            "",
            "## 残差块分解",
            "",
            *block_table,
            "",
            "## 结论",
            "",
            "若最弱方向主要由松弛变量组成，且沿该方向扰动几乎不改变末端位置和终端边界，则说明末端拟合对该方向不可辨识。此时即使末端误差很小，也不能把该解用于灵敏度发布，因为局部导数会受任意松弛幅值选择影响。",
            "",
            f"- 总耗时：`{wall_s:.2f} s`",
            "",
            "## 下一步计划",
            "",
            "1. 若零空间确认为松弛主导，停止该低维松弛模型的灵敏度发布路线。",
            "2. 后续应转向显式物理残差耦合，或将松弛幅值作为被物理方程确定的状态而非自由优化变量。",
            "3. 对两点工况形成阶段性不可发布边界：末端拟合可达，但局部灵敏度受不可辨识松弛方向污染。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    failures = OUT_DIR / "failure_events.jsonl"
    failures.write_text("", encoding="utf-8")
    params, _current_i = p2c61.p2c57.p2c24._params_from_saved_case(p2c61.p2c57.CASE_NAME, p2c61.p2c57.ALPHA)
    target_p, y_slack = p2c61.p2c57.p2c24._load_saved_target_and_seed(p2c61.p2c57.CASE_NAME, p2c61.p2c57.ALPHA)
    y_seed = p2c61.p2c57.p2c47._base_slack_to_segment_y(params, y_slack, steps_per_flex=p2c61.STEPS_PER_FLEX)
    analyses: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    null_vectors: dict[str, Any] = {}
    for cfg in CASES:
        analysis, extra = _analyze_case(params, target_p, y_seed, cfg)
        analyses.append(analysis)
        block_rows.extend(extra["block_rows"])
        null_vectors[cfg["label"]] = extra["null_vec"]
        if not analysis["baseline_gate"]:
            _append_jsonl(failures, {"ts": time.time(), "failure": analysis["failure_class"], "evidence": analysis})
    wall_s = float(time.perf_counter() - t0)
    _plot(analyses, block_rows)
    _write_csv(OUT_DIR / "nullspace_analysis_rows.csv", analyses)
    _write_csv(OUT_DIR / "residual_block_rows.csv", block_rows)
    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": wall_s,
        "case": p2c61.p2c57.CASE_NAME,
        "alpha": p2c61.p2c57.ALPHA,
        "analyses": analyses,
        "block_rows": block_rows,
        "null_vectors": null_vectors,
        "git": {
            "branch": _run(["git", "branch", "--show-current"]),
            "commit": _run(["git", "rev-parse", "HEAD"]),
            "status_short": _run(["git", "status", "--short"]),
        },
        "jax_devices": [str(device) for device in jax.devices()],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "record.md").write_text(_record(analyses, block_rows, wall_s), encoding="utf-8")
    print(json.dumps(_jsonable({"wall_s": wall_s, "analyses": analyses}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
