"""Attribute stationarity failures by residual block for scaled interface slack."""

from __future__ import annotations

import argparse
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
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"
P2C24_DIR = ROOT / "experiment_records_ik_optimize" / "20260606160747_p2c24_interface_weight_continuation_repair"
P2C41_DIR = ROOT / "experiment_records_ik_optimize" / "20260607144044_p2c41_scaled_interface_slack_baseline_gate"
for path in (ROOT, FK_OPT_DIR, P2C24_DIR, P2C41_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk
from forward_kinematics_optimized.equilibrium_solver_nondim import with_coil_currents

import run_p2c24_interface_weight_continuation_repair as p2c24
import run_p2c41_scaled_interface_slack_baseline_gate as p2c41

jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent


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
        writer.writerows([{field: row.get(field) for field in fields} for row in rows])


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def _parse_specs(text: str) -> list[tuple[str, float]]:
    specs = []
    for raw in text.split(","):
        raw = raw.strip()
        if raw:
            case_name, alpha = raw.split(":", 1)
            specs.append((case_name.strip(), float(alpha)))
    return specs


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1.0e-30 or nb <= 1.0e-30:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _block_functions(params_base: Any, target_p: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    target_p_j = jnp.asarray(target_p, dtype=jnp.float64).reshape(3)

    def params_for_I(I: jnp.ndarray) -> Any:
        return with_coil_currents(params_base, jnp.asarray(I, dtype=jnp.float64).reshape(8))

    def position_block(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        out = p2c41.p2c10t.forward_interface_slack(params_for_I(I), y, variant="wrench", steps_per_flex=args.steps_per_flex)
        return float(args.position_scale) * (out.tip_p - target_p_j)

    def boundary_block(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        out = p2c41.p2c10t.forward_interface_slack(params_for_I(I), y, variant="wrench", steps_per_flex=args.steps_per_flex)
        return float(args.boundary_scale) * out.boundary

    def slack_block(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        params = params_for_I(I)
        return float(args.slack_scale) * p2c41.load_scaled_slack_residual(
            params,
            y,
            variant="wrench",
            load_floor=args.load_floor,
            steps_per_flex=args.steps_per_flex,
        )

    def full_residual(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([position_block(y, I).reshape(-1), boundary_block(y, I).reshape(-1), slack_block(y, I).reshape(-1)])

    return {
        "position": jax.jit(position_block),
        "boundary": jax.jit(boundary_block),
        "slack": jax.jit(slack_block),
        "full": jax.jit(full_residual),
        "J_position": jax.jit(jax.jacfwd(position_block, argnums=0)),
        "J_boundary": jax.jit(jax.jacfwd(boundary_block, argnums=0)),
        "J_slack": jax.jit(jax.jacfwd(slack_block, argnums=0)),
        "J_full": jax.jit(jax.jacfwd(full_residual, argnums=0)),
    }


def _solve_final(case_name: str, alpha: float, args: argparse.Namespace) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any]]:
    params, I0 = p2c24._params_from_saved_case(case_name, alpha)
    target_p, y_ref = p2c24._load_saved_target_and_seed(case_name, alpha)
    target_p = np.asarray(target_p, dtype=float)
    y = p2c24._seed_from_reference(np.asarray(y_ref, dtype=float), eta=1.0, base_scale=1.0)
    funcs = p2c41._make_functions(params, target_p, args, mode="load_scaled")
    for _stage, boundary_scale, slack_scale in p2c41._parse_schedule(args.schedule):
        solve = p2c41._solve(y, I0, funcs, boundary_scale=boundary_scale, slack_scale=slack_scale, max_iter=args.max_iter)
        y = solve["y"]
    return params, I0, target_p, solve


def _analyze_case(case_name: str, alpha: float, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    params, I0, target_p, solve = _solve_final(case_name, alpha, args)
    funcs = _block_functions(params, target_p, args)
    y_j = jnp.asarray(solve["y"], dtype=jnp.float64)
    I_j = jnp.asarray(I0, dtype=jnp.float64)
    block_rows = []
    gradient_rows = []
    gradients: dict[str, np.ndarray] = {}
    residual_norm_sum = 0.0
    gradient_norm_sum = 0.0

    for name in ["position", "boundary", "slack"]:
        r = np.asarray(jax.device_get(funcs[name](y_j, I_j)), dtype=float).reshape(-1)
        J = np.asarray(jax.device_get(funcs[f"J_{name}"](y_j, I_j)), dtype=float)
        g = J.T @ r
        gradients[name] = g
        residual_norm_sum += float(np.linalg.norm(r))
        gradient_norm_sum += float(np.linalg.norm(g))
        svals = np.linalg.svd(J, compute_uv=False) if J.size else np.asarray([], dtype=float)
        block_rows.append(
            {
                "case": case_name,
                "alpha": alpha,
                "block": name,
                "residual_norm": float(np.linalg.norm(r)),
                "gradient_norm": float(np.linalg.norm(g)),
                "jacobian_rank": int(np.linalg.matrix_rank(J)) if J.size else 0,
                "jacobian_condition": float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny)) if svals.size else float("nan"),
                "jacobian_col_norm_max": float(np.max(np.linalg.norm(J, axis=0))) if J.size else 0.0,
                "jacobian_col_norm_median": float(np.median(np.linalg.norm(J, axis=0))) if J.size else 0.0,
            }
        )

    for row in block_rows:
        row["residual_norm_share"] = row["residual_norm"] / max(residual_norm_sum, 1.0e-30)
        row["gradient_norm_share"] = row["gradient_norm"] / max(gradient_norm_sum, 1.0e-30)

    for a, b in [("position", "boundary"), ("position", "slack"), ("boundary", "slack")]:
        gradient_rows.append(
            {
                "case": case_name,
                "alpha": alpha,
                "block_a": a,
                "block_b": b,
                "gradient_cosine": _safe_cosine(gradients[a], gradients[b]),
                "gradient_sum_norm": float(np.linalg.norm(gradients[a] + gradients[b])),
                "gradient_diff_norm": float(np.linalg.norm(gradients[a] - gradients[b])),
            }
        )

    full_r = np.asarray(jax.device_get(funcs["full"](y_j, I_j)), dtype=float)
    full_J = np.asarray(jax.device_get(funcs["J_full"](y_j, I_j)), dtype=float)
    full_g = full_J.T @ full_r
    full_svals = np.linalg.svd(full_J, compute_uv=False)
    load = p2c41._load_audit(params, solve["y"], I0)
    position_error = float(np.linalg.norm(np.asarray(solve["out"]["tip_p"], dtype=float) - target_p))
    dominant_gradient = max(block_rows, key=lambda row: row["gradient_norm_share"])["block"]
    dominant_residual = max(block_rows, key=lambda row: row["residual_norm_share"])["block"]
    summary = {
        "case": case_name,
        "alpha": alpha,
        "position_error_m": position_error,
        "stationarity_norm": float(np.linalg.norm(full_g)),
        "jacobian_condition": float(full_svals[0] / max(float(full_svals[-1]), np.finfo(float).tiny)),
        "max_slack_force_over_rigid_force": load["max_slack_force_over_rigid_force"],
        "max_slack_torque_over_rigid_torque": load["max_slack_torque_over_rigid_torque"],
        "boundary_norm": float(np.linalg.norm(np.asarray(solve["out"]["boundary"], dtype=float))),
        "dominant_gradient_block": dominant_gradient,
        "dominant_residual_block": dominant_residual,
        "steady_s": solve["steady_s"],
        "nfev": solve["nfev"],
    }
    return summary, block_rows, gradient_rows


def _plot(block_rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    block_cn = {"position": "末端位置", "boundary": "边界平衡", "slack": "接口松弛"}
    cases = [row["case"] for row in summaries]
    blocks = ["position", "boundary", "slack"]
    x = np.arange(len(cases))
    width = 0.24
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for i, block in enumerate(blocks):
        vals = [next(row for row in block_rows if row["case"] == case and row["block"] == block)["gradient_norm_share"] for case in cases]
        axes[0].bar(x + (i - 1) * width, vals, width=width, label=block_cn[block])
        vals2 = [next(row for row in block_rows if row["case"] == case and row["block"] == block)["residual_norm_share"] for case in cases]
        axes[1].bar(x + (i - 1) * width, vals2, width=width, label=block_cn[block])
    axes[0].set_ylabel("一阶最优性贡献占比")
    axes[1].set_ylabel("残差范数占比")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cases, rotation=10, ha="right")
    for ax in axes:
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.suptitle("P2c42 残差分块归因")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "p2c42_block_attribution_cn.png", dpi=180)
    plt.close(fig)


def _record(args: argparse.Namespace, summaries: list[dict[str, Any]]) -> str:
    table = [
        "| 工况 | 主导一阶最优性块 | 主导残差块 | 一阶最优性 | 条件数 | 载荷传播比 | 结论 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    block_cn = {"position": "末端位置", "boundary": "边界平衡", "slack": "接口松弛"}
    for row in summaries:
        if row["dominant_gradient_block"] == "position":
            conclusion = "末端任务与物理闭合竞争，需要降低任务压制或做任务分层。"
        elif row["dominant_gradient_block"] == "boundary":
            conclusion = "边界平衡主导病态，需要显式检查末端内力内矩闭合。"
        else:
            conclusion = "接口松弛主导病态，需要进一步约束或重参数化载荷链。"
        table.append(
            f"| {row['case']} | {block_cn[row['dominant_gradient_block']]} | "
            f"{block_cn[row['dominant_residual_block']]} | {row['stationarity_norm']:.3e} | "
            f"{row['jacobian_condition']:.3e} | {row['max_slack_force_over_rigid_force']:.3e} | {conclusion} |"
        )
    return "\n".join(
        [
            "# P2c42 一阶最优性残差分块归因记录",
            "",
            "## 实验目的",
            "",
            "P2c41 表明载荷定标松弛可以降低载荷传播比例，但单点与两点工况仍未通过基线门禁。本轮实验将总残差拆分为末端位置、边界平衡和接口松弛三类，分别计算各块对一阶最优性方程的贡献，用于判断下一步应修正任务权重、边界闭合，还是接口载荷链。",
            "",
            "## 参数",
            "",
            f"- 工况：`{args.specs}`",
            f"- 使用模型：载荷定标接口松弛",
            f"- 阶段调度：`{args.schedule}`",
            f"- 最终边界权重：`{args.boundary_scale}`",
            f"- 最终接口松弛权重：`{args.slack_scale}`",
            "",
            "## 结果",
            "",
            *table,
            "",
            "## 结论",
            "",
            "本轮没有得到可发布灵敏度。其价值在于给出下一步模型修复方向：若主导项是接口松弛，则当前松弛变量仍在影响物理载荷链；若主导项是边界平衡，则应转向显式末端内力内矩闭合；若主导项是末端位置，则说明目标任务和物理平衡不可同时满足，需要缩小任务空间或重新选择工作点。",
            "",
            "## 下一步计划",
            "",
            "1. 对主导块做残差尺度复核，排除单纯数值尺度误判。",
            "2. 若两点磁控主要由边界平衡或末端任务主导，则实现显式刚柔段载荷耦合模型。",
            "3. 若单点磁控仍由接口松弛主导，则判定当前接口自由度参数化不可发布，转向有效子空间或更低维任务。",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", default="n1_tip_z_rand15:0.10,n2_z_pair_rand12:0.10")
    parser.add_argument("--schedule", default="loose:0.10:0.50;mid:0.25:1.00;strict:0.50:2.00")
    parser.add_argument("--boundary_scale", type=float, default=0.50)
    parser.add_argument("--slack_scale", type=float, default=2.00)
    parser.add_argument("--max_iter", type=int, default=120)
    parser.add_argument("--steps_per_flex", type=int, default=4)
    parser.add_argument("--position_scale", type=float, default=60.0)
    parser.add_argument("--load_floor", type=float, default=0.05)
    args = parser.parse_args()

    failures = OUT_DIR / "failure_events.jsonl"
    failures.write_text("", encoding="utf-8")
    t0 = time.perf_counter()
    summaries: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    for case_name, alpha in _parse_specs(args.specs):
        summary, blocks, gradients = _analyze_case(case_name, alpha, args)
        summaries.append(summary)
        block_rows.extend(blocks)
        gradient_rows.extend(gradients)
        _append_jsonl(
            failures,
            {
                "ts": time.time(),
                "case": case_name,
                "failure": "baseline_stationarity_block_attribution",
                "dominant_gradient_block": summary["dominant_gradient_block"],
                "evidence": summary,
            },
        )
    _plot(block_rows, summaries)
    _write_csv(OUT_DIR / "block_attribution.csv", block_rows)
    _write_csv(OUT_DIR / "gradient_coupling.csv", gradient_rows)
    _write_csv(OUT_DIR / "summary.csv", summaries)
    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": float(time.perf_counter() - t0),
        "args": vars(args),
        "summaries": summaries,
        "block_rows": block_rows,
        "gradient_rows": gradient_rows,
        "git": {
            "branch": _run(["git", "branch", "--show-current"]),
            "commit": _run(["git", "rev-parse", "HEAD"]),
            "status_short": _run(["git", "status", "--short"]),
        },
        "jax_devices": [str(device) for device in jax.devices()],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "record.md").write_text(_record(args, summaries), encoding="utf-8")
    (OUT_DIR / "commands.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points\n"
        "XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n evolution_of_mcrs python "
        "experiment_records_ik_optimize/20260607145635_p2c42_stationarity_block_attribution/"
        "run_p2c42_stationarity_block_attribution.py\n",
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
