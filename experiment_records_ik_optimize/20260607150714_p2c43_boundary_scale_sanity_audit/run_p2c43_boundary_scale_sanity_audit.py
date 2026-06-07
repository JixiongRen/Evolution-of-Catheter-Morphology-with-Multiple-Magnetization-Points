"""Audit whether boundary-dominated stationarity is caused by residual scaling."""

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
P2C42_DIR = ROOT / "experiment_records_ik_optimize" / "20260607145635_p2c42_stationarity_block_attribution"
for path in (ROOT, FK_OPT_DIR, P2C24_DIR, P2C41_DIR, P2C42_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c24_interface_weight_continuation_repair as p2c24
import run_p2c41_scaled_interface_slack_baseline_gate as p2c41
import run_p2c42_stationarity_block_attribution as p2c42

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
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


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


def _parse_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _classify(row: dict[str, Any]) -> tuple[str, str]:
    if row["baseline_gate"]:
        return "通过", "满足所有基线门禁"
    if row["stationarity_norm"] > 1.0e-5 and row["max_slack_force_over_rigid_force"] <= 1.5:
        return "边界闭合病态", "载荷传播已受控，但一阶最优性未闭合"
    if row["max_slack_force_over_rigid_force"] > 2.0:
        return "载荷传播失配", "界面松弛仍超过刚段外力两倍"
    if row["jacobian_condition"] >= 1.0e9:
        return "条件数病态", "局部线性系统严重病态"
    return "综合失败", "多个门禁同时失败"


def _analyze_blocks(
    params: Any,
    I0: np.ndarray,
    target_p: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    *,
    boundary_scale: float,
) -> tuple[str, dict[str, float]]:
    block_args = argparse.Namespace(**vars(args))
    block_args.boundary_scale = float(boundary_scale)
    funcs = p2c42._block_functions(params, target_p, block_args)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    I_j = jnp.asarray(I0, dtype=jnp.float64)
    gradients = {}
    shares = {}
    total = 0.0
    for block in ["position", "boundary", "slack"]:
        r = np.asarray(jax.device_get(funcs[block](y_j, I_j)), dtype=float).reshape(-1)
        J = np.asarray(jax.device_get(funcs[f"J_{block}"](y_j, I_j)), dtype=float)
        g = J.T @ r
        gradients[block] = g
        shares[block] = float(np.linalg.norm(g))
        total += shares[block]
    for key in shares:
        shares[key] /= max(total, 1.0e-30)
    dominant = max(shares, key=shares.get)
    shares["position_boundary_cosine"] = p2c42._safe_cosine(gradients["position"], gradients["boundary"])
    shares["boundary_slack_cosine"] = p2c42._safe_cosine(gradients["boundary"], gradients["slack"])
    return dominant, shares


def _run_one(case_name: str, alpha: float, boundary_scale: float, args: argparse.Namespace) -> dict[str, Any]:
    params, I0 = p2c24._params_from_saved_case(case_name, alpha)
    target_p, y_ref = p2c24._load_saved_target_and_seed(case_name, alpha)
    target_p = np.asarray(target_p, dtype=float)
    y = p2c24._seed_from_reference(np.asarray(y_ref, dtype=float), eta=1.0, base_scale=1.0)
    funcs = p2c41._make_functions(params, target_p, args, mode="load_scaled")
    solve = p2c41._solve(y, I0, funcs, boundary_scale=boundary_scale, slack_scale=args.slack_scale, max_iter=args.max_iter)
    load = p2c41._load_audit(params, solve["y"], I0)
    position_error = float(np.linalg.norm(np.asarray(solve["out"]["tip_p"], dtype=float) - target_p))
    dominant, shares = _analyze_blocks(params, I0, target_p, solve["y"], args, boundary_scale=boundary_scale)
    row = {
        "case": case_name,
        "alpha": alpha,
        "boundary_scale": boundary_scale,
        "slack_scale": args.slack_scale,
        "success": solve["success"],
        "nfev": solve["nfev"],
        "steady_s": solve["steady_s"],
        "position_error_m": position_error,
        "stationarity_norm": solve["stationarity_norm"],
        "jacobian_condition": solve["jacobian_condition"],
        "boundary_norm": solve["boundary_norm"],
        "max_slack_force_over_rigid_force": load["max_slack_force_over_rigid_force"],
        "max_slack_torque_over_rigid_torque": load["max_slack_torque_over_rigid_torque"],
        "dominant_gradient_block": dominant,
        "position_gradient_share": shares["position"],
        "boundary_gradient_share": shares["boundary"],
        "slack_gradient_share": shares["slack"],
        "position_boundary_cosine": shares["position_boundary_cosine"],
        "boundary_slack_cosine": shares["boundary_slack_cosine"],
    }
    row["baseline_gate"] = bool(
        row["success"]
        and row["position_error_m"] <= 0.05 * 0.11
        and row["stationarity_norm"] <= 1.0e-5
        and row["jacobian_condition"] < 1.0e7
        and row["max_slack_force_over_rigid_force"] <= 1.5
    )
    row["failure_class"], row["failure_reason"] = _classify(row)
    return row


def _case_summary(case_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = min(rows, key=lambda row: (not row["baseline_gate"], row["stationarity_norm"], row["max_slack_force_over_rigid_force"]))
    best_load = min(rows, key=lambda row: row["max_slack_force_over_rigid_force"])
    best_cond = min(rows, key=lambda row: row["jacobian_condition"])
    return {
        "case": case_name,
        "best_boundary_scale": best["boundary_scale"],
        "best_failure_class": best["failure_class"],
        "best_stationarity_norm": best["stationarity_norm"],
        "best_jacobian_condition": best["jacobian_condition"],
        "best_slack_force_ratio": best["max_slack_force_over_rigid_force"],
        "best_position_error_m": best["position_error_m"],
        "best_dominant_gradient_block": best["dominant_gradient_block"],
        "min_slack_force_ratio": best_load["max_slack_force_over_rigid_force"],
        "min_condition": best_cond["jacobian_condition"],
        "any_gate_pass": any(row["baseline_gate"] for row in rows),
        "boundary_dominant_count": sum(1 for row in rows if row["dominant_gradient_block"] == "boundary"),
        "row_count": len(rows),
    }


def _plot(rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    cases = sorted({row["case"] for row in rows})
    for case in cases:
        sub = sorted([row for row in rows if row["case"] == case], key=lambda row: row["boundary_scale"])
        xs = [row["boundary_scale"] for row in sub]
        axes[0].plot(xs, [row["stationarity_norm"] for row in sub], marker="o", label=case)
        axes[1].plot(xs, [row["max_slack_force_over_rigid_force"] for row in sub], marker="o", label=case)
        axes[2].plot(xs, [row["jacobian_condition"] for row in sub], marker="o", label=case)
    axes[0].axhline(1.0e-5, color="black", ls="--", lw=1)
    axes[1].axhline(1.5, color="black", ls="--", lw=1)
    axes[2].axhline(1.0e7, color="black", ls="--", lw=1)
    axes[0].set_ylabel("一阶最优性残差")
    axes[1].set_ylabel("载荷传播比")
    axes[2].set_ylabel("条件数")
    axes[2].set_xlabel("边界平衡权重")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("P2c43 边界权重尺度复核")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "p2c43_boundary_scale_audit_cn.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [summary["case"] for summary in summaries]
    x = np.arange(len(labels))
    ax.bar(x, [summary["boundary_dominant_count"] / max(summary["row_count"], 1) for summary in summaries], color="#f97316")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("边界平衡主导比例")
    ax.set_title("不同边界权重下的主导失败来源")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "p2c43_boundary_dominance_cn.png", dpi=180)
    plt.close(fig)


def _record(args: argparse.Namespace, summaries: list[dict[str, Any]]) -> str:
    table = [
        "| 工况 | 是否有通过点 | 最优边界权重 | 最优一阶最优性 | 最优载荷传播比 | 边界主导比例 | 结论 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for summary in summaries:
        if summary["any_gate_pass"]:
            conclusion = "尺度调整可修复，允许进入扰动重求解。"
        elif summary["boundary_dominant_count"] >= max(2, summary["row_count"] // 2):
            conclusion = "边界闭合为稳定主因，不是单点权重误设。"
        else:
            conclusion = "失败来源随权重变化，需要继续分块诊断。"
        table.append(
            f"| {summary['case']} | {summary['any_gate_pass']} | {summary['best_boundary_scale']:.3g} | "
            f"{summary['best_stationarity_norm']:.3e} | {summary['best_slack_force_ratio']:.3e} | "
            f"{summary['boundary_dominant_count']}/{summary['row_count']} | {conclusion} |"
        )
    return "\n".join(
        [
            "# P2c43 边界平衡尺度复核记录",
            "",
            "## 实验目的",
            "",
            "P2c42 显示单点与两点工况的一阶最优性失败由边界平衡主导。本轮实验扫描边界平衡权重，判断该结论是否只是残差尺度设置造成的假象，还是稳定的物理闭合缺口。",
            "",
            "## 实验方案",
            "",
            "使用载荷定标接口松弛模型，只推进三维末端位置任务。对每个工况扫描边界平衡权重，固定接口松弛权重，并在每个权重下重新求解。每个求解结果均记录一阶最优性、条件数、载荷传播比、末端位置误差和主导一阶最优性残差块。",
            "",
            "## 参数",
            "",
            f"- 工况：`{args.specs}`",
            f"- 边界平衡权重：`{args.boundary_scales}`",
            f"- 接口松弛权重：`{args.slack_scale}`",
            f"- 最大函数调用次数：`{args.max_iter}`",
            "",
            "## 结果",
            "",
            *table,
            "",
            "## 结论",
            "",
            "若扫描中不存在通过点，且边界平衡在多数权重下仍主导一阶最优性，则说明失败不是单纯权重尺度误设，而是末端内力内矩闭合与当前任务/接口参数化之间存在结构性不相容。",
            "",
            "## 下一步计划",
            "",
            "1. 若边界平衡持续主导，则进入显式末端内力内矩闭合模型，不再继续扫接口松弛权重。",
            "2. 若某个权重下门禁通过，则对该点执行 eps=0.005 与 0.010 的扰动重求解。",
            "3. 若单点仍不通过，应转向有效子空间任务；若两点仍不通过，应优先审查末端边界条件和刚性磁体外载耦合。",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", default="n1_tip_z_rand15:0.10,n2_z_pair_rand12:0.10")
    parser.add_argument("--boundary_scales", default="0.02,0.05,0.10,0.25,0.50,1.00")
    parser.add_argument("--slack_scale", type=float, default=2.0)
    parser.add_argument("--max_iter", type=int, default=120)
    parser.add_argument("--steps_per_flex", type=int, default=4)
    parser.add_argument("--position_scale", type=float, default=60.0)
    parser.add_argument("--load_floor", type=float, default=0.05)
    args = parser.parse_args()

    failures = OUT_DIR / "failure_events.jsonl"
    failures.write_text("", encoding="utf-8")
    t0 = time.perf_counter()
    rows = []
    summaries = []
    for case_name, alpha in _parse_specs(args.specs):
        case_rows = []
        for boundary_scale in _parse_floats(args.boundary_scales):
            row = _run_one(case_name, alpha, boundary_scale, args)
            rows.append(row)
            case_rows.append(row)
            if not row["baseline_gate"]:
                _append_jsonl(failures, {"ts": time.time(), "case": case_name, "boundary_scale": boundary_scale, "failure": row["failure_class"], "evidence": row})
        summaries.append(_case_summary(case_name, case_rows))
    _plot(rows, summaries)
    _write_csv(OUT_DIR / "boundary_scale_rows.csv", rows)
    _write_csv(OUT_DIR / "summary.csv", summaries)
    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": float(time.perf_counter() - t0),
        "args": vars(args),
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
    (OUT_DIR / "record.md").write_text(_record(args, summaries), encoding="utf-8")
    (OUT_DIR / "commands.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points\n"
        "XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n evolution_of_mcrs python "
        "experiment_records_ik_optimize/20260607150714_p2c43_boundary_scale_sanity_audit/"
        "run_p2c43_boundary_scale_sanity_audit.py\n",
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
