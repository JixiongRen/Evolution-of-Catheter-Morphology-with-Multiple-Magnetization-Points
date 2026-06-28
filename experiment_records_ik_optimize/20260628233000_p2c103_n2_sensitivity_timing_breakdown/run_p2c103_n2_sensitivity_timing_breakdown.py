"""Profile fixed and per-direction costs in the n2 sensitivity audit."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
P2C79_DIR = ROOT / "experiment_records_ik_optimize" / "20260628020500_p2c79_branch_random_validation"
P2C98_DIR = ROOT / "experiment_records_ik_optimize" / "20260628213500_p2c98_n2_eps0125_random_current_batch"
P2C102_DIR = ROOT / "experiment_records_ik_optimize" / "20260628230500_p2c102_n2_sentinel_audit_efficiency"
for path in (ROOT, P2C79_DIR, P2C98_DIR, P2C102_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import run_p2c79_branch_random_validation as p2c79
import run_p2c98_n2_eps0125_random_current_batch as p2c98


OUT_DIR = Path(__file__).resolve().parent
EPS = 0.0125
P2C102_SUMMARY = P2C102_DIR / "summary.json"


def _timed(name: str, fn):
    """Run a callable and return its output with elapsed wall time."""
    t0 = time.perf_counter()
    out = fn()
    return out, {"stage": name, "wall_s": float(time.perf_counter() - t0)}


def _safe_rel(pred: np.ndarray, fd: np.ndarray) -> tuple[float, float, float]:
    """Compute relative error, cosine, and finite-difference norm."""
    return p2c79._safe_rel(np.asarray(pred, dtype=float), np.asarray(fd, dtype=float))


def _audit_direction_timed(branch: dict, I0: np.ndarray, name: str, direction: np.ndarray) -> tuple[dict, list[dict]]:
    """Audit one current direction and report component timings."""
    funcs = branch["funcs"]
    basis = branch["basis"]
    y = branch["y"]
    timings: list[dict] = []
    t0 = time.perf_counter()
    q_plus, plus_info = p2c79.p2c75._solve_q(y_base=y, basis=basis, I=I0 + EPS * direction, funcs=funcs)
    timings.append({"direction": name, "stage": "plus_reoptimization", "wall_s": plus_info["steady_s"], "nfev": plus_info["nfev"]})
    q_minus, minus_info = p2c79.p2c75._solve_q(y_base=y, basis=basis, I=I0 - EPS * direction, funcs=funcs)
    timings.append({"direction": name, "stage": "minus_reoptimization", "wall_s": minus_info["steady_s"], "nfev": minus_info["nfev"]})
    y_plus = y + basis @ q_plus
    y_minus = y + basis @ q_minus
    t_eval = time.perf_counter()
    out_plus = funcs["output"](jnp.asarray(y_plus, dtype=jnp.float64), jnp.asarray(I0 + EPS * direction, dtype=jnp.float64))
    out_minus = funcs["output"](jnp.asarray(y_minus, dtype=jnp.float64), jnp.asarray(I0 - EPS * direction, dtype=jnp.float64))
    fd = (np.asarray(jax.device_get(out_plus.tip_p), dtype=float) - np.asarray(jax.device_get(out_minus.tip_p), dtype=float)) / (2.0 * EPS)
    pred = branch["S"] @ direction
    rel, cosine, fd_norm = _safe_rel(pred, fd)
    timings.append({"direction": name, "stage": "output_and_compare", "wall_s": float(time.perf_counter() - t_eval), "nfev": 0})
    row = {
        "direction": name,
        "status": "publishable" if rel < p2c79.TASK_GATE and plus_info["success"] and minus_info["success"] else "diagnostic_only",
        "failure_class": "PASS" if rel < p2c79.TASK_GATE and plus_info["success"] and minus_info["success"] else "哨兵方向有限差分不一致",
        "vector_rel_err": rel,
        "vector_cosine": cosine,
        "fd_norm": fd_norm,
        "pred_norm": float(np.linalg.norm(pred)),
        "plus_nfev": plus_info["nfev"],
        "minus_nfev": minus_info["nfev"],
        "plus_success": plus_info["success"],
        "minus_success": minus_info["success"],
        "direction_total_wall_s": float(time.perf_counter() - t0),
    }
    return row, timings


def _plot(stage_rows: list[dict]) -> None:
    """Plot timing components."""
    labels = [row["stage"] for row in stage_rows]
    values = [row["wall_s"] for row in stage_rows]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(labels, values, color="#457b9d")
    ax.set_ylabel("耗时/s")
    ax.set_title("P2c103 n2 灵敏度审计耗时分解")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_sensitivity_timing_breakdown_cn.png", dpi=180)
    plt.close(fig)


def _record(stage_rows: list[dict], audit_rows: list[dict], total_wall_s: float, p2c102_wall_s: float) -> str:
    """Render the Chinese timing report."""
    stage_table = [
        "| 阶段 | 耗时/s | 占比 |",
        "| --- | ---: | ---: |",
    ]
    for row in stage_rows:
        stage_table.append(f"| {row['stage']} | {row['wall_s']:.2f} | {row['wall_s'] / total_wall_s:.3f} |")
    audit_table = [
        "| 方向 | 状态 | 总耗时/s | 正向耗时/s | 负向耗时/s | 相对误差 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in audit_rows:
        audit_table.append(
            f"| {row['direction']} | {row['status']} | {row['direction_total_wall_s']:.2f} | "
            f"{row['plus_wall_s']:.2f} | {row['minus_wall_s']:.2f} | {row['vector_rel_err']:.3e} |"
        )
    fixed_cost = next(row["wall_s"] for row in stage_rows if row["stage"] == "recover_trusted_branch")
    direction_cost = sum(row["direction_total_wall_s"] for row in audit_rows)
    return "\n".join(
        [
            "# P2c103 n2 灵敏度审计耗时分解记录",
            "",
            "## 实验目的",
            "",
            "P2c102 显示方向数从 20 降到 2 后，墙钟耗时没有下降。本轮对同一哨兵方向审计做分解计时，判断耗时主要来自受信任分支恢复、线性化系统构建，还是单方向正负扰动重求解。",
            "",
            "## 阶段耗时",
            "",
            *stage_table,
            "",
            "## 方向耗时",
            "",
            *audit_table,
            "",
            "## 结论",
            "",
            f"本轮总耗时 `{total_wall_s:.2f} s`，与 P2c102 的 `{p2c102_wall_s:.2f} s` 接近。受信任分支恢复耗时 `{fixed_cost:.2f} s`，两个哨兵方向合计重求解耗时约 `{direction_cost:.2f} s`。",
            "",
            "若受信任分支恢复占主导，则下一步应缓存 recovered branch、`basis`、任务灵敏度矩阵和编译后的残差函数；若方向重求解占主导，则才应继续减少方向数或并行化方向。当前数据用于决定后续效率优化路径。",
            "",
            "该实验不改变任何可信性门禁，也不扩大 n2 的发布范围。",
            "",
            "## 下一步计划",
            "",
            "1. 若固定成本占主导，开发缓存受信任分支的复核入口。",
            "2. 若方向成本占主导，开发方向批量化或并行化复核入口。",
            "3. 所有效率优化必须复跑 P2c98 或哨兵方向门禁，不能降低线性化误差门限。",
            "",
        ]
    )


def main() -> None:
    p2c79.OUT_DIR = OUT_DIR
    p2c79.EPS = EPS
    p2c102_summary = json.loads(P2C102_SUMMARY.read_text(encoding="utf-8"))
    sentinel_names = list(p2c102_summary["sentinel_names"])
    direction_map = dict(p2c98._directions())
    total_t0 = time.perf_counter()
    stage_rows: list[dict] = []

    (params, I0), row = _timed(
        "load_case",
        lambda: p2c79.p2c57.p2c24._params_from_saved_case(p2c79.p2c57.CASE_NAME, p2c79.p2c57.ALPHA),
    )
    stage_rows.append(row)
    (target_p, y_slack), row = _timed(
        "load_seed",
        lambda: p2c79.p2c57.p2c24._load_saved_target_and_seed(p2c79.p2c57.CASE_NAME, p2c79.p2c57.ALPHA),
    )
    stage_rows.append(row)
    y_seed, row = _timed(
        "convert_seed",
        lambda: p2c79.p2c57.p2c47._base_slack_to_segment_y(params, y_slack, steps_per_flex=p2c79.p2c57.Args.steps_per_flex),
    )
    stage_rows.append(row)
    branch, row = _timed("recover_trusted_branch", lambda: p2c79._recover_branch(params, I0, target_p, y_seed, "trusted_seed"))
    stage_rows.append(row)
    branch_row, row = _timed("baseline_gate_row", lambda: p2c79._branch_row(branch, branch))
    stage_rows.append(row)

    audit_rows = []
    timing_rows = []
    for name in sentinel_names:
        row, timings = _audit_direction_timed(branch, I0, name, direction_map[name])
        row["plus_wall_s"] = next(item["wall_s"] for item in timings if item["stage"] == "plus_reoptimization")
        row["minus_wall_s"] = next(item["wall_s"] for item in timings if item["stage"] == "minus_reoptimization")
        audit_rows.append(row)
        timing_rows.extend(timings)
    total_wall_s = float(time.perf_counter() - total_t0)
    p2c79._write_csv(OUT_DIR / "stage_timing_rows.csv", stage_rows)
    p2c79._write_csv(OUT_DIR / "direction_timing_rows.csv", timing_rows)
    p2c79._write_csv(OUT_DIR / "audit_rows.csv", audit_rows)
    p2c79._write_csv(OUT_DIR / "summary.csv", [branch_row] + stage_rows + audit_rows)
    failures = [
        {"failure": row["failure_class"], "evidence": row}
        for row in audit_rows
        if row["status"] != "publishable"
    ]
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(p2c79._jsonable(item), ensure_ascii=False) + "\n" for item in failures),
        encoding="utf-8",
    )
    payload = {
        "experiment": OUT_DIR.name,
        "eps": EPS,
        "sentinel_names": sentinel_names,
        "total_wall_s": total_wall_s,
        "p2c102_wall_s": p2c102_summary["wall_s"],
        "stage_rows": stage_rows,
        "direction_timing_rows": timing_rows,
        "audit_rows": audit_rows,
        "branch_row": branch_row,
        "jax_devices": [str(device) for device in jax.devices()],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(p2c79._jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(stage_rows)
    (OUT_DIR / "record.md").write_text(_record(stage_rows, audit_rows, total_wall_s, p2c102_summary["wall_s"]), encoding="utf-8")
    print(json.dumps(p2c79._jsonable({"total_wall_s": total_wall_s, "stage_rows": stage_rows, "audit_rows": audit_rows}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
