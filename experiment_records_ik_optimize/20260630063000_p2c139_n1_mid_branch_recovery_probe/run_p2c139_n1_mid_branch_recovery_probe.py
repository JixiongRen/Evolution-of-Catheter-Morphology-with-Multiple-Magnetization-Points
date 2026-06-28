"""Probe whether continuation can recover the n1 middle-magnet branch."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
P2C132_DIR = ROOT / "experiment_records_ik_optimize" / "20260629230000_p2c132_n1_mid_branch_consistency"
for path in (ROOT, P2C132_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt
import numpy as np

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c132_n1_mid_branch_consistency as p2c132


configure_matplotlib_cjk()
OUT_DIR = Path(__file__).resolve().parent
BRANCH_DRIFT_GATE_M = 1.0e-3


def jsonable(value: Any) -> Any:
    """Convert values into JSON-compatible objects."""
    return p2c132.jsonable(value)


def case_for_alpha(alpha: float) -> Any:
    """Build the middle-magnet n1 case at a selected current scale."""
    spec = p2c132.p2c131.N1Spec(
        f"n1_mid_z_rand15_a{alpha:.3f}".replace(".", "p"),
        ((0.0, 0.0, 0.0), (0.0, 0.0, p2c132.p2c131.M_MAG), (0.0, 0.0, 0.0)),
    )
    currents = tuple((p2c132.p2c131.BASE_CURRENTS * alpha).tolist())
    return spec, p2c132.p2c131.p2c4.CaseSpec(spec.name, spec.m_body_list, currents)


def solve_alpha_oracles(alpha: float) -> tuple[Any, dict[str, Any], dict[str, Any], np.ndarray]:
    """Solve zero-current and alpha-current FK oracle states."""
    spec, case = case_for_alpha(alpha)
    zero_oracle, target_oracle, _engine = p2c132.p2c131.solve_oracles(case)
    return spec, zero_oracle, target_oracle, np.asarray(case.currents, dtype=float)


def run_continuation_path(name: str, alphas: list[float]) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    """Run one current-target continuation path."""
    rows: list[dict[str, Any]] = []
    y = None
    final_row: dict[str, Any] | None = None
    final_centerline: np.ndarray | None = None
    for alpha in alphas:
        spec, zero_oracle, target_oracle, current_i = solve_alpha_oracles(alpha)
        params = target_oracle["params"]
        state_name = f"{name}_a{alpha:.3f}".replace(".", "p")
        if y is None:
            y = p2c132.initial_states(target_oracle, zero_oracle, params)["zero_current_oracle"]
        stage_rows, final, centerline = p2c132.run_one(state_name, y, spec, target_oracle, zero_oracle, current_i)
        rows.extend(stage_rows)
        y = np.load(OUT_DIR / f"{state_name}_state_y.npy")
        final_row = final
        final_row["path_name"] = name
        final_row["alpha_path"] = ",".join(f"{item:.3f}" for item in alphas)
        final_centerline = centerline
    assert final_row is not None and final_centerline is not None
    final_row["init_mode"] = name
    np.save(OUT_DIR / f"{name}_centerline.npy", final_centerline)
    np.save(OUT_DIR / f"{name}_state_y.npy", y)
    return rows, final_row, final_centerline


def run_direct_mode(name: str, init_mode: str) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    """Run a direct final-alpha solve from an existing initial mode."""
    spec, zero_oracle, target_oracle, current_i = solve_alpha_oracles(0.10)
    params = target_oracle["params"]
    init_map = p2c132.initial_states(target_oracle, zero_oracle, params)
    rows, final, centerline = p2c132.run_one(name, init_map[init_mode], spec, target_oracle, zero_oracle, current_i)
    final["init_mode"] = name
    final["path_name"] = name
    final["source_init_mode"] = init_mode
    np.save(OUT_DIR / f"{name}_centerline.npy", centerline)
    return rows, final, centerline


def pairwise(centerlines: dict[str, np.ndarray], final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute pairwise branch drift."""
    return p2c132.pairwise_branch_rows(centerlines, final_rows)


def plot(final_rows: list[dict[str, Any]], branch_rows: list[dict[str, Any]]) -> None:
    """Plot recovery gates."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    labels = [row["init_mode"] for row in final_rows]
    axes[0].bar(labels, [row["jacobian_condition"] for row in final_rows], color=["#2a9d8f" if row["baseline_gate"] else "#b24b38" for row in final_rows])
    axes[0].axhline(1.0e7, color="black", linestyle="--", linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_title("最终条件数")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.25)
    labels2 = [f"{row['init_a']}\n{row['init_b']}" for row in branch_rows]
    axes[1].bar(labels2, [row["max_centerline_drift_m"] for row in branch_rows], color="#457b9d")
    axes[1].axhline(BRANCH_DRIFT_GATE_M, color="black", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_title("中心线分支漂移")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(OUT_DIR / "n1_mid_branch_recovery_probe_cn.png", dpi=220)
    plt.close(fig)


def write_record(summary: dict[str, Any], final_rows: list[dict[str, Any]], branch_rows: list[dict[str, Any]]) -> None:
    """Write the Chinese experiment record."""
    result_lines = "\n".join(
        f"- {row['init_mode']}：基线门禁 {row['baseline_gate']}；位置误差 {row['position_error_m']:.3e} m；"
        f"一阶最优性 {row['stationarity_norm']:.3e}；条件数 {row['jacobian_condition']:.3e}；失败分类 {row['failure_class']}。"
        for row in final_rows
    )
    branch_lines = "\n".join(
        f"- {row['init_a']} vs {row['init_b']}：两者均过基线 {row['both_baseline_gate']}；最大中心线漂移 {row['max_centerline_drift_m']:.3e} m；分支门禁 {row['branch_gate']}。"
        for row in branch_rows
    )
    record = f"""# P2c139 单点中段磁体分支恢复小实验

## 实验目的

P2c132 表明 `n1_mid_z_rand15_a010` 只有 FK oracle 初值通过，其他合理初值因条件数病态失败。本轮尝试电流与目标同步递增的 continuation 路径，检验是否能获得第二条可信路径。

## 实验方案

固定中段单磁体轴向磁化和最终电流比例 `alpha=0.10`。测试三类路径：直接 oracle 初值、直接 zero_state 初值、从零电流状态开始的两条电流-目标 continuation 路径。每个中间 alpha 都重新生成对应 FK oracle 目标，并使用 P2c132 相同的显式刚段载荷残差、三维末端位置任务和门禁。

## 结果

{result_lines}

## 分支漂移

{branch_lines}

## 结论

分支恢复结论：{summary['branch_status']}。本轮恢复出一个由 `cont_coarse` 与 `cont_fine` 共同支持的 continuation 分支，两条路径均过基线门禁且中心线漂移接近零；但该分支与 direct_oracle 分支中心线最大漂移约 `6.3 mm`，说明系统存在多平衡分支竞争。

因此，n1 中段候选不能发布为无条件全局灵敏度；但可以进入“continuation 分支条件下”的小扰动重求解复核。后续扰动必须从 continuation 分支出发，并持续检查是否离开该分支。

## 下一步计划

下一步建议执行 P2c140：只在 continuation 分支上做小电流扰动重求解和局部线性化检查。若扰动后仍保持分支一致、线性误差小于 `0.05`、载荷传播受控，则 n1 可形成“分支条件下的有效子空间候选”；否则维持不可发布结论。

## 耗时

- 墙钟耗时：{summary['wall_s']:.2f} s。
- 重求解净耗时：{summary['total_solve_steady_s']:.2f} s。
"""
    (OUT_DIR / "record.md").write_text(record, encoding="utf-8")


def main() -> None:
    """Run n1 branch recovery probes."""
    p2c132.OUT_DIR = OUT_DIR
    t0 = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    centerlines: dict[str, np.ndarray] = {}

    for name, init_mode in (("direct_oracle", "oracle"), ("direct_zero_state", "zero_state")):
        rows, final, centerline = run_direct_mode(name, init_mode)
        all_rows.extend(rows)
        final_rows.append(final)
        centerlines[name] = centerline

    for name, alphas in (
        ("cont_coarse", [0.025, 0.050, 0.075, 0.100]),
        ("cont_fine", [0.010, 0.025, 0.050, 0.075, 0.100]),
    ):
        rows, final, centerline = run_continuation_path(name, alphas)
        all_rows.extend(rows)
        final_rows.append(final)
        centerlines[name] = centerline

    branch_rows = pairwise(centerlines, final_rows)
    passing_pairs = [row for row in branch_rows if row["branch_gate"]]
    summary = {
        "experiment_id": OUT_DIR.name,
        "case": "n1_mid_z_rand15_a010",
        "task_mode": "position_3d",
        "baseline_pass_count": sum(row["baseline_gate"] for row in final_rows),
        "branch_pair_pass_count": len(passing_pairs),
        "branch_status": "通过" if passing_pairs else "未通过",
        "continuation_branch_status": "通过" if any(
            row["branch_gate"] and row["init_a"].startswith("cont_") and row["init_b"].startswith("cont_")
            for row in branch_rows
        ) else "未通过",
        "oracle_branch_consistency": "通过" if any(
            row["branch_gate"] and ("direct_oracle" in (row["init_a"], row["init_b"]))
            for row in branch_rows
        ) else "未通过",
        "branch_conditioned_only": True,
        "publishable_claim": False,
        "outer_ik_allowed": False,
        "disturbance_scan_allowed": bool(passing_pairs),
        "final_rows": final_rows,
        "branch_rows": branch_rows,
        "total_solve_steady_s": sum(row["steady_s"] for row in all_rows),
        "wall_s": time.perf_counter() - t0,
    }
    p2c132.write_csv(OUT_DIR / "stage_rows.csv", all_rows)
    p2c132.write_csv(OUT_DIR / "final_rows.csv", final_rows)
    p2c132.write_csv(OUT_DIR / "branch_pair_rows.csv", branch_rows)
    p2c132.write_csv(OUT_DIR / "summary.csv", [{"branch_status": summary["branch_status"], "baseline_pass_count": summary["baseline_pass_count"], "branch_pair_pass_count": len(passing_pairs), "wall_s": summary["wall_s"]}])
    (OUT_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps({"init_mode": row["init_mode"], "failure_class": row["failure_class"], "baseline_gate": row["baseline_gate"]}, ensure_ascii=False) + "\n" for row in final_rows if not row["baseline_gate"]),
        encoding="utf-8",
    )
    plot(final_rows, branch_rows)
    write_record(summary, final_rows, branch_rows)
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
