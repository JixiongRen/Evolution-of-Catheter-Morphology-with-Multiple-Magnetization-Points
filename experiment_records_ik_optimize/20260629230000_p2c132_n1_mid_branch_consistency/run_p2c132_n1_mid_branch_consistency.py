"""Check branch consistency for the n1 middle-magnet candidate."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "experiment_records_ik_optimize"
P2C131_DIR = EXPERIMENT_ROOT / "20260629210000_p2c131_n1_configuration_boundary"
for path in (ROOT, EXPERIMENT_ROOT, P2C131_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import matplotlib.pyplot as plt
import numpy as np

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c131_n1_configuration_boundary as p2c131


jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent
TASK_MODE = "position_3d"
BRANCH_DRIFT_GATE_M = 1.0e-3
INIT_MODES = ("oracle", "half_oracle", "zero_state", "zero_current_oracle")


def jsonable(value: Any) -> Any:
    """Convert values into JSON-compatible objects."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def arc_resample(points: np.ndarray, n: int) -> np.ndarray:
    """Resample a centerline by arc length."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == n:
        return pts.copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 1.0e-12:
        return np.repeat(pts[:1], n, axis=0)
    target = np.linspace(0.0, s[-1], n)
    return np.vstack([np.interp(target, s, pts[:, k]) for k in range(3)]).T


def initial_states(target_oracle: dict[str, Any], zero_oracle: dict[str, Any], params: Any) -> dict[str, np.ndarray]:
    """Build independent initial guesses for branch testing."""
    oracle_y = p2c131.p2c2.p2c1._y_guess_from_oracle(target_oracle["z"], params, "segment_wrench")
    zero_oracle_y = p2c131.p2c2.p2c1._y_guess_from_oracle(zero_oracle["z"], params, "segment_wrench")
    return {
        "oracle": oracle_y,
        "half_oracle": 0.5 * oracle_y,
        "zero_state": np.zeros_like(oracle_y),
        "zero_current_oracle": zero_oracle_y,
    }


def run_one(init_mode: str, y0: np.ndarray, spec: p2c131.N1Spec, target_oracle: dict[str, Any], zero_oracle: dict[str, Any], current_i: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    """Run staged solve from one initial state."""
    params = target_oracle["params"]
    target_p = np.asarray(jax.device_get(target_oracle["sites"]["tip_p_dim"]), dtype=float).reshape(3)
    zero_p = np.asarray(jax.device_get(zero_oracle["sites"]["tip_p_dim"]), dtype=float).reshape(3)
    direction_raw = target_p - zero_p
    direction = direction_raw / max(float(np.linalg.norm(direction_raw)), 1.0e-12)
    y = np.asarray(y0, dtype=float).reshape(-1)
    y_ref = p2c131.p2c2.p2c1._y_guess_from_oracle(target_oracle["z"], params, "segment_wrench")
    rows: list[dict[str, Any]] = []
    final_solve: dict[str, Any] | None = None
    for stage, (position_scale, max_iter) in enumerate(p2c131.STAGES, start=1):
        args = p2c131.p2c110.Args()
        args.position_scale = float(position_scale)
        args.max_iter = int(max_iter)
        funcs = p2c131.make_functions(params, target_p, direction, y_ref, args, task_mode=TASK_MODE)
        solve = p2c131.p2c110._solve(y, current_i, funcs, args)
        load = p2c131.p2c110.p2c57.p2c47._interface_load_metrics(params, solve["y"], steps_per_flex=args.steps_per_flex)
        boundary = np.asarray(solve["out"]["boundary"], dtype=float).reshape(6)
        tip_p = np.asarray(solve["out"]["tip_p"], dtype=float).reshape(3)
        err = tip_p - target_p
        row = {
            "case": spec.name,
            "task_mode": TASK_MODE,
            "init_mode": init_mode,
            "stage": stage,
            "position_scale": position_scale,
            "max_iter": max_iter,
            "position_error_m": float(np.linalg.norm(err)),
            "projected_position_error_m": float(abs(np.dot(err, direction))),
            "stationarity_norm": solve["stationarity_norm"],
            "jacobian_condition": solve["jacobian_condition"],
            "jacobian_rank": solve["jacobian_rank"],
            "continuity_norm": solve["continuity_norm"],
            "terminal_force_norm": float(np.linalg.norm(boundary[:3])),
            "terminal_torque_norm": float(np.linalg.norm(boundary[3:])),
            **load,
            "nfev": solve["nfev"],
            "steady_s": solve["steady_s"],
            "success": solve["success"],
            "message": solve["message"],
        }
        row["task_error_gate"] = bool(row["position_error_m"] <= 5.0e-5)
        row["baseline_gate"] = bool(
            row["success"]
            and row["task_error_gate"]
            and row["stationarity_norm"] <= 1.0e-5
            and row["jacobian_condition"] < 1.0e7
            and row["continuity_norm"] <= 1.0e-3
            and row["max_interface_force_jump_over_rigid_force"] <= 1.5
        )
        row["failure_class"] = p2c131.classify(row)
        rows.append(row)
        y = solve["y"]
        final_solve = solve
    assert final_solve is not None
    centerline = np.asarray(final_solve["out"]["centerline"], dtype=float)
    np.save(OUT_DIR / f"{init_mode}_centerline.npy", centerline)
    np.save(OUT_DIR / f"{init_mode}_state_y.npy", np.asarray(final_solve["y"], dtype=float))
    return rows, rows[-1], centerline


def pairwise_branch_rows(centerlines: dict[str, np.ndarray], final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute pairwise branch drift among final centerlines."""
    rows: list[dict[str, Any]] = []
    row_by_mode = {row["init_mode"]: row for row in final_rows}
    modes = list(centerlines)
    for i, a in enumerate(modes):
        for b in modes[i + 1 :]:
            n = max(len(centerlines[a]), len(centerlines[b]))
            ca = arc_resample(centerlines[a], n)
            cb = arc_resample(centerlines[b], n)
            d = np.linalg.norm(ca - cb, axis=1)
            rows.append(
                {
                    "init_a": a,
                    "init_b": b,
                    "both_baseline_gate": bool(row_by_mode[a]["baseline_gate"] and row_by_mode[b]["baseline_gate"]),
                    "max_centerline_drift_m": float(np.max(d)),
                    "rms_centerline_drift_m": float(np.sqrt(np.mean(d**2))),
                    "tip_drift_m": float(np.linalg.norm(ca[-1] - cb[-1])),
                    "branch_gate": bool(row_by_mode[a]["baseline_gate"] and row_by_mode[b]["baseline_gate"] and np.max(d) <= BRANCH_DRIFT_GATE_M),
                }
            )
    return rows


def plot_results(final_rows: list[dict[str, Any]], branch_rows: list[dict[str, Any]], centerlines: dict[str, np.ndarray], target_centerline: np.ndarray) -> None:
    """Plot gate metrics and final centerlines."""
    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    ax0 = fig.add_subplot(121, projection="3d")
    ax0.plot(target_centerline[:, 0], target_centerline[:, 1], target_centerline[:, 2], "k-", lw=2.0, label="FK 参考")
    for mode, cl in centerlines.items():
        ax0.plot(cl[:, 0], cl[:, 1], cl[:, 2], "--", lw=1.8, label=mode)
    ax0.set_title("不同初值收敛形态")
    ax0.set_xlabel("x / m")
    ax0.set_ylabel("y / m")
    ax0.set_zlabel("z / m")
    ax0.legend(fontsize=8)
    ax1 = fig.add_subplot(122)
    labels = [row["init_mode"] for row in final_rows]
    ax1.bar(labels, [row["position_error_m"] for row in final_rows], color="#f4a261", label="位置误差")
    ax1.axhline(5.0e-5, color="black", linestyle="--", linewidth=1)
    ax1.set_yscale("log")
    ax1.set_title("末端位置误差")
    ax1.grid(axis="y", alpha=0.25)
    ax1.tick_params(axis="x", rotation=20)
    fig.savefig(OUT_DIR / "n1_mid_branch_consistency_cn.png", dpi=220)
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    labels2 = [f"{row['init_a']}\n{row['init_b']}" for row in branch_rows]
    ax.bar(labels2, [row["max_centerline_drift_m"] for row in branch_rows], color="#457b9d")
    ax.axhline(BRANCH_DRIFT_GATE_M, color="black", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_title("两两中心线最大漂移")
    ax.set_ylabel("m")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig2.savefig(OUT_DIR / "n1_mid_branch_pairwise_drift_cn.png", dpi=220)
    plt.close(fig2)


def write_record(summary: dict[str, Any], final_rows: list[dict[str, Any]], branch_rows: list[dict[str, Any]]) -> None:
    """Write the Chinese experiment record."""
    final_lines = "\n".join(
        f"- 初值 {row['init_mode']}：基线门禁 {row['baseline_gate']}；位置误差 {row['position_error_m']:.3e} m；"
        f"一阶最优性 {row['stationarity_norm']:.3e}；条件数 {row['jacobian_condition']:.3e}；"
        f"载荷传播比例 {row['max_interface_force_jump_over_rigid_force']:.3e}；失败分类 {row['failure_class']}。"
        for row in final_rows
    )
    branch_lines = "\n".join(
        f"- {row['init_a']} vs {row['init_b']}：两者均过基线 {row['both_baseline_gate']}；最大中心线漂移 {row['max_centerline_drift_m']:.3e} m；分支门禁 {row['branch_gate']}。"
        for row in branch_rows
    )
    record = f"""# P2c132 单点中段磁体候选分支一致性复核

## 实验目的

P2c131 发现 `n1_mid_z_rand15_a010` 的三维末端位置任务可通过基线门禁。本轮检查该结果是否依赖 FK oracle 初值。如果不同合理初值收敛到同一中心线分支，才允许后续进入扰动重求解；否则只能作为初始化敏感候选。

## 实验方案

固定中段单磁体轴向磁化、随机电流 `alpha=0.10` 和三维末端位置任务。测试四种初值：FK oracle 内力、半 FK oracle 内力、零状态、零电流 FK oracle 内力。所有求解使用与 P2c131 相同的显式刚段载荷残差和相同门禁。

分支一致性门禁定义为：至少两条过基线路径之间的中心线最大逐点漂移不超过 `1e-3 m`。

## 基线结果

{final_lines}

## 分支漂移

{branch_lines}

## 结论

分支一致性结论：{summary['branch_status']}。本轮仍不发布灵敏度；若存在至少两条过基线且互相满足分支漂移门禁的路径，下一轮才允许进行小扰动重求解。

## 下一步计划

若分支一致性通过，执行 P2c133：对该中段单磁体候选做 `eps=0.005` 和 `0.010` 的扰动重求解一致性检查。若不通过，先归因初始化敏感或多平衡竞争，不进入扰动扫描。

## 耗时

- 墙钟耗时：{summary['wall_s']:.2f} s。
- 重求解净耗时：{summary['total_solve_steady_s']:.2f} s。
"""
    (OUT_DIR / "record.md").write_text(record, encoding="utf-8")


def main() -> None:
    """Run the branch consistency audit."""
    t0 = time.perf_counter()
    spec = p2c131.N1Spec("n1_mid_z_rand15_a010", ((0.0, 0.0, 0.0), (0.0, 0.0, p2c131.M_MAG), (0.0, 0.0, 0.0)))
    case = p2c131.build_case(spec)
    zero_oracle, target_oracle, _engine = p2c131.solve_oracles(case)
    params = target_oracle["params"]
    current_i = np.asarray(case.currents, dtype=float)
    init_map = initial_states(target_oracle, zero_oracle, params)
    all_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    centerlines: dict[str, np.ndarray] = {}
    for mode in INIT_MODES:
        rows, final, centerline = run_one(mode, init_map[mode], spec, target_oracle, zero_oracle, current_i)
        all_rows.extend(rows)
        final_rows.append(final)
        centerlines[mode] = centerline
    branch_rows = pairwise_branch_rows(centerlines, final_rows)
    passing_pairs = [row for row in branch_rows if row["branch_gate"]]
    baseline_pass_count = sum(row["baseline_gate"] for row in final_rows)
    branch_status = "通过" if len(passing_pairs) >= 1 else "未通过"
    target_centerline = np.asarray(target_oracle["centerline"], dtype=float)
    np.save(OUT_DIR / "target_oracle_centerline.npy", target_centerline)
    plot_results(final_rows, branch_rows, centerlines, target_centerline)
    summary = {
        "experiment_id": OUT_DIR.name,
        "case": spec.name,
        "task_mode": TASK_MODE,
        "baseline_pass_count": baseline_pass_count,
        "branch_pair_pass_count": len(passing_pairs),
        "branch_status": branch_status,
        "publishable_claim": False,
        "outer_ik_allowed": False,
        "disturbance_scan_allowed": bool(len(passing_pairs) >= 1),
        "final_rows": final_rows,
        "branch_rows": branch_rows,
        "total_solve_steady_s": sum(row["steady_s"] for row in all_rows),
        "wall_s": time.perf_counter() - t0,
    }
    write_csv(OUT_DIR / "stage_rows.csv", all_rows)
    write_csv(OUT_DIR / "final_rows.csv", final_rows)
    write_csv(OUT_DIR / "branch_pair_rows.csv", branch_rows)
    write_csv(OUT_DIR / "summary.csv", [{"branch_status": branch_status, "baseline_pass_count": baseline_pass_count, "branch_pair_pass_count": len(passing_pairs), "wall_s": summary["wall_s"]}])
    (OUT_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(
            json.dumps({"init_mode": row["init_mode"], "failure_class": row["failure_class"], "baseline_gate": row["baseline_gate"]}, ensure_ascii=False) + "\n"
            for row in final_rows
            if not row["baseline_gate"]
        ),
        encoding="utf-8",
    )
    write_record(summary, final_rows, branch_rows)
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
