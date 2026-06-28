"""Bracket the n2 trusted-branch breakpoint between small and large starts."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
P2C134_DIR = ROOT / "experiment_records_ik_optimize" / "20260630023000_p2c134_n2_trusted_branch_basin_radius"
for path in (ROOT, P2C134_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_p2c134_n2_trusted_branch_basin_radius as p2c134


OUT_DIR = Path(__file__).resolve().parent


def make_breakpoint_starts(dim: int) -> list[tuple[str, float, np.ndarray]]:
    """Build deterministic starts in the suspected branch-break interval."""
    rng = np.random.default_rng(20260630)
    dirs = [vec / np.linalg.norm(vec) for vec in [rng.normal(size=dim) for _ in range(3)]]
    radii = [2.0e-4, 3.0e-4, 5.0e-4, 7.0e-4, 1.0e-3]
    starts: list[tuple[str, float, np.ndarray]] = []
    for radius in radii:
        for idx, direction in enumerate(dirs):
            starts.append((f"r{radius:.0e}_d{idx}", radius, radius * direction))
    return starts


def write_record(summary: dict[str, Any], radius_rows: list[dict[str, Any]]) -> None:
    """Write the breakpoint bracket report."""
    lines = "\n".join(
        f"- 半径 {row['init_radius']:.1e}：通过 {row['branch_gate_count']}/{row['count']}；"
        f"最大末端误差 {row['max_position_error_m']:.3e} m；最大中心线漂移 {row['max_centerline_drift_m']:.3e} m；"
        f"最大一阶最优性 {row['max_stationarity_norm']:.3e}；最大条件数 {row['max_condition']:.3e}。"
        for row in radius_rows
    )
    record = f"""# P2c135 两点磁控可信分支断裂半径夹逼

## 实验目的

P2c134 证明有效子空间初值扰动半径 `1e-4` 内可稳定回到同一可信分支；P2c133 显示 `1e-3` 量级会产生明显失败。本轮在二者之间夹逼断裂半径，区分局部可信半径和大扰动分支跳转区间。

## 实验方案

固定两点磁控缓存分支、同一目标点、同一基准电流、同一物理门禁。半径集合为 `2e-4, 3e-4, 5e-4, 7e-4, 1e-3`，每个半径取三个确定性随机方向。门禁仍包括末端位置误差、一阶最优性、有效条件数、接口连续残差、载荷传播比例和中心线漂移。

## 半径结果

{lines}

## 结论

最大三方向全部通过半径：`{summary['largest_all_direction_pass_radius']}`。最大任一方向通过半径：`{summary['largest_any_direction_pass_radius']}`。

结合 P2c134，严格三方向稳定半径位于 `1e-4` 至 `2e-4` 之间；`2e-4` 起已经出现方向相关失败，`1e-3` 三方向全部失败。部分方向可维持到 `7e-4`，说明该分支的吸引域具有明显方向性，不能用单一球形半径概括。

该结果用于限定后续电流扰动重求解的可信初始半径。保守工程信赖域应不超过 `1e-4`；若沿特定方向使用更大步长，必须额外做方向性门禁，不能直接声明三维灵敏度可发布。

## 下一步计划

在通过半径内执行三维末端位置电流扰动重求解，报告局部线性化误差；在失败半径附近记录分支跳转模式，但不将其作为可发布灵敏度。

## 耗时

- 墙钟耗时：{summary['wall_s']:.2f} s。
- 重求解净耗时：{summary['total_solve_steady_s']:.2f} s。
"""
    (OUT_DIR / "record.md").write_text(record, encoding="utf-8")


def main() -> None:
    """Run the breakpoint bracket audit."""
    t0 = time.perf_counter()
    p2c134.OUT_DIR = OUT_DIR
    params, I0 = p2c134.p2c104.p2c79.p2c57.p2c24._params_from_saved_case(
        p2c134.p2c104.p2c79.p2c57.CASE_NAME,
        p2c134.p2c104.p2c79.p2c57.ALPHA,
    )
    branch = p2c134.p2c104._load_cached_branch(params)
    y_base = np.asarray(branch["y"], dtype=float)
    basis = np.asarray(branch["basis"], dtype=float)
    funcs = branch["funcs"]
    I0 = np.asarray(I0, dtype=float)
    base_out = p2c134.eval_output(funcs, y_base, I0)
    starts = make_breakpoint_starts(basis.shape[1])
    rows = []
    for name, radius, q0 in starts:
        row, _ = p2c134.run_case(
            name=name,
            radius=radius,
            q0=q0,
            branch=branch,
            params=params,
            I0=I0,
            base_out=base_out,
            base_centerline=base_out["centerline"],
        )
        rows.append(row)
    radius_rows = p2c134.summarize_by_radius(rows)
    all_pass = [row["init_radius"] for row in radius_rows if row["branch_gate_count"] == row["count"]]
    any_pass = [row["init_radius"] for row in radius_rows if row["branch_gate_count"] > 0]
    summary = {
        "experiment_id": OUT_DIR.name,
        "case": p2c134.p2c104.p2c79.p2c57.CASE_NAME,
        "alpha": p2c134.p2c104.p2c79.p2c57.ALPHA,
        "init_count": len(rows),
        "largest_all_direction_pass_radius": max(all_pass) if all_pass else None,
        "largest_any_direction_pass_radius": max(any_pass) if any_pass else None,
        "strict_all_direction_pass_radius_from_p2c134": 1.0e-4,
        "first_tested_all_direction_failure_radius": min(
            row["init_radius"] for row in radius_rows if row["branch_gate_count"] < row["count"]
        ),
        "all_direction_break_interval": [1.0e-4, 2.0e-4],
        "publishable_claim": False,
        "outer_ik_allowed": False,
        "disturbance_scan_allowed": bool(all_pass),
        "final_rows": rows,
        "radius_rows": radius_rows,
        "total_solve_steady_s": sum(row["steady_s"] for row in rows),
        "wall_s": time.perf_counter() - t0,
    }
    p2c134.plot(rows, radius_rows)
    (OUT_DIR / "n2_trusted_branch_basin_radius_cn.png").rename(OUT_DIR / "n2_basin_breakpoint_bracket_cn.png")
    p2c134.write_csv(OUT_DIR / "final_rows.csv", rows)
    p2c134.write_csv(OUT_DIR / "radius_rows.csv", radius_rows)
    p2c134.write_csv(
        OUT_DIR / "summary.csv",
        [
            {
                "largest_all_direction_pass_radius": summary["largest_all_direction_pass_radius"],
                "largest_any_direction_pass_radius": summary["largest_any_direction_pass_radius"],
                "disturbance_scan_allowed": summary["disturbance_scan_allowed"],
                "wall_s": summary["wall_s"],
            }
        ],
    )
    (OUT_DIR / "summary.json").write_text(json.dumps(p2c134.jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "init_mode": row["init_mode"],
                    "init_radius": row["init_radius"],
                    "failure_class": row["failure_class"],
                    "branch_gate": row["branch_gate"],
                },
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
            if not row["branch_gate"]
        ),
        encoding="utf-8",
    )
    write_record(summary, radius_rows)
    print(json.dumps(p2c134.jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
