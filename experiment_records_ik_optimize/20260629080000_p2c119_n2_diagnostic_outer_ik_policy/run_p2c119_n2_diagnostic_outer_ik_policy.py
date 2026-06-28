"""Validate the diagnostic outer-IK policy for n2 local position targets."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk
from n2_diagnostic_outer_ik_policy import N2DiagnosticOuterIKPolicy


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write CSV rows."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_json(path: Path, data: dict) -> None:
    """Write JSON output."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_targets(policy: N2DiagnosticOuterIKPolicy) -> list[tuple[str, np.ndarray]]:
    """Build representative diagnostic target offsets."""
    U = policy.U
    s = policy.singular_values
    radius = policy.radius_A
    return [
        ("mode1_three_steps", U[:, 0] * s[0] * radius * 3.0),
        ("mode1_eight_steps", U[:, 0] * s[0] * radius * 8.0),
        ("mode2_three_steps", U[:, 1] * s[1] * radius * 3.0),
        ("mode3_three_steps", U[:, 2] * s[2] * radius * 3.0),
        ("cart_x_small", np.array([s[0] * radius * 2.0, 0.0, 0.0])),
        ("cart_y_weak", np.array([0.0, s[0] * radius * 2.0, 0.0])),
        ("cart_y_large", np.array([0.0, s[0] * radius * 12.0, 0.0])),
        ("mixed_weak_heavy", 0.4 * U[:, 0] * s[0] * radius + 4.0 * U[:, 2] * s[2] * radius),
    ]


def plot_rows(rows: list[dict]) -> None:
    """Plot policy action and estimated step count."""
    configure_matplotlib_cjk()
    names = [row["case"] for row in rows]
    steps = [row["estimated_steps"] for row in rows]
    weak = [row["weak_mode_fraction"] for row in rows]
    color_map = {"快速局部校正": "#207a3c", "慢速诊断推进": "#c58a19", "拒绝直接外层 IK": "#b33a3a"}
    colors = [color_map[row["action"]] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axes[0].bar(names, steps, color=colors)
    axes[0].axhline(8, color="black", linestyle="--", linewidth=1.0, label="诊断步数上限 8")
    axes[0].set_ylabel("估计小步数")
    axes[0].set_title("n2 诊断模式外层 IK 策略分类")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()
    axes[1].bar(names, weak, color=colors)
    axes[1].axhline(0.25, color="black", linestyle=":", linewidth=1.0, label="快速校正弱方向阈值")
    axes[1].axhline(0.60, color="black", linestyle="--", linewidth=1.0, label="拒绝阈值")
    axes[1].set_ylabel("弱可控方向占比")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()
    fig.savefig(OUT_DIR / "n2_diagnostic_outer_ik_policy_cn.png", dpi=220)
    plt.close(fig)


def main() -> None:
    """Run policy validation."""
    t0 = time.perf_counter()
    policy = N2DiagnosticOuterIKPolicy(ROOT)
    rows: list[dict] = []
    failures: list[dict] = []
    for name, target in build_targets(policy):
        decision = policy.classify(target)
        dI = policy.first_step(target)
        step_decision = policy.interface.validate_step_request(
            {
                **policy.interface.as_dict()["required_request"],
                "step_type": "current_increment",
                "dI_A": dI.tolist(),
                "uses_physical_budget_as_linear_radius": False,
            }
        )
        if not step_decision.accepted:
            failures.append({"case": name, "failure_class": "first_step_rejected", "reasons": step_decision.reasons})
        rows.append(
            {
                "case": name,
                "action": decision.action,
                "reason": decision.reason,
                "target_norm_um": float(np.linalg.norm(target) * 1.0e6),
                "estimated_steps": decision.estimated_steps,
                "required_current_norm_A": decision.required_current_norm_A,
                "first_step_norm_A": decision.first_step_norm_A,
                "first_step_contract_accepted": step_decision.accepted,
                "controllability_ratio": decision.controllability_ratio,
                "weak_mode_fraction": decision.weak_mode_fraction,
            }
        )

    action_counts = {action: sum(row["action"] == action for row in rows) for action in ["快速局部校正", "慢速诊断推进", "拒绝直接外层 IK"]}
    summary = {
        "experiment_id": OUT_DIR.name,
        "status": "diagnostic_policy_ready" if not failures else "diagnostic_policy_failed",
        "policy": policy.as_dict(),
        "action_counts": action_counts,
        "total_cases": len(rows),
        "failure_count": len(failures),
        "wall_s": time.perf_counter() - t0,
        "interpretation": "该策略把 n2 灵敏度限制为诊断模式局部外层 IK，不允许默认生产接入或大步全局求解。",
    }
    write_csv(OUT_DIR / "policy_rows.csv", rows)
    write_csv(OUT_DIR / "summary.csv", [{**action_counts, "total_cases": len(rows), "failure_count": len(failures), "wall_s": summary["wall_s"]}])
    write_json(OUT_DIR / "summary.json", summary)
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in failures),
        encoding="utf-8",
    )
    plot_rows(rows)
    row_lines = "\n".join(
        f"- {row['case']}：{row['action']}，估计 {row['estimated_steps']} 步，弱方向占比 {row['weak_mode_fraction']:.3f}。"
        for row in rows
    )
    record = f"""# P2c119 n2 诊断模式外层 IK 策略

## 实验目的

P2c118 表明 n2 小步外层 IK 能在强可控方向快速降低误差，但对弱方向推进很慢。本轮将该现象整理为诊断模式策略：在执行任何外层步进前，先根据局部灵敏度矩阵评估目标方向、估计步数，并决定是否允许进入小步诊断推进。

## 策略规则

- 单步电流半径固定为 P2c117 的 `0.0125 A`。
- 预计步数不超过 4 且弱方向占比小于 0.25：快速局部校正。
- 预计步数不超过 8 且弱方向占比小于 0.60：慢速诊断推进。
- 超出上述范围：拒绝直接外层 IK，建议任务重规划或更换工作点。

## 结果

- 快速局部校正：{action_counts["快速局部校正"]} 组。
- 慢速诊断推进：{action_counts["慢速诊断推进"]} 组。
- 拒绝直接外层 IK：{action_counts["拒绝直接外层 IK"]} 组。
- 首步契约失败：{len(failures)} 组。
- 墙钟耗时：{summary["wall_s"]:.6f} s。

{row_lines}

## 结论

该策略使 n2 灵敏度具备工程可解释性：它不是全局逆运动学求解器，而是可信分支、小电流半径内的局部诊断与校正工具。强方向可快速修正，弱方向必须报告低效率或拒绝直接求解。

## 下一步计划

后续应把该策略用于更大样本的目标方向扫描，输出 n2 工作点的局部可控锥和不可控方向图；同时回到 n1，继续判断是否存在可发布的有效子空间位置灵敏度。
"""
    (OUT_DIR / "record.md").write_text(record, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
