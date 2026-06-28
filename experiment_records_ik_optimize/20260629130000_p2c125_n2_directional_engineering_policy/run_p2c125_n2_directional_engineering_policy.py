"""Generate the n2 directional engineering policy package."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(OUT_DIR) not in sys.path:
    sys.path.insert(0, str(OUT_DIR))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk  # noqa: E402
from n2_directional_engineering_policy import N2DirectionalEngineeringPolicy  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV rows."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write stable JSON output."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def decision_row(name: str, target: np.ndarray, policy: N2DirectionalEngineeringPolicy) -> dict[str, Any]:
    """Evaluate one policy test case."""
    decision = policy.classify(target)
    return {
        "case": name,
        "action": decision.action,
        "reason": decision.reason,
        "target_norm_nm": decision.target_norm_nm,
        "estimated_steps": decision.estimated_steps,
        "weak_mode_fraction": decision.weak_mode_fraction,
        "required_current_norm_A": decision.required_current_norm_A,
        "allowed_for_default_outer_ik": decision.allowed_for_default_outer_ik,
    }


def build_validation_rows(policy: N2DirectionalEngineeringPolicy) -> list[dict[str, Any]]:
    """Build deterministic validation cases."""
    strong = policy.U[:, 0]
    mid = policy.U[:, 1]
    weak = policy.U[:, -1]
    return [
        decision_row("强方向_10nm", strong * 10.0e-9, policy),
        decision_row("强方向_50nm", strong * 50.0e-9, policy),
        decision_row("强方向_100nm", strong * 100.0e-9, policy),
        decision_row("弱方向_1nm", weak * 1.0e-9, policy),
        decision_row("弱方向_5nm", weak * 5.0e-9, policy),
        decision_row("中间方向_5nm", mid * 5.0e-9, policy),
        decision_row("混合强弱_5nm", (0.8 * strong + 0.6 * weak) * 5.0e-9, policy),
    ]


def plot_rows(rows: list[dict[str, Any]]) -> None:
    """Plot policy decisions."""
    configure_matplotlib_cjk()
    labels = [row["case"] for row in rows]
    step_values = [row["estimated_steps"] for row in rows]
    weak_values = [row["weak_mode_fraction"] for row in rows]
    colors = []
    for row in rows:
        if row["action"] == "强方向小目标闭环候选":
            colors.append("#247a43")
        elif "诊断" in row["action"]:
            colors.append("#d8a13b")
        else:
            colors.append("#b24b38")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].bar(labels, step_values, color=colors, alpha=0.88)
    axes[0].axhline(policy_steps_limit(), color="black", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("预计小步数")
    axes[0].set_title("n2 方向工程策略：步数门禁")
    axes[1].bar(labels, weak_values, color=colors, alpha=0.88)
    axes[1].axhline(0.60, color="black", linestyle="--", linewidth=1.0, label="弱方向阈值")
    axes[1].set_ylabel("弱方向成分")
    axes[1].set_title("n2 方向工程策略：弱方向成分")
    axes[1].legend()
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUT_DIR / "n2_directional_engineering_policy_cn.png", dpi=220)
    plt.close(fig)


def policy_steps_limit() -> int:
    """Return the plotted diagnostic step limit."""
    return 12


def write_record(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write the Chinese experiment record."""
    row_lines = "\n".join(
        f"- {row['case']}：{row['action']}；目标 {row['target_norm_nm']:.3g} nm；预计步数 {row['estimated_steps']}；弱方向成分 {row['weak_mode_fraction']:.3f}。"
        for row in rows
    )
    record = f"""# P2c125 两点磁控方向工程策略接口报告

## 实验目的

本轮不新增物理求解，而是将 P2c122、P2c123 和 P2c124 的证据固化为只读工程策略接口。目标是防止后续外层逆运动学误用局部灵敏度，尤其防止将弱方向可信导数误解释为可快速推进能力。

## 策略依据

- P2c122：方向扫描显示当前工作点各向异性强，强方向和弱方向单步位移效率差异显著。
- P2c123：强弱方向单步真实重求解闭环全部通过，说明局部导数在小半径内具有物理一致性。
- P2c124：强方向 50 nm 可在 3 步内闭合；弱方向 5 nm 在 12 步后仅下降约 14%，说明弱方向主要受推进效率限制。

## 接口规则

- 强方向目标不超过 50 nm：可作为小目标闭环候选，但每步必须真实重求解复核。
- 弱方向目标不超过 1 nm：仅允许诊断观察，不作为主动修正器。
- 弱方向 5 nm 或更大目标：拒绝直接外层 IK。
- 任意目标均不得作为默认生产外层 IK；当前只允许同一可信分支下的位置任务诊断。

## 验证用例

{row_lines}

## 结论

该策略没有扩大可发布范围，而是将可发布三维末端位置局部灵敏度进一步约束为方向条件下的诊断接口。强方向可做纳米级闭环候选，弱方向只给诊断提示或拒绝直接推进。完整末端位姿、跨分支复用、跨工作点复用和默认生产外层 IK 仍然禁止。

## 下一步计划

后续若要提升工程实用性，应优先寻找更好的工作点或磁体配置，使弱方向位移效率提高；其次才考虑缓存、多步调度或隐式灵敏度加速。
"""
    (OUT_DIR / "record.md").write_text(record, encoding="utf-8")


def main() -> None:
    """Generate the policy package."""
    t0 = time.perf_counter()
    policy = N2DirectionalEngineeringPolicy(ROOT)
    rows = build_validation_rows(policy)
    failure_rows = [
        {"case": row["case"], "failure_class": "default_outer_ik_would_be_forbidden"}
        for row in rows
        if row["allowed_for_default_outer_ik"]
    ]
    summary = {
        "experiment_id": OUT_DIR.name,
        "status": "directional_engineering_policy_ready" if not failure_rows else "policy_guard_failed",
        "policy_snapshot": policy.as_dict(),
        "validation_cases": len(rows),
        "default_outer_ik_allowed_count": sum(1 for row in rows if row["allowed_for_default_outer_ik"]),
        "action_counts": {action: sum(1 for row in rows if row["action"] == action) for action in sorted({row["action"] for row in rows})},
        "wall_s": time.perf_counter() - t0,
    }
    write_csv(OUT_DIR / "policy_validation_rows.csv", rows)
    write_csv(OUT_DIR / "summary.csv", [{"status": summary["status"], "validation_cases": len(rows), "wall_s": summary["wall_s"]}])
    write_json(OUT_DIR / "summary.json", summary)
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failure_rows),
        encoding="utf-8",
    )
    plot_rows(rows)
    write_record(summary, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
