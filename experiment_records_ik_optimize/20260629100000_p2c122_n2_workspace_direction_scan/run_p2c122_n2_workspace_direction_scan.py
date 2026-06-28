"""Scan the directional workspace implied by the trusted n2 sensitivity."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
P2C119_DIR = ROOT / "experiment_records_ik_optimize/20260629080000_p2c119_n2_diagnostic_outer_ik_policy"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(P2C119_DIR) not in sys.path:
    sys.path.insert(0, str(P2C119_DIR))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk  # noqa: E402
from n2_diagnostic_outer_ik_policy import N2DiagnosticOuterIKPolicy  # noqa: E402


TARGET_RADII_M = (5.0e-8, 5.0e-7, 5.0e-6)
FAST_STEP_LIMIT = 4
DIAGNOSTIC_STEP_LIMIT = 8


def write_json(path: Path, data: dict) -> None:
    """Write stable JSON output."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write CSV rows."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def fibonacci_sphere(n: int) -> np.ndarray:
    """Return near-uniform unit vectors on a sphere."""
    points = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (i / max(n - 1, 1)) * 2.0
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))
    return np.asarray(points, dtype=float)


def named_directions(U: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Build named Cartesian and singular-vector directions."""
    base = [
        ("x+", np.array([1.0, 0.0, 0.0])),
        ("x-", np.array([-1.0, 0.0, 0.0])),
        ("y+", np.array([0.0, 1.0, 0.0])),
        ("y-", np.array([0.0, -1.0, 0.0])),
        ("z+", np.array([0.0, 0.0, 1.0])),
        ("z-", np.array([0.0, 0.0, -1.0])),
    ]
    for idx in range(U.shape[1]):
        base.append((f"主灵敏方向{idx + 1}+", U[:, idx]))
        base.append((f"主灵敏方向{idx + 1}-", -U[:, idx]))
    return base


def classify_steps(required_steps: int, weak_fraction: float) -> str:
    """Classify one direction using the same thresholds as the diagnostic policy."""
    if required_steps <= FAST_STEP_LIMIT and weak_fraction < 0.25:
        return "快速局部校正"
    if required_steps <= DIAGNOSTIC_STEP_LIMIT and weak_fraction < 0.60:
        return "慢速诊断推进"
    return "拒绝直接外层 IK"


def build_direction_row(
    name: str,
    direction: np.ndarray,
    S_pinv: np.ndarray,
    U: np.ndarray,
    radius_A: float,
    singular_values: np.ndarray,
) -> dict:
    """Build metrics for one task-space direction."""
    direction = np.asarray(direction, dtype=float).reshape(3)
    direction = direction / max(float(np.linalg.norm(direction)), 1.0e-18)
    current_per_meter = float(np.linalg.norm(S_pinv @ direction))
    one_step_reach_m = radius_A / max(current_per_meter, 1.0e-18)
    coeff = U.T @ direction
    weak_fraction = float(abs(coeff[-1]) / max(float(np.linalg.norm(coeff)), 1.0e-18))
    strongest_fraction = float(abs(coeff[0]) / max(float(np.linalg.norm(coeff)), 1.0e-18))
    condition_direction = float(current_per_meter * singular_values[0])
    row = {
        "direction": name,
        "ux": float(direction[0]),
        "uy": float(direction[1]),
        "uz": float(direction[2]),
        "current_per_meter_A_per_m": current_per_meter,
        "one_step_reach_m": one_step_reach_m,
        "one_step_reach_um": one_step_reach_m * 1.0e6,
        "strong_direction_fraction": strongest_fraction,
        "weak_direction_fraction": weak_fraction,
        "directional_condition_proxy": condition_direction,
    }
    for radius in TARGET_RADII_M:
        steps = int(math.ceil(radius / max(one_step_reach_m, 1.0e-18)))
        row[f"steps_for_{radius:.0e}_m"] = steps
        row[f"action_for_{radius:.0e}_m"] = classify_steps(steps, weak_fraction)
    return row


def build_rows(policy: N2DiagnosticOuterIKPolicy) -> tuple[list[dict], list[dict]]:
    """Build named and sphere direction scan rows."""
    S = policy.S
    U = policy.U
    singular_values = policy.singular_values
    S_pinv = np.linalg.pinv(S, rcond=1.0e-12)
    named_rows = [
        build_direction_row(name, direction, S_pinv, U, policy.radius_A, singular_values)
        for name, direction in named_directions(U)
    ]
    sphere_rows = [
        build_direction_row(f"sphere_{idx:03d}", direction, S_pinv, U, policy.radius_A, singular_values)
        for idx, direction in enumerate(fibonacci_sphere(240))
    ]
    return named_rows, sphere_rows


def summarize(named_rows: list[dict], sphere_rows: list[dict], policy: N2DiagnosticOuterIKPolicy, wall_s: float) -> dict:
    """Summarize direction scan results."""
    all_rows = named_rows + sphere_rows
    reaches = np.asarray([row["one_step_reach_um"] for row in all_rows], dtype=float)
    actions_by_radius = {}
    for radius in TARGET_RADII_M:
        key = f"action_for_{radius:.0e}_m"
        counts: dict[str, int] = {}
        for row in all_rows:
            counts[row[key]] = counts.get(row[key], 0) + 1
        actions_by_radius[f"{radius:.0e}_m"] = counts
    best = max(all_rows, key=lambda row: row["one_step_reach_m"])
    worst = min(all_rows, key=lambda row: row["one_step_reach_m"])
    return {
        "experiment_id": OUT_DIR.name,
        "status": "n2_workspace_direction_scan_done",
        "task_scope": "两点磁控可信分支下的三维末端位置局部导数方向扫描",
        "publish_scope": "仅用于分支条件下局部位置灵敏度解释；不构成全局 IK 或完整位姿灵敏度",
        "max_linear_current_step_A": policy.radius_A,
        "singular_values_m_per_A": policy.singular_values.tolist(),
        "condition_number": float(policy.singular_values[0] / policy.singular_values[-1]),
        "one_step_reach_um": {
            "min": float(np.min(reaches)),
            "median": float(np.median(reaches)),
            "max": float(np.max(reaches)),
        },
        "best_direction": {
            "direction": best["direction"],
            "one_step_reach_um": best["one_step_reach_um"],
        },
        "worst_direction": {
            "direction": worst["direction"],
            "one_step_reach_um": worst["one_step_reach_um"],
        },
        "actions_by_target_radius": actions_by_radius,
        "interpretation": "该工作点方向性极强。可信线性半径内，强方向也主要只适合纳米级诊断推进；微米级目标多数方向应拒绝直接外层 IK。",
        "wall_s": wall_s,
    }


def plot_rows(named_rows: list[dict], sphere_rows: list[dict], summary: dict) -> None:
    """Plot directional reach and action classes."""
    configure_matplotlib_cjk()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    reaches = np.asarray([row["one_step_reach_um"] for row in sphere_rows], dtype=float)
    axes[0].hist(reaches, bins=32, color="#356f8f", alpha=0.85)
    axes[0].set_xlabel("单步可信电流半径内的方向位移上限（微米）")
    axes[0].set_ylabel("方向数量")
    axes[0].set_title("两点磁控局部方向可达性分布")
    axes[0].grid(alpha=0.25)

    labels = [row["direction"] for row in named_rows]
    values = [row["one_step_reach_um"] for row in named_rows]
    axes[1].barh(labels, values, color="#8f4f35", alpha=0.85)
    axes[1].set_xlabel("单步方向位移上限（微米）")
    axes[1].set_title("坐标轴与主灵敏方向对比")
    axes[1].grid(axis="x", alpha=0.25)

    fig.suptitle("n2 工作空间方向扫描：可信局部灵敏度边界")
    fig.savefig(OUT_DIR / "n2_workspace_direction_scan_cn.png", dpi=220)
    plt.close(fig)

    radii = list(summary["actions_by_target_radius"].keys())
    actions = ["快速局部校正", "慢速诊断推进", "拒绝直接外层 IK"]
    matrix = np.asarray(
        [[summary["actions_by_target_radius"][radius].get(action, 0) for radius in radii] for action in actions],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    bottom = np.zeros(len(radii))
    colors = ["#2b8cbe", "#a6bddb", "#d95f0e"]
    for action, values, color in zip(actions, matrix, colors, strict=True):
        ax.bar(radii, values, bottom=bottom, label=action, color=color, alpha=0.88)
        bottom += values
    ax.set_ylabel("方向数量")
    ax.set_xlabel("目标位移尺度")
    ax.set_title("不同目标尺度下的方向策略分类")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUT_DIR / "n2_workspace_direction_actions_cn.png", dpi=220)
    plt.close(fig)


def write_record(summary: dict, named_rows: list[dict]) -> None:
    """Write the Chinese experiment record."""
    named_lines = "\n".join(
        f"- {row['direction']}：单步上限 {row['one_step_reach_um']:.4g} 微米；弱方向占比 {row['weak_direction_fraction']:.3f}。"
        for row in named_rows
    )
    actions = "\n".join(
        f"- 目标尺度 {radius}：{counts}。"
        for radius, counts in summary["actions_by_target_radius"].items()
    )
    record = f"""# P2c122 两点磁控导管工作空间方向扫描报告

## 实验目的

本轮基于 P2c117 和 P2c119 已通过门禁的两点磁控局部位置灵敏度，扫描末端位置空间中的方向可控性。目标是明确哪些方向适合局部诊断推进，哪些方向因电流需求过大或弱可控成分过高而不应直接接入外层逆运动学。

## 实验方案

本轮不重新求解物理打靶问题，而是读取可信分支缓存中的三维末端位置对八路线圈电流的局部导数矩阵。设局部导数为 \(S \\in \\mathbb{{R}}^{{3\\times 8}}\)，目标方向单位向量为 \(u\)。实现该方向单位位移所需的最小范数电流近似为：

\\[
\\Delta I = S^+ u,
\\]

其中 \(S^+\) 为 Moore-Penrose 广义逆。若单步线性信赖域半径为 \(r_I=0.0125\\,\\mathrm{{A}}\)，则该方向单步可信位移上限为：

\\[
\\delta x_{{\\max}}(u)=\\frac{{r_I}}{{\\|S^+u\\|_2}}.
\\]

该指标只描述同一可信分支、同一工作点附近的局部方向性，不表示全局可达空间。

## 关键结果

- 矩阵条件数：{summary['condition_number']:.3g}。
- 单步方向位移上限：最小 {summary['one_step_reach_um']['min']:.4g} 微米，中位数 {summary['one_step_reach_um']['median']:.4g} 微米，最大 {summary['one_step_reach_um']['max']:.4g} 微米。
- 最强方向：{summary['best_direction']['direction']}，单步上限 {summary['best_direction']['one_step_reach_um']:.4g} 微米。
- 最弱方向：{summary['worst_direction']['direction']}，单步上限 {summary['worst_direction']['one_step_reach_um']:.4g} 微米。

## 坐标轴与主方向

{named_lines}

## 策略分类

{actions}

## 结论

两点磁控导管当前工作点的局部位置灵敏度具有明显各向异性。强方向主要只能用于纳米级诊断修正；对亚微米和微米级目标，绝大多数方向需要大量小步或应拒绝直接外层逆运动学。该结论强化了前一轮判断：当前结果适合作为分支条件下的局部诊断灵敏度，不适合作为默认生产逆运动学模型。

## 下一步计划

下一轮应选择两类代表方向做真实重求解闭环：一类为主灵敏方向，另一类为弱方向。若强方向闭环稳定、弱方向失败，应形成“局部可控锥”工程边界；若强方向也失败，则回到物理闭合或分支稳定性审计。
"""
    (OUT_DIR / "record.md").write_text(record, encoding="utf-8")


def main() -> None:
    """Generate the n2 workspace direction scan package."""
    t0 = time.perf_counter()
    policy = N2DiagnosticOuterIKPolicy(ROOT)
    named_rows, sphere_rows = build_rows(policy)
    wall_s = time.perf_counter() - t0
    summary = summarize(named_rows, sphere_rows, policy, wall_s)

    write_csv(OUT_DIR / "named_direction_rows.csv", named_rows)
    write_csv(OUT_DIR / "sphere_direction_rows.csv", sphere_rows)
    write_csv(
        OUT_DIR / "summary.csv",
        [
            {
                "status": summary["status"],
                "condition_number": summary["condition_number"],
                "one_step_reach_min_um": summary["one_step_reach_um"]["min"],
                "one_step_reach_median_um": summary["one_step_reach_um"]["median"],
                "one_step_reach_max_um": summary["one_step_reach_um"]["max"],
                "wall_s": summary["wall_s"],
            }
        ],
    )
    write_json(OUT_DIR / "summary.json", summary)
    (OUT_DIR / "failure_events.jsonl").write_text("", encoding="utf-8")
    plot_rows(named_rows, sphere_rows, summary)
    write_record(summary, named_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
