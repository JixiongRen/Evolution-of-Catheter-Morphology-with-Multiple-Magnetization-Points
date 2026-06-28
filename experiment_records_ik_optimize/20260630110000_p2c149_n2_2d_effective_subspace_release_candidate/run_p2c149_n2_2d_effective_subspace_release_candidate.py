"""Build the n2 effective-subspace release candidate package."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

P2C144 = ROOT / "experiment_records_ik_optimize/20260630083000_p2c144_n2_candidate_baseline_gate"
P2C145 = ROOT / "experiment_records_ik_optimize/20260630090000_p2c145_n2_candidate_spectrum_audit"
P2C147 = ROOT / "experiment_records_ik_optimize/20260630100000_p2c147_n2_principal_radius_bracket"
P2C148 = ROOT / "experiment_records_ik_optimize/20260630103000_p2c148_n2_all_principal_safe_radius"

SAFE_RADIUS_A = 1.0e-3
LINEARIZATION_GATE = 0.05


def _clean_key(key: str) -> str:
    """Normalize CSV field names produced by spreadsheet-safe writers."""
    return key.replace("\ufeff", "")


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file and normalize its field names."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{_clean_key(k): v for k, v in row.items()} for row in reader]


def read_json(path: Path) -> dict:
    """Read a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: str | float | int) -> float:
    """Convert a numeric table value to float."""
    return float(value)


def as_bool(value: str | bool) -> bool:
    """Convert common CSV boolean text to bool."""
    if isinstance(value, bool):
        return value
    return value.strip().lower() == "true"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows with stable field ordering."""
    if not rows:
        raise ValueError("no rows to write")
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_release_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Merge P2c144-P2c148 evidence into per-workpoint release rows."""
    baseline_rows = {r["direction"]: r for r in read_rows(P2C144 / "candidate_summary.csv")}
    spectrum_rows = {r["direction"]: r for r in read_rows(P2C145 / "spectrum_rows.csv")}
    radius_rows = read_rows(P2C147 / "radius_bracket_rows.csv")
    principal_rows = read_rows(P2C148 / "all_principal_rows.csv")

    radius_by_direction = {}
    for row in radius_rows:
        if abs(as_float(row["radius_A"]) - SAFE_RADIUS_A) < 1e-12:
            radius_by_direction[row["direction"]] = row

    principal_by_direction: dict[str, dict[int, dict[str, str]]] = {}
    for row in principal_rows:
        direction = row["direction"]
        mode = int(row["mode_index"])
        principal_by_direction.setdefault(direction, {})[mode] = row

    release_rows: list[dict[str, object]] = []
    failure_events: list[dict[str, object]] = []
    for direction in sorted(baseline_rows):
        baseline = baseline_rows[direction]
        spectrum = spectrum_rows[direction]
        modes = principal_by_direction[direction]

        mode_pass = {}
        mode_err = {}
        mode_actual_um = {}
        for mode in (1, 2, 3):
            row = modes[mode]
            mode_pass[mode] = row["release_class"] == "通过"
            mode_err[mode] = as_float(row["linear_rel_err"])
            mode_actual_um[mode] = as_float(row["actual_norm_um"])

        baseline_pass = as_bool(baseline["all_steps_gate_ok"])
        spectrum_pass = as_bool(spectrum["baseline_ok"]) and int(spectrum["effective_rank"]) == 3
        two_dim_pass = baseline_pass and spectrum_pass and mode_pass[1] and mode_pass[2] and not mode_pass[3]
        full_3d_pass = baseline_pass and spectrum_pass and all(mode_pass.values())

        row = {
            "direction": direction,
            "baseline_gate_pass": baseline_pass,
            "spectrum_gate_pass": spectrum_pass,
            "task_condition": as_float(spectrum["task_condition"]),
            "effective_rank_from_spectrum": int(spectrum["effective_rank"]),
            "safe_radius_A": SAFE_RADIUS_A,
            "mode1_pass": mode_pass[1],
            "mode2_pass": mode_pass[2],
            "mode3_pass": mode_pass[3],
            "mode1_linearization_error": mode_err[1],
            "mode2_linearization_error": mode_err[2],
            "mode3_linearization_error": mode_err[3],
            "mode1_actual_displacement_um": mode_actual_um[1],
            "mode2_actual_displacement_um": mode_actual_um[2],
            "mode3_actual_displacement_um": mode_actual_um[3],
            "first_principal_safe_radius_pass": radius_by_direction[direction]["release_class"] == "通过",
            "full_3d_position_publishable": full_3d_pass,
            "effective_subspace_dimension": 2 if two_dim_pass else 0,
            "release_class": "二维有效子空间候选" if two_dim_pass else "不可发布",
            "reason": "第一、第二主方向通过；第三主方向弱信号且线性化误差超限"
            if two_dim_pass
            else "证据链未闭合",
        }
        release_rows.append(row)

        if not mode_pass[3]:
            failure_events.append(
                {
                    "direction": direction,
                    "failure_class": "第三主方向弱信号不可发布",
                    "radius_A": SAFE_RADIUS_A,
                    "linearization_error": mode_err[3],
                    "actual_displacement_um": mode_actual_um[3],
                    "gate": f"linearization_error < {LINEARIZATION_GATE}",
                    "interpretation": "第三主方向响应幅值接近数值噪声，不能作为三维位置灵敏度发布依据。",
                }
            )

    return release_rows, failure_events


def plot_release_matrix(rows: list[dict[str, object]]) -> None:
    """Plot pass/fail status by workpoint and principal direction."""
    configure_matplotlib_cjk()
    directions = [str(row["direction"]) for row in rows]
    data = np.array(
        [
            [1 if row["mode1_pass"] else 0, 1 if row["mode2_pass"] else 0, 1 if row["mode3_pass"] else 0]
            for row in rows
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    image = ax.imshow(data, cmap=plt.get_cmap("RdYlGn"), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1, 2], labels=["第一主方向", "第二主方向", "第三主方向"])
    ax.set_yticks(np.arange(len(directions)), labels=directions)
    ax.set_title("两点磁控二维有效子空间证据矩阵")
    ax.set_xlabel("电流扰动主方向")
    ax.set_ylabel("候选工作点")
    for i, row in enumerate(rows):
        for j, key in enumerate(("mode1_linearization_error", "mode2_linearization_error", "mode3_linearization_error")):
            ok = data[i, j] > 0.5
            text = "通过" if ok else "不发布"
            err = float(row[key])
            ax.text(j, i, f"{text}\n误差={err:.3g}", ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(image, ax=ax, shrink=0.85)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["不发布", "通过"])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_2d_effective_subspace_release_candidate_cn.png", dpi=200)
    plt.close(fig)


def write_record(rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    """Write the Chinese experiment record."""
    lines = [
        "# P2c149 两点磁控二维有效子空间发布候选证据综合",
        "",
        "## 实验目的",
        "",
        "本轮不重新求解物理模型，而是综合 P2c144 至 P2c148 的既有证据，判断两点磁控导管是否已经形成可追溯的局部位置灵敏度发布候选。重点区分三维末端位置灵敏度与二维有效子空间灵敏度，避免把弱信号方向误报为完整三维可控。",
        "",
        "## 数据来源",
        "",
        "- P2c144：候选工作点基线门禁，检查一阶最优性、载荷传播、分支回返和中心线漂移。",
        "- P2c145：候选工作点谱结构审计，检查有效秩和条件数。",
        "- P2c147：第一主方向扰动半径夹逼，确认 0.001 A 为当前安全半径。",
        "- P2c148：三个主方向在 0.001 A 下的扰动重求解一致性审计。",
        "",
        "## 判定口径",
        "",
        "二维有效子空间指局部灵敏度矩阵中前两个主响应方向张成的任务子空间。它不是全局固定的 x-y 平面，也不是完整三维末端位置控制结论，而是在当前物理解分支和小扰动半径内成立的局部线性近似。",
        "",
        "发布候选必须满足：基线门禁通过、谱审计有效秩不低于三、第一和第二主方向扰动重求解通过、第三主方向不被强行发布。",
        "",
        "## 结果",
        "",
        f"- 候选工作点总数：{summary['candidate_count']}。",
        f"- 二维有效子空间候选数：{summary['n2_2d_effective_subspace_candidate_count']}。",
        f"- 完整三维位置可发布候选数：{summary['full_3d_position_publishable_count']}。",
        f"- 当前安全扰动半径：{summary['safe_radius_A']} A。",
        "",
        "| 工作点 | 条件数 | 第一主方向 | 第二主方向 | 第三主方向 | 发布类别 |",
        "|---|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {direction} | {condition:.2f} | {m1} | {m2} | {m3} | {cls} |".format(
                direction=row["direction"],
                condition=float(row["task_condition"]),
                m1="通过" if row["mode1_pass"] else "不发布",
                m2="通过" if row["mode2_pass"] else "不发布",
                m3="通过" if row["mode3_pass"] else "不发布",
                cls=row["release_class"],
            )
        )
    lines.extend(
        [
            "",
            "## 数据分析",
            "",
            "五个候选工作点均通过基线门禁和谱审计，说明这些工况不是一阶最优性退化或条件数病态导致的假阳性。P2c148 进一步显示，第一和第二主方向在 0.001 A 扰动下均能通过局部线性化门禁，而第三主方向全部失败。",
            "",
            "第三主方向失败的主要原因不是分支大幅漂移，而是任务响应幅值过小。其末端位移量约处于纳米量级到亚纳米量级，接近当前数值求解和重优化误差背景，因此相对线性化误差会被放大。该方向应解释为弱信号或不可辨识方向，而不是三维位置灵敏度已经失败。",
            "",
            "## 实验结论",
            "",
            "当前两点磁控结果尚不能支持完整三维末端位置灵敏度发布，但已经形成五个工作点上的二维有效子空间候选证据链。该结论是分支条件下、0.001 A 小扰动半径内的局部结论，不能外推为全局灵敏度。",
            "",
            "## 工程建议",
            "",
            "- 两点磁控可优先接入二维有效子空间诊断接口，但暂不接入完整三维外层 IK。",
            "- 外层控制若使用该结果，必须限制在可信物理解分支和 0.001 A 量级的小步长内。",
            "- 第三主方向不应作为可控方向使用；若必须获得三维能力，需要改变磁化配置、工作点或线圈电流组合。",
            "",
            "## 下一步计划",
            "",
            "下一轮应把二维有效子空间接口形式固定下来，包括主方向基、有效任务投影、步长限制、失败回退规则和跨工作点复核。随后再判断是否值得继续寻找三维可发布工作点，或转入单点磁控不可发布边界归因。",
        ]
    )
    (OUT_DIR / "record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Generate all P2c149 artifacts."""
    rows, failure_events = build_release_rows()
    full_3d_count = sum(1 for row in rows if row["full_3d_position_publishable"])
    two_dim_count = sum(1 for row in rows if row["release_class"] == "二维有效子空间候选")
    summary = {
        "experiment_id": "20260630110000_p2c149_n2_2d_effective_subspace_release_candidate",
        "candidate_count": len(rows),
        "n2_2d_effective_subspace_candidate_count": two_dim_count,
        "full_3d_position_publishable_count": full_3d_count,
        "full_3d_position_publishable": full_3d_count > 0,
        "effective_subspace_dimension": 2 if two_dim_count == len(rows) else None,
        "safe_radius_A": SAFE_RADIUS_A,
        "outer_ik_allowed": False,
        "local_2d_interface_candidate": two_dim_count == len(rows),
        "reason": "五个候选工作点前两主方向通过，第三主方向弱信号且线性化误差超限。",
        "source_experiments": [
            str(P2C144.relative_to(ROOT)),
            str(P2C145.relative_to(ROOT)),
            str(P2C147.relative_to(ROOT)),
            str(P2C148.relative_to(ROOT)),
        ],
    }

    write_csv(OUT_DIR / "release_candidate_rows.csv", rows)
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in failure_events),
        encoding="utf-8",
    )
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_release_matrix(rows)
    write_record(rows, summary)


if __name__ == "__main__":
    main()
