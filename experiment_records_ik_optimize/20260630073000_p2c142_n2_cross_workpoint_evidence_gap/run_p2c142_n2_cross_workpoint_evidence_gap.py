"""Audit n2 cross-workpoint evidence coverage from existing results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")


N2_SUMMARIES = [
    ROOT / "experiment_records_ik_optimize/20260628150600_p2c89_n2_target_point_continuation/summary.json",
    ROOT / "experiment_records_ik_optimize/20260628153500_p2c90_n2_target_radius_boundary/summary.json",
    ROOT / "experiment_records_ik_optimize/20260628161500_p2c91_n2_target_radius_50um/summary.json",
    ROOT / "experiment_records_ik_optimize/20260628165500_p2c92_n2_target_radius_100um/summary.json",
    ROOT / "experiment_records_ik_optimize/20260628175300_p2c93_n2_target_combo_100um/summary.json",
    ROOT / "experiment_records_ik_optimize/20260628183000_p2c94_n2_target_random_100um/summary.json",
    ROOT / "experiment_records_ik_optimize/20260628211500_p2c97_n2_publish_candidate_synthesis/summary.json",
    ROOT / "experiment_records_ik_optimize/20260629100000_p2c122_n2_workspace_direction_scan/summary.json",
    ROOT / "experiment_records_ik_optimize/20260630043000_p2c136_n2_trust_region_current_reopt/summary.json",
    ROOT / "experiment_records_ik_optimize/20260630053000_p2c137_n2_effective_subspace_release_candidate/summary.json",
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def workpoint_key(payload: dict) -> str:
    case = payload.get("case")
    if case:
        return str(case)
    rows = payload.get("rows")
    if isinstance(rows, list) and rows:
        return str(rows[0].get("case", "unknown"))
    return "unknown"


def classify(payload: dict) -> tuple[str, str]:
    status = str(payload.get("status", payload.get("release_status", "")))
    if "effective_subspace_candidate" in status:
        return "灵敏度发布候选", "优势方向候选"
    if "publish_candidate" in status:
        return "历史发布候选", "缺少后续严格分支半径审计"
    if "target" in status:
        return "目标闭环诊断", "不是跨工作点灵敏度审计"
    if "workspace_direction_scan" in status:
        return "方向空间诊断", "同一工作点的方向可达性"
    if "current_reopt" in payload.get("experiment_id", ""):
        return "扰动重求解", "同一工作点的局部重求解"
    return "其他诊断", "证据口径不足"


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in N2_SUMMARIES:
        payload = load_json(path)
        category, limitation = classify(payload)
        singular_values = payload.get("singular_values_m_per_A")
        condition = payload.get("condition", payload.get("condition_number"))
        rows.append(
            {
                "实验": path.parent.name,
                "工作点": workpoint_key(payload),
                "证据类别": category,
                "状态": str(payload.get("status", payload.get("release_status", ""))),
                "条件数": "" if condition is None else f"{float(condition):.6g}",
                "奇异值": "" if not singular_values else ";".join(f"{float(x):.3e}" for x in singular_values),
                "主要限制": limitation,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_coverage(rows: list[dict[str, str]], path: Path) -> None:
    font = FontProperties(fname=str(FONT_PATH)) if FONT_PATH.exists() else None
    categories = ["目标闭环诊断", "方向空间诊断", "扰动重求解", "灵敏度发布候选"]
    counts = [sum(row["证据类别"] == category for row in rows) for category in categories]
    publishable_workpoints = 1
    required_workpoints = 5

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    axes[0].bar(categories, counts, color=["#8ecae6", "#219ebc", "#ffb703", "#fb8500"])
    axes[0].set_title("n2 既有证据类型", fontproperties=font)
    axes[0].set_ylabel("实验数量", fontproperties=font)
    axes[0].tick_params(axis="x", rotation=25)
    for label in axes[0].get_xticklabels() + axes[0].get_yticklabels():
        label.set_fontproperties(font)

    axes[1].bar(["已有严格候选", "建议最小数量"], [publishable_workpoints, required_workpoints], color=["#2a9d8f", "#e76f51"])
    axes[1].set_ylim(0, required_workpoints + 1)
    axes[1].set_title("跨工作点证据缺口", fontproperties=font)
    axes[1].set_ylabel("工作点数量", fontproperties=font)
    for label in axes[1].get_xticklabels() + axes[1].get_yticklabels():
        label.set_fontproperties(font)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_record(rows: list[dict[str, str]], summary: dict) -> None:
    text = f"""# P2c142 两点磁控跨工作点证据缺口审计

## 实验目的

本轮不新增昂贵求解，目标是复核现有两点磁控 n2 证据是否足以支持“跨工作点稳定灵敏度”结论。审计重点是区分同一工作点内的方向扰动、目标闭环诊断与真正跨工作点灵敏度验证。

## 审计结果

当前严格灵敏度候选集中在 `n2_z_pair_rand12` 单一工作点。P2c122-P2c137 证明该工作点存在一维优势灵敏方向，但不能证明不同目标位置、不同电流基点或不同磁化响应下仍保持同样性质。

既有 P2c89-P2c94 覆盖了目标点延拓、目标半径和随机方向目标闭环，但这些实验主要验证求解器能否回到同一物理解分支，并不等价于跨工作点灵敏度审计。其结果可作为工况设计参考，不能直接作为发布证据。

## 量化缺口

- 严格候选工作点数量：{summary['strict_candidate_workpoint_count']}
- 建议最小工作点数量：{summary['recommended_min_workpoint_count']}
- 当前完整三维位置灵敏度：不可发布
- 当前外层 IK 接入：禁止

## 工程解释

两点磁控当前表现为高度各向异性系统。第一主方向增益约为弱方向的数十到数百倍，因此单一工作点的一维优势方向通过门禁，并不能推出三维任务空间普遍可控。

## 下一步实验设计

下一轮应新增 3-5 个两点磁控工作点，而不是继续在 `n2_z_pair_rand12` 上重复扰动。每个工作点先做基线门禁，再做奇异值分解和第一主方向扰动重求解。若至少 3 个工作点均出现稳定优势方向，可发布“分支条件下有效子空间灵敏度族”；若奇异值谱在不同工作点变化很大，应转向工作空间分区建议。

建议候选工况：

1. 保持两段 z 向相反磁化，改变电流基点。
2. 保持电流基点，改变目标末端位置方向。
3. 使用 x/z 混合磁化，检查有效秩是否改善。
4. 控制单线圈幅值不超过既有安全范围，优先选择末端位移信号可判读的工作点。

## 产物

- `n2_workpoint_evidence_rows.csv`
- `summary.json`
- `n2_cross_workpoint_gap_cn.png`
"""
    (OUT_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    rows = build_rows()
    workpoints = sorted({row["工作点"] for row in rows})
    strict_candidate_workpoints = sorted(
        {row["工作点"] for row in rows if row["证据类别"] == "灵敏度发布候选"}
    )
    summary = {
        "experiment_id": OUT_DIR.name,
        "purpose": "audit_n2_cross_workpoint_evidence_gap",
        "workpoints_seen": workpoints,
        "strict_candidate_workpoints": strict_candidate_workpoints,
        "strict_candidate_workpoint_count": len(strict_candidate_workpoints),
        "recommended_min_workpoint_count": 5,
        "cross_workpoint_publishable": False,
        "full_3d_position_publishable": False,
        "outer_ik_allowed": False,
        "next_step": "run_3_to_5_new_n2_workpoints_with_baseline_gate_then_principal_direction_reopt",
    }
    write_csv(OUT_DIR / "n2_workpoint_evidence_rows.csv", rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_coverage(rows, OUT_DIR / "n2_cross_workpoint_gap_cn.png")
    write_record(rows, summary)


if __name__ == "__main__":
    main()
