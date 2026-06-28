"""Build a release-readiness audit from existing IK sensitivity experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")


SOURCE_SUMMARIES = {
    "n2_effective_subspace": ROOT
    / "experiment_records_ik_optimize/20260630053000_p2c137_n2_effective_subspace_release_candidate/summary.json",
    "n1_boundary": ROOT
    / "experiment_records_ik_optimize/20260630060000_p2c138_n1_effective_subspace_boundary_package/summary.json",
    "n1_branch_recovery": ROOT
    / "experiment_records_ik_optimize/20260630063000_p2c139_n1_mid_branch_recovery_probe/summary.json",
    "n1_current_reopt": ROOT
    / "experiment_records_ik_optimize/20260629003000_p2c140_n1_continuation_branch_current_reopt/summary.json",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def bool_cn(value: bool) -> str:
    return "通过" if value else "未通过"


def build_rows(data: dict[str, dict]) -> list[dict[str, str]]:
    n2 = data["n2_effective_subspace"]
    n1_boundary = data["n1_boundary"]
    n1_branch = data["n1_branch_recovery"]
    n1_reopt = data["n1_current_reopt"]

    n2_release_count = int(n2.get("release_direction_count", 0))
    n2_effective_ok = n2.get("release_status") == "effective_subspace_candidate"
    n2_full3d_ok = bool(n2.get("full_3d_position_publishable", False))
    n2_outer_ok = bool(n2.get("outer_ik_allowed", False))

    n1_branch_ok = n1_branch.get("continuation_branch_status") == "通过"
    n1_signal_ok = int(n1_reopt.get("low_signal_direction_count", 0)) == 0
    n1_release_count = int(n1_reopt.get("release_direction_count", 0))
    n1_effective_ok = n1_reopt.get("release_status") != "not_publishable" and n1_signal_ok

    return [
        {
            "对象": "两点磁控 n2",
            "任务层级": "末端位置有效子空间",
            "当前结论": "候选可发布" if n2_effective_ok else "不可发布",
            "可发布方向数": str(n2_release_count),
            "完整三维位置": bool_cn(n2_full3d_ok),
            "分支可信": "局部可信，半径约 1e-4",
            "主要缺口": "只支持第一主灵敏方向；完整三维位置仍不可发布",
            "外层IK接入": bool_cn(n2_outer_ok),
            "证据来源": SOURCE_SUMMARIES["n2_effective_subspace"].name,
        },
        {
            "对象": "单点磁控 n1",
            "任务层级": "末端位置有效子空间",
            "当前结论": "候选可发布" if n1_effective_ok else "不可发布",
            "可发布方向数": str(n1_release_count),
            "完整三维位置": bool_cn(False),
            "分支可信": bool_cn(n1_branch_ok),
            "主要缺口": "连续分支可恢复，但电流扰动引起的末端位移低于可靠判读阈值",
            "外层IK接入": bool_cn(bool(n1_reopt.get("outer_ik_allowed", False))),
            "证据来源": ",".join(
                [
                    SOURCE_SUMMARIES["n1_boundary"].name,
                    SOURCE_SUMMARIES["n1_branch_recovery"].name,
                    SOURCE_SUMMARIES["n1_current_reopt"].name,
                ]
            ),
        },
    ]


def build_gate_rows(data: dict[str, dict]) -> list[dict[str, str]]:
    n2 = data["n2_effective_subspace"]
    n1_branch = data["n1_branch_recovery"]
    n1_reopt = data["n1_current_reopt"]
    return [
        {
            "对象": "两点磁控 n2",
            "物理解分支": "通过",
            "有效秩": "仅第一主方向可用",
            "局部线性化": "第一主方向通过",
            "扰动重求解一致性": "局部通过",
            "输出信号量": "可判读",
            "条件数": f"{float(n2.get('condition', 0.0)):.3g}",
            "发布等级": "分支条件下局部导数候选",
        },
        {
            "对象": "单点磁控 n1",
            "物理解分支": bool_cn(n1_branch.get("continuation_branch_status") == "通过"),
            "有效秩": "未形成可发布方向",
            "局部线性化": "受低信号限制，不判为通过",
            "扰动重求解一致性": "数学上局部稳定，但物理信号过小",
            "输出信号量": f"未通过，低于 {n1_reopt.get('min_reliable_tip_delta_m', 1e-10):.1e} m",
            "条件数": "约 3.7e6",
            "发布等级": "不可发布",
        },
    ]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_readiness(data: dict[str, dict], path: Path) -> None:
    font = FontProperties(fname=str(FONT_PATH)) if FONT_PATH.exists() else None
    labels = ["物理解分支", "有效秩", "局部线性化", "扰动重求解", "输出信号", "外层接入"]
    n2_scores = [1.0, 0.45, 0.55, 0.55, 0.8, 0.0]
    n1_scores = [0.75, 0.0, 0.0, 0.25, 0.0, 0.0]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = range(len(labels))
    width = 0.36
    ax.bar([i - width / 2 for i in x], n2_scores, width, label="两点磁控 n2")
    ax.bar([i + width / 2 for i in x], n1_scores, width, label="单点磁控 n1")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("证据成熟度", fontproperties=font)
    ax.set_title("IK 灵敏度发布证据矩阵", fontproperties=font)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontproperties=font, rotation=20, ha="right")
    ax.legend(prop=font)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_record(data: dict[str, dict], release_rows: list[dict[str, str]], gate_rows: list[dict[str, str]]) -> None:
    n1 = data["n1_current_reopt"]
    n2 = data["n2_effective_subspace"]
    text = f"""# P2c141 IK 灵敏度发布证据矩阵审计

## 实验目的

本轮不新增昂贵求解，目标是汇总 P2c137-P2c140 的当前证据，明确单点与两点磁控导管灵敏度研究的发布边界、主要缺口和下一步优先级。该审计用于防止仅凭末端拟合图像误判灵敏度可信性。

## 数据来源

- 两点磁控：`20260630053000_p2c137_n2_effective_subspace_release_candidate`
- 单点磁控边界：`20260630060000_p2c138_n1_effective_subspace_boundary_package`
- 单点分支恢复：`20260630063000_p2c139_n1_mid_branch_recovery_probe`
- 单点电流扰动重求解：`20260629003000_p2c140_n1_continuation_branch_current_reopt`

## 主要结果

| 对象 | 当前结论 | 可发布方向数 | 主要限制 |
|---|---:|---:|---|
| 两点磁控 n2 | {release_rows[0]['当前结论']} | {release_rows[0]['可发布方向数']} | {release_rows[0]['主要缺口']} |
| 单点磁控 n1 | {release_rows[1]['当前结论']} | {release_rows[1]['可发布方向数']} | {release_rows[1]['主要缺口']} |

两点磁控目前只支持“分支条件下第一主灵敏方向的局部导数候选”。其条件数约为 `{float(n2.get('condition', 0.0)):.3g}`，但完整三维末端位置灵敏度仍不可发布，也不能接入外层 IK。

单点磁控在连续加载路径下可以恢复一致分支，但 P2c140 显示所有电流方向的末端位移均低于 `{float(n1.get('min_reliable_tip_delta_m', 1e-10)):.1e} m` 的可靠判读阈值。因此，当前工作点不能发布有效子空间灵敏度。

## 失败机制归因

两点磁控的主要问题不是末端拟合，而是可控方向不足。当前证据仅支持一维有效子空间；若强行发布三维位置灵敏度，会把病态弱方向解释为真实控制能力。

单点磁控的主要问题不是分支不可恢复，而是输出信号量过小。该现象说明在当前磁化配置、工作点和电流扰动范围下，末端位置响应低于数值和物理判读下限。

## 下一步计划

1. 两点磁控：优先做第一主方向的重复工况验证，确认该方向不是偶然样本；同时探索能否通过磁化配置或工作点选择扩展到二维有效子空间。
2. 单点磁控：停止在当前工作点继续扰动扫描，转向工作点与磁化配置筛选，优先寻找末端位移信号超过可靠阈值的样本。
3. 外层 IK：当前不允许接入。只有当某一任务层级同时通过物理解分支、局部线性化、扰动重求解和信号量门禁后，才进入外层 IK 接口验证。

## 产物

- `release_matrix.csv`
- `gate_matrix.csv`
- `summary.json`
- `ik_sensitivity_release_evidence_cn.png`
"""
    (OUT_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    data = {name: load_json(path) for name, path in SOURCE_SUMMARIES.items()}
    release_rows = build_rows(data)
    gate_rows = build_gate_rows(data)

    write_csv(OUT_DIR / "release_matrix.csv", release_rows)
    write_csv(OUT_DIR / "gate_matrix.csv", gate_rows)
    plot_readiness(data, OUT_DIR / "ik_sensitivity_release_evidence_cn.png")

    summary = {
        "experiment_id": OUT_DIR.name,
        "purpose": "audit_release_readiness_from_existing_ik_sensitivity_evidence",
        "n2_status": "effective_subspace_candidate",
        "n2_release_direction_count": int(data["n2_effective_subspace"].get("release_direction_count", 0)),
        "n2_full_3d_position_publishable": bool(
            data["n2_effective_subspace"].get("full_3d_position_publishable", False)
        ),
        "n1_status": "not_publishable",
        "n1_release_direction_count": int(data["n1_current_reopt"].get("release_direction_count", 0)),
        "n1_low_signal_direction_count": int(data["n1_current_reopt"].get("low_signal_direction_count", 0)),
        "outer_ik_allowed": False,
        "next_priority": [
            "repeat_n2_first_principal_direction_on_more_workpoints",
            "search_n1_workpoints_or_magnetization_with_observable_tip_response",
            "do_not_connect_outer_ik_before_all_gates_pass",
        ],
        "source_summaries": {name: str(path.relative_to(ROOT)) for name, path in SOURCE_SUMMARIES.items()},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_record(data, release_rows, gate_rows)


if __name__ == "__main__":
    main()
