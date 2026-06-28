from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent

SOURCES = {
    "P2c145": "20260630090000_p2c145_n2_candidate_spectrum_audit",
    "P2c148": "20260630103000_p2c148_n2_all_principal_safe_radius",
    "P2c149": "20260630110000_p2c149_n2_2d_effective_subspace_release_candidate",
    "P2c151": "20260630120000_p2c151_n2_numeric_subspace_basis",
    "P2c152": "20260630123000_p2c152_n2_2d_subspace_closure",
    "P2c154": "20260630133000_p2c154_n2_2d_cross_radius_closure",
    "P2c156": "20260630143000_p2c156_n2_2d_multistep_prototype",
    "P2c158": "20260630153000_p2c158_n2_engineering_boundary_synthesis",
}


def load_source(name: str) -> dict:
    path = ROOT / "experiment_records_ik_optimize" / SOURCES[name] / "summary.json"
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    data["_source_path"] = str(path.relative_to(ROOT))
    return data


def configure_chinese_font() -> None:
    candidates = [
        ROOT / "assets" / "fonts" / "SourceHanSerifCN-Regular.otf",
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    ]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            prop = font_manager.FontProperties(fname=str(path))
            plt.rcParams["font.family"] = prop.get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    data = {name: load_source(name) for name in SOURCES}

    spectrum_rows = data["P2c145"].get("rows", [])
    radius_rows = data["P2c148"].get("rows", [])
    mode3_rows = [r for r in radius_rows if r.get("mode_index") == 3]
    mode12_rows = [r for r in radius_rows if r.get("mode_index") in (1, 2)]
    mode12_pass = [r for r in mode12_rows if r.get("release_class") == "通过"]
    mode3_pass = [r for r in mode3_rows if r.get("release_class") == "通过"]

    sigma3_ratios = [r.get("sigma_3_over_sigma_1") for r in spectrum_rows]
    mode3_errors = [r.get("linear_rel_err") for r in mode3_rows]
    mode12_errors = [r.get("linear_rel_err") for r in mode12_rows]

    summary = {
        "experiment_id": "20260630170000_p2c160_n2_3d_position_boundary_synthesis",
        "status": "n2_full_3d_position_not_publishable_2d_single_step_diagnostic_only",
        "n2_full_3d_position_publishable": False,
        "n2_effective_2d_single_step_publishable_as_diagnostic": True,
        "outer_ik_allowed": False,
        "multistep_allowed": False,
        "full_pose_publishable": False,
        "candidate_workpoint_count": data["P2c149"].get("candidate_count", 5),
        "effective_subspace_dimension": 2,
        "conservative_radius_A": data["P2c154"].get("conservative_passed_radius_A", 0.001),
        "mode12_pass_count": len(mode12_pass),
        "mode12_test_count": len(mode12_rows),
        "mode3_pass_count": len(mode3_pass),
        "mode3_test_count": len(mode3_rows),
        "sigma3_over_sigma1_min": min(sigma3_ratios),
        "sigma3_over_sigma1_max": max(sigma3_ratios),
        "mode12_linear_error_max": max(mode12_errors),
        "mode3_linear_error_min": min(mode3_errors),
        "mode3_linear_error_max": max(mode3_errors),
        "failure_boundary": {
            "full_3d": "第三主方向信号幅值过弱，扰动重求解线性化误差超限，不能发布完整三维末端位置灵敏度。",
            "two_dimensional": "前两主方向在 0.001 A 半径内通过重求解闭合，可作为分支条件下二维单步诊断灵敏度。",
            "large_radius": "0.0015 A 出现接口连续残差超限。",
            "multistep": "固定局部二维子空间第二步出现接口连续残差超限。",
        },
        "source_summaries": {k: v["_source_path"] for k, v in data.items()},
    }
    write_json(EXP_DIR / "summary.json", summary)

    decision_rows = [
        {
            "scope": "两点磁控完整三维末端位置灵敏度",
            "decision": "不可发布",
            "allowed_use": "不可辨识边界说明",
            "forbidden_use": "外层 IK、三维控制律、完整任务灵敏度发布",
            "reason": "第三主方向弱信号且线性化误差超限",
        },
        {
            "scope": "两点磁控二维有效子空间单步灵敏度",
            "decision": "可作为诊断候选",
            "allowed_use": "指定工作点、指定分支、0.001 A 半径内单步诊断",
            "forbidden_use": "跨工作点复用、多步固定子空间迭代、完整三维任务替代",
            "reason": "五个工作点前两主方向重求解闭合均通过",
        },
        {
            "scope": "两点磁控固定子空间多步迭代",
            "decision": "不可发布",
            "allowed_use": "失败机制分析",
            "forbidden_use": "外层 IK 默认迭代",
            "reason": "第二步接口连续残差超限",
        },
    ]
    write_csv(EXP_DIR / "engineering_decision_rows.csv", decision_rows)

    failure_events = [
        {
            "failure_class": "有效秩不足",
            "evidence": "P2c145/P2c148/P2c149",
            "interpretation": "第三主方向相对第一主方向仅约 0.25% 至 0.50%，有效可控空间应降为二维。",
            "severity": "hard_stop_for_3d",
        },
        {
            "failure_class": "局部线性化误差超限",
            "evidence": "P2c148",
            "interpretation": "第三主方向在扰动重求解中误差超过 0.05 门限，不能用于发布三维导数。",
            "severity": "hard_stop_for_3d",
        },
        {
            "failure_class": "接口连续残差超限",
            "evidence": "P2c154/P2c156",
            "interpretation": "较大半径和固定子空间第二步破坏接口连续性，说明当前结果只支持局部单步诊断。",
            "severity": "hard_stop_for_outer_ik",
        },
    ]
    write_jsonl(EXP_DIR / "failure_events.jsonl", failure_events)

    configure_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(["第一/二主方向", "第三主方向"], [len(mode12_pass), len(mode3_pass)], color=["#4C78A8", "#E45756"])
    axes[0].set_ylabel("通过数量")
    axes[0].set_title("扰动重求解通过情况")
    axes[0].set_ylim(0, max(len(mode12_rows), 1) + 1)
    axes[1].boxplot([mode12_errors, mode3_errors], tick_labels=["第一/二主方向", "第三主方向"])
    axes[1].axhline(0.05, color="#E45756", linestyle="--", linewidth=1.5, label="线性化门限")
    axes[1].set_ylabel("局部线性化误差")
    axes[1].set_title("线性化误差分布")
    axes[1].legend()
    fig.suptitle("两点磁控 n2 三维位置边界综合")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "n2_3d_position_boundary_synthesis_cn.png", dpi=200)
    plt.close(fig)

    record = f"""# P2c160 两点磁控 n2 三维末端位置边界综合

## 实验目的

本轮综合既有 n2 证据，判断两点磁控导管是否已经支持完整三维末端位置灵敏度发布，并明确二维有效子空间结果的工程边界。

## 证据基础

P2c145 表明 5 个候选工作点均通过基线谱检查，但第三主方向相对第一主方向很弱。P2c148 对三个主方向做扰动重求解，前两主方向通过，第三主方向因局部线性化误差超限失败。P2c152 证明二维子空间 15 个单步样本全部通过。P2c154 证明保守电流半径为 0.001 A。P2c156 证明固定二维子空间不能直接用于多步迭代。

## 主要数值结论

- 第三主方向与第一主方向的奇异值比值范围为 `{summary['sigma3_over_sigma1_min']:.4g}` 至 `{summary['sigma3_over_sigma1_max']:.4g}`。
- 第一、二主方向通过 `{len(mode12_pass)}/{len(mode12_rows)}` 个扰动重求解样本。
- 第三主方向通过 `{len(mode3_pass)}/{len(mode3_rows)}` 个扰动重求解样本。
- 二维子空间单步闭合最大线性化误差为 `{data['P2c152'].get('max_linear_rel_err'):.4g}`。
- 固定二维子空间第二步通过数为 `0/20`。

## 工程结论

当前证据不支持发布两点磁控完整三维末端位置灵敏度。失败原因不是末端拟合不足，而是第三主方向信号弱且重求解线性化误差超限，因此完整三维任务存在不可辨识方向。

当前可保留的结果是：在指定工作点、指定分支、0.001 A 半径内，二维有效子空间单步末端位置灵敏度可作为诊断候选。该结果不能替代完整三维灵敏度，不能跨工作点复用，也不能直接接入外层 IK。

## 下一步计划

若目标是获得可发布三维末端位置灵敏度，应优先改变磁体配置或工作点选择，使第三主方向响应幅值提高。若目标是工程推进，应优先实现滚动局部分支重建：每一步重新求解当前分支和二维子空间，而不是复用固定子空间做多步迭代。
"""
    (EXP_DIR / "record.md").write_text(record, encoding="utf-8")

    commands = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/20260630170000_p2c160_n2_3d_position_boundary_synthesis/run_p2c160_n2_3d_position_boundary_synthesis.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")


if __name__ == "__main__":
    main()
