from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
P2C163 = ROOT / "experiment_records_ik_optimize/20260630183000_p2c163_n2_second_step_radius_block_audit"


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


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    p2c163_summary = json.loads((P2C163 / "summary.json").read_text(encoding="utf-8"))
    rows = p2c163_summary["rows"]
    min_radius_row = min(rows, key=lambda row: row["second_radius_A"])
    max_radius_row = max(rows, key=lambda row: row["second_radius_A"])

    repair_rows = [
        {
            "priority": 1,
            "candidate": "显式接口矩连续残差增强",
            "mechanism": "在扰动重求解残差中单独提高接口矩连续约束权重，优先压低接口 0 的矩跳变。",
            "expected_benefit": "直接对应 P2c163 主失败分量；不改变任务维度。",
            "risk": "可能牺牲末端位移预测精度或导致一阶最优性退化。",
            "next_test": "P2c165 torque-weight continuation；只跑一个工作点和一条两步路径。",
        },
        {
            "priority": 2,
            "candidate": "显式刚段力臂矩平衡审计",
            "mechanism": "将刚段外力产生的力臂矩与磁外矩分开记录，确认矩跳变是否来自力矩参考点约定。",
            "expected_benefit": "区分模型缺项与坐标/参考点约定问题。",
            "risk": "只读审计不能直接改善求解。",
            "next_test": "P2c165 或 P2c166 输出 tau_ext、r×F、post-rigid moment 三项。",
        },
        {
            "priority": 3,
            "candidate": "矩连续松弛变量正则化",
            "mechanism": "引入小维度矩松弛变量，但以强正则约束，避免自由变量替代真实物理链。",
            "expected_benefit": "若硬矩连续导致不可达，可识别所需最小松弛量。",
            "risk": "存在物理解释被弱化风险，不能直接发布灵敏度。",
            "next_test": "只作为诊断分支，若松弛量非零且稳定，再回到物理模型修复。",
        },
        {
            "priority": 4,
            "candidate": "继续扫半径或工作点",
            "mechanism": "扩大第二步半径/路径/工作点扫描。",
            "expected_benefit": "短期可能找到偶然通过样本。",
            "risk": "P2c163 已显示最小半径仍失败；继续扫参不能解释矩连续失配。",
            "next_test": "暂缓；除非模型修复后再做复核。",
        },
    ]

    failure_events = [
        {
            "failure_class": "接口矩连续残差主导",
            "evidence": "P2c163",
            "min_radius_A": min_radius_row["second_radius_A"],
            "min_radius_continuity_norm": min_radius_row["continuity_norm"],
            "min_radius_force_block_norm": min_radius_row["dominant_force_norm"],
            "min_radius_torque_block_norm": min_radius_row["dominant_torque_norm"],
            "interpretation": "第二步缩半径后仍失败，且主导分量为矩残差；应优先修复矩连续模型。",
        },
        {
            "failure_class": "扫参边际收益不足",
            "evidence": "P2c163",
            "max_radius_A": max_radius_row["second_radius_A"],
            "max_radius_continuity_norm": max_radius_row["continuity_norm"],
            "interpretation": "从 0.00025 A 到 0.001 A 均不通过，继续扩大路径扫描不符合最小归因闭环。",
        },
    ]

    summary = {
        "experiment_id": EXP_DIR.name,
        "status": "torque_continuity_repair_map_defined",
        "source_experiment": str((P2C163 / "summary.json").relative_to(ROOT)),
        "dominant_failure": "接口 0 的矩连续残差主导",
        "second_step_radius_pass_found": False,
        "recommended_next": "P2c165 torque-weight continuation on the same n2 workpoint and two-step path",
        "outer_ik_allowed": False,
        "full_3d_position_publishable": False,
        "multistep_publishable": False,
        "min_radius_A": min_radius_row["second_radius_A"],
        "min_radius_continuity_norm": min_radius_row["continuity_norm"],
        "min_radius_force_block_norm": min_radius_row["dominant_force_norm"],
        "min_radius_torque_block_norm": min_radius_row["dominant_torque_norm"],
        "repair_candidates": repair_rows,
    }
    write_json(EXP_DIR / "summary.json", summary)
    write_csv(EXP_DIR / "torque_repair_candidates.csv", repair_rows)
    write_jsonl(EXP_DIR / "failure_events.jsonl", failure_events)

    configure_chinese_font()
    labels = ["力分量", "矩分量"]
    values = [min_radius_row["dominant_force_norm"], min_radius_row["dominant_torque_norm"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(labels, values, color=["#4C78A8", "#E45756"])
    ax.set_yscale("log")
    ax.set_ylabel("主导接口分块残差")
    ax.set_title("P2c163 最小第二步半径下的接口残差分解")
    ax.text(0.5, -0.22, "矩分量主导，下一轮优先修复矩连续约束。", transform=ax.transAxes, ha="center")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "n2_torque_continuity_repair_map_cn.png", dpi=200)
    plt.close(fig)

    record = f"""# P2c164 两点磁控 n2 接口矩连续修复图谱

## 实验目的

P2c163 已证明，滚动重建后第二步即使缩小到 0.00025 A 仍不能通过接口连续门禁。本轮不再继续扫半径，而是基于 P2c163 的残差分块结果，明确下一轮最小模型修复方向。

## 证据基础

P2c163 中最小第二步半径为 {min_radius_row['second_radius_A']:.5f} A。该条件下接口连续残差为 {min_radius_row['continuity_norm']:.3e}，超过 1.0e-3 门限。主导接口为接口 {min_radius_row['dominant_interface_index']}，力分量为 {min_radius_row['dominant_force_norm']:.3e}，矩分量为 {min_radius_row['dominant_torque_norm']:.3e}。

## 物理解释

当前接口残差比较刚段推进后的内力矩与下一柔性段近端内力矩。若力连续基本稳定而矩连续超限，说明主要问题不在总外力传递，而在刚段外矩、力臂矩、姿态参考点或矩连续权重的耦合。继续缩小电流步长不能消除此误差，说明该误差含有基线模型误差成分。

## 修复候选

| 优先级 | 候选 | 机制 | 风险 |
|---:|---|---|---|
| 1 | 显式接口矩连续残差增强 | 单独提高接口 0 矩连续约束权重 | 可能牺牲末端位移或一阶最优性 |
| 2 | 显式刚段力臂矩平衡审计 | 分离外磁矩与力臂矩 | 只读审计不能直接改善求解 |
| 3 | 矩连续松弛变量正则化 | 识别最小必要松弛量 | 物理解释会变弱 |
| 4 | 继续扫半径或工作点 | 扩大经验搜索 | 不能解释主失败机制，暂缓 |

## 工程结论

n2 多步灵敏度目前不能发布，也不能接入外层 IK。下一轮应做 P2c165：在同一工作点和同一路径上进行矩连续权重延续实验，判断增强矩连续约束是否能让第二步通过，同时检查末端位移、线性化误差和一阶最优性是否退化。

## 下一步计划

P2c165 只跑一个工作点、一条两步路径、少量矩连续权重。若权重增强后第二步通过，再扩展路径；若仍失败，转入刚段力臂矩与磁外矩的参考点审计。
"""
    (EXP_DIR / "record.md").write_text(record, encoding="utf-8")

    commands = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/20260630190000_p2c164_n2_torque_continuity_repair_map/run_p2c164_n2_torque_continuity_repair_map.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")


if __name__ == "__main__":
    main()
