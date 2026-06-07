"""Build a decision audit for single- and double-magnet interface models."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"


def configure_chinese_font() -> None:
    """Configure matplotlib to render Chinese labels."""
    if Path(FONT_PATH).exists():
        fm.fontManager.addfont(FONT_PATH)
        plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
    plt.rcParams["axes.unicode_minus"] = False


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a list of dictionaries to CSV."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: object) -> None:
    """Write JSON with stable formatting."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_rows() -> list[dict[str, object]]:
    """Return evidence rows for candidate interface models."""
    return [
        {
            "model_id": "current_segment_wrench",
            "中文名称": "当前分段内力内矩模型",
            "证据来源": "20260606173500_p2c28, 20260607142339_p2c39",
            "适用对象": "n1/n2 诊断基线",
            "一阶最优性证据": "n1=2.47e-5, n2=1.10e0",
            "载荷传播证据": "n1=2.73, n2=2.94",
            "主要优点": "实现成熟，能复现实验基线",
            "主要风险": "独立分段自由度允许非物理载荷补偿",
            "推荐等级": 2,
            "决策": "保留为对照，不作为下一步修复主线",
        },
        {
            "model_id": "hard_continuity_kkt",
            "中文名称": "硬连续约束模型",
            "证据来源": "20260605080801_p2c10r, 20260604132850_p2c9d",
            "适用对象": "连续性极限诊断",
            "一阶最优性证据": "约束方程组条件数约 1e16-1e17",
            "载荷传播证据": "约束与终端任务难以同时满足",
            "主要优点": "物理约束形式清晰",
            "主要风险": "数值病态严重，代表工况多初值失败",
            "推荐等级": 1,
            "决策": "暂不作为实现主线，只保留为反证",
        },
        {
            "model_id": "centered_magnetic_wrench",
            "中文名称": "刚段中心载荷修正模型",
            "证据来源": "20260604232300_p2c10d",
            "适用对象": "磁载荷作用点诊断",
            "一阶最优性证据": "alpha=0.25 由 20.59 恶化到 27.44",
            "载荷传播证据": "单纯力臂修正未降低主失配",
            "主要优点": "物理解释直接，改动局部",
            "主要风险": "只改载荷作用点，未修复刚柔接口传递链",
            "推荐等级": 2,
            "决策": "可借鉴载荷中心定义，但不单独推进",
        },
        {
            "model_id": "base_slack",
            "中文名称": "连续基座内力内矩加接口松弛模型",
            "证据来源": "20260604130324_p2c9c",
            "适用对象": "低维物理闭合候选",
            "一阶最优性证据": "部分样本可达 1e-8 量级",
            "载荷传播证据": "松弛量可能隐藏真实接口失配",
            "主要优点": "避免每段完全独立，数值上比硬约束温和",
            "主要风险": "松弛变量若无限制，会污染灵敏度",
            "推荐等级": 4,
            "决策": "作为下一步主原型基础，但必须加入物理尺度化和门禁",
        },
        {
            "model_id": "scaled_interface_slack",
            "中文名称": "物理尺度化接口松弛模型",
            "证据来源": "由 P2c28/P2c39 失败机制推导",
            "适用对象": "n1/n2 物理闭合修复",
            "一阶最优性证据": "待实现验证",
            "载荷传播证据": "目标是将载荷传播比压到 <=1.5",
            "主要优点": "保留可解性，同时限制人工载荷补偿",
            "主要风险": "权重和尺度若设错，可能转化为新的软惩罚调参",
            "推荐等级": 5,
            "决策": "推荐为 P2c41 主线",
        },
    ]


def build_failure_events() -> list[dict[str, object]]:
    """Return failure-mode records for traceability."""
    return [
        {
            "case": "n1_tip_z_rand15",
            "source": "20260606173500_p2c28",
            "primary_failure": "刚性磁体段载荷传播失配",
            "evidence": "线性化误差约 1.14e-6，但载荷传播比约 2.73；末端线性化好不能证明物理灵敏度可信。",
            "next_action": "暂停纯扫参，进入接口载荷闭合修复。",
        },
        {
            "case": "n2_z_pair_rand12",
            "source": "20260606173500_p2c28",
            "primary_failure": "一阶最优性退化与刚性磁体段载荷传播失配并存",
            "evidence": "基线一阶最优性约 1.10，扰动后约 0.585，载荷传播比约 2.94。",
            "next_action": "先修复接口模型，再做三维末端位置灵敏度。",
        },
        {
            "case": "hard_continuity_probe",
            "source": "20260605080801_p2c10r",
            "primary_failure": "硬连续约束导致数值病态",
            "evidence": "约束方程组条件数约 1e16-1e17，代表工况多初值失败。",
            "next_action": "不直接使用硬连续 KKT；改用受控松弛或重参数化。",
        },
        {
            "case": "centered_wrench_probe",
            "source": "20260604232300_p2c10d",
            "primary_failure": "单纯载荷作用点修正不足",
            "evidence": "alpha=0.25 一阶最优性从约 20.59 恶化至约 27.44。",
            "next_action": "将刚段中心载荷定义作为组成部分，而非单独方案。",
        },
    ]


def plot_decision(rows: list[dict[str, object]]) -> None:
    """Plot the recommendation score for each model."""
    configure_chinese_font()
    names = [str(row["中文名称"]) for row in rows]
    scores = [float(row["推荐等级"]) for row in rows]
    colors = ["#9ca3af", "#ef4444", "#f97316", "#3b82f6", "#16a34a"]
    notes = {
        "current_segment_wrench": "对照",
        "hard_continuity_kkt": "反证",
        "centered_magnetic_wrench": "局部借鉴",
        "base_slack": "原型基础",
        "scaled_interface_slack": "下一步主线",
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(names, scores, color=colors)
    ax.set_xlim(0, 6.2)
    ax.set_xlabel("推荐等级：1 表示仅作反证，5 表示下一步主线")
    ax.set_title("单点与两点磁控 IK 接口模型决策")
    ax.grid(axis="x", alpha=0.25)
    for bar, row in zip(bars, rows):
        ax.text(
            bar.get_width() + 0.08,
            bar.get_y() + bar.get_height() / 2,
            notes[str(row["model_id"])],
            va="center",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(ROOT / "interface_model_decision_cn.png", dpi=180)
    plt.close(fig)


def build_record(rows: list[dict[str, object]], failures: list[dict[str, object]]) -> str:
    """Build the markdown experiment record."""
    table_lines = [
        "| 模型 | 证据来源 | 主要证据 | 决策 |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        table_lines.append(
            f"| {row['中文名称']} | {row['证据来源']} | "
            f"{row['一阶最优性证据']}；{row['载荷传播证据']} | {row['决策']} |"
        )

    failure_lines = [
        "| 工况 | 主失败类型 | 证据 | 下一步 |",
        "| --- | --- | --- | --- |",
    ]
    for event in failures:
        failure_lines.append(
            f"| {event['case']} | {event['primary_failure']} | "
            f"{event['evidence']} | {event['next_action']} |"
        )

    return "\n".join(
        [
            "# P2c40 单点与两点磁控接口模型决策记录",
            "",
            "## 实验目的",
            "",
            "本轮不新增大规模数值求解，而是对已有单点、两点磁控打靶 IK 证据进行接口模型决策审计。目标是避免继续在不可信物理链路上扫权重或扫初值，并为下一轮物理闭合修复选定最小实现路径。",
            "",
            "## 当前判断",
            "",
            "单点 `n1_tip_z_rand15` 与两点 `n2_z_pair_rand12` 的主要问题不是末端位置线性化完全失败，而是刚性磁体段载荷没有以受控方式传递到相邻柔性段。当前分段内力内矩模型允许界面松弛变量承担过多非物理补偿，因此即使末端扰动响应看似平滑，也不能发布为物理可信灵敏度。",
            "",
            "## 接口模型对比",
            "",
            *table_lines,
            "",
            "## 失败事件归因",
            "",
            *failure_lines,
            "",
            "## 决策",
            "",
            "下一步主线不再继续纯权重扫描，也不直接采用硬连续 KKT。推荐实现 `物理尺度化接口松弛模型`：以连续基座内力内矩为主变量，保留低维接口松弛用于数值可解性，但必须按刚段外载尺度归一化并加入载荷传播门禁。其目标不是让松弛变量拟合形态，而是显式检验并抑制非物理载荷补偿。",
            "",
            "## 下一步计划",
            "",
            "1. 在实验级 runner 中实现物理尺度化接口松弛残差。",
            "2. 只测试 `n1_tip_z_rand15` 和 `n2_z_pair_rand12` 的 position-only 任务。",
            "3. 先做基线门禁，不通过时不做扰动灵敏度扫描。",
            "4. 若载荷传播比仍大于 2，判定当前参数化不能修复该工况，转入更强的刚柔段载荷耦合模型。",
            "5. 若载荷传播比降至 1.5 以下且一阶最优性通过，再做 eps=0.005 和 0.010 的扰动重求解。",
            "",
            "## 结论",
            "",
            "P2c40 形成的是结构性决策，不是可发布灵敏度结果。当前最合理的推进方向是受控接口松弛与载荷传播门禁联合修复；硬连续约束与单纯载荷中心修正均不适合作为下一步主线。",
            "",
        ]
    )


def main() -> None:
    """Generate all P2c40 artifacts."""
    rows = build_rows()
    failures = build_failure_events()
    write_csv(ROOT / "interface_model_decision.csv", rows)
    write_json(ROOT / "failure_events.jsonl", failures)
    (ROOT / "failure_events.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in failures) + "\n",
        encoding="utf-8",
    )
    summary = {
        "experiment_id": ROOT.name,
        "type": "evidence_synthesis",
        "scope": ["single_magnet_n1", "double_magnet_n2"],
        "publishable_sensitivity_claim": False,
        "recommended_next_model": "scaled_interface_slack",
        "reason": "n1/n2 evidence points to rigid magnet load propagation mismatch rather than pure terminal linearization failure.",
        "hard_continuity_rejected": True,
        "pure_weight_scan_rejected": True,
        "next_gate": {
            "stationarity_norm": "<=1e-5",
            "load_propagation_ratio": "<=1.5 target, >2 reject",
            "branch_rms_m": "<=1e-3",
            "linearization_error": "<0.05",
        },
    }
    write_json(ROOT / "summary.json", summary)
    plot_decision(rows)
    (ROOT / "record.md").write_text(build_record(rows, failures), encoding="utf-8")
    (ROOT / "commands.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points\n"
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "conda run -n evolution_of_mcrs python "
        "experiment_records_ik_optimize/20260607143544_p2c40_single_double_interface_model_decision/"
        "run_p2c40_interface_model_decision.py\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
