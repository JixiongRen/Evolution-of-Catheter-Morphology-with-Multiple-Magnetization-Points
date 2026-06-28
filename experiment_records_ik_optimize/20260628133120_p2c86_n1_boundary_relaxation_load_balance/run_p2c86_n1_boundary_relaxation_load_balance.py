"""Run the P2c86 n1 boundary-relaxation and load-balance diagnostic."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
P2C84_DIR = ROOT / "experiment_records_ik_optimize" / "20260628121600_p2c84_block_stationarity_load_closure_audit"


def _set_chinese_font() -> None:
    """Use a local Chinese font when available."""
    for font_file in [
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    ]:
        if font_file.exists():
            fm.fontManager.addfont(str(font_file))
            prop = fm.FontProperties(fname=str(font_file))
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [prop.get_name()]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _read_csv(path: Path) -> list[dict[str, Any]]:
    """Read CSV rows and convert numeric fields."""
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _n1_data() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load n1 near-gate data from P2c84."""
    gate_rows = _read_csv(P2C84_DIR / "gate_rows.csv")
    block_rows = _read_csv(P2C84_DIR / "block_gradient_rows.csv")
    cosine_rows = _read_csv(P2C84_DIR / "block_gradient_cosines.csv")
    label = "n1_soft_near_gate"
    return (
        next(row for row in gate_rows if row["label"] == label),
        [row for row in block_rows if row["label"] == label],
        [row for row in cosine_rows if row["label"] == label],
    )


def _gram_matrix(blocks: list[dict[str, Any]], cosines: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    """Reconstruct the block-gradient Gram matrix from norms and pairwise cosines."""
    names = [row["block"] for row in blocks]
    norms = np.array([row["gradient_norm"] for row in blocks], dtype=float)
    gram = np.diag(norms**2)
    index = {name: i for i, name in enumerate(names)}
    for row in cosines:
        i = index[row["block_a"]]
        j = index[row["block_b"]]
        gram[i, j] = row["gradient_cosine"] * norms[i] * norms[j]
        gram[j, i] = gram[i, j]
    return names, gram


def _candidate_metrics(
    gate: dict[str, Any],
    names: list[str],
    gram: np.ndarray,
    candidate: str,
    scales: dict[str, float],
    load_scale: float,
    original_physics: bool,
) -> dict[str, Any]:
    """Evaluate a first-order block-scaling counterfactual."""
    scale_vec = np.array([scales.get(name, 1.0) for name in names], dtype=float)
    scaled_gram = gram * np.outer(scale_vec, scale_vec)
    block_norms = np.sqrt(np.maximum(np.diag(scaled_gram), 0.0))
    sum_norm = float(np.sqrt(max(np.sum(scaled_gram), 0.0)))
    abs_sum = float(np.sum(block_norms))
    cancellation_ratio = sum_norm / abs_sum if abs_sum else 0.0
    negative_coupling = 0.0
    total_pair = 0.0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            value = scaled_gram[i, j]
            total_pair += abs(value)
            if value < 0.0:
                negative_coupling += abs(value)
    conflict_fraction = negative_coupling / total_pair if total_pair else 0.0
    apparent_load_ratio = gate["max_interface_force_jump_over_rigid_force"] * load_scale
    original_load_ratio = gate["max_interface_force_jump_over_rigid_force"]
    return {
        "candidate": candidate,
        "terminal_boundary_scale": scales.get("末端边界", 1.0),
        "interface_continuity_scale": scales.get("接口连续", 1.0),
        "load_scale": load_scale,
        "proxy_gradient_sum_norm": sum_norm,
        "cancellation_ratio": cancellation_ratio,
        "negative_coupling_fraction": conflict_fraction,
        "apparent_load_ratio": apparent_load_ratio,
        "original_load_ratio": original_load_ratio,
        "stationarity_norm": gate["stationarity_norm"],
        "jacobian_condition": gate["jacobian_condition"],
        "continuity_norm": gate["continuity_norm"],
        "position_error_m": gate["position_error_m"],
        "original_physics_evaluation": original_physics,
        "publishable": False,
        "interpretation": _interpret(candidate, conflict_fraction, apparent_load_ratio, original_physics),
    }


def _interpret(candidate: str, conflict_fraction: float, apparent_load_ratio: float, original_physics: bool) -> str:
    """Write a compact Chinese interpretation."""
    if original_physics:
        return "原始物理模型评价；当前仍因载荷传播比例超过门限而不可发布。"
    if apparent_load_ratio <= 1.5 and conflict_fraction < 0.75:
        return f"{candidate} 可降低表观冲突或载荷比例，但属于反事实诊断，不构成物理通过。"
    return f"{candidate} 未同时压低冲突和载荷传播风险，不能作为修复方向。"


def _candidate_rows(gate: dict[str, Any], blocks: list[dict[str, Any]], cosines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build all diagnostic candidates."""
    names, gram = _gram_matrix(blocks, cosines)
    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    whitening_scales = {name: float(1.0 / max(norm, 1.0e-12)) for name, norm in zip(names, norms)}
    return [
        _candidate_metrics(gate, names, gram, "原始物理模型", {}, 1.0, True),
        _candidate_metrics(gate, names, gram, "末端边界弱化", {"末端边界": 0.25}, 1.0, False),
        _candidate_metrics(gate, names, gram, "接口载荷重标定", {}, 1.0 / gate["max_interface_force_jump_over_rigid_force"], False),
        _candidate_metrics(
            gate,
            names,
            gram,
            "边界-接口联合白化",
            {**whitening_scales, "正则项": 1.0},
            1.0 / gate["max_interface_force_jump_over_rigid_force"],
            False,
        ),
    ]


def _plot(rows: list[dict[str, Any]]) -> None:
    """Plot P2c86 diagnostic metrics."""
    _set_chinese_font()
    labels = [row["candidate"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    axes[0].bar(labels, [row["negative_coupling_fraction"] for row in rows], color="#457b9d")
    axes[0].axhline(0.75, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("负耦合占比")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, [row["apparent_load_ratio"] for row in rows], color="#e76f51")
    axes[1].axhline(1.5, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title("表观载荷传播比例")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    axes[2].bar(labels, [row["cancellation_ratio"] for row in rows], color="#2a9d8f")
    axes[2].set_title("一阶抵消比例")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("P2c86 单点边界冲突反事实诊断")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n1_boundary_relaxation_load_balance_cn.png", dpi=180)
    plt.close(fig)


def _record(gate: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    """Render the Chinese experiment record."""
    table = [
        "| 候选处理 | 负耦合占比 | 表观载荷比例 | 一阶抵消比例 | 原始物理评价 | 可发布 | 解释 |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        table.append(
            f"| {row['candidate']} | {row['negative_coupling_fraction']:.3f} | "
            f"{row['apparent_load_ratio']:.3f} | {row['cancellation_ratio']:.3f} | "
            f"{'是' if row['original_physics_evaluation'] else '否'} | "
            f"{'是' if row['publishable'] else '否'} | {row['interpretation']} |"
        )
    return "\n".join(
        [
            "# P2c86 n1 末端边界弱化与载荷平衡反事实诊断",
            "",
            "## 实验目的",
            "",
            "本轮承接 P2c85。目标不是发布新的灵敏度，而是用 P2c84 的残差块一阶结构，判断单点 n1 的失败更接近边界权重问题、接口载荷尺度问题，还是二者耦合问题。",
            "",
            "## 方法说明",
            "",
            "本轮不重新求解打靶问题，而是做一阶反事实诊断。方法为：从 P2c84 的残差块梯度范数与块间夹角余弦重构块梯度内积矩阵；随后分别缩放末端边界块、接口载荷指标和联合白化块梯度，观察负耦合占比、表观载荷传播比例和一阶抵消比例变化。",
            "",
            "这些处理只用于定位模型闭合缺口。除原始物理模型外，任何反事实候选均不能直接视为物理门禁通过，也不能接入扰动灵敏度。",
            "",
            "## 原始门禁",
            "",
            f"- 一阶最优性：`{gate['stationarity_norm']:.3e}`",
            f"- 条件数：`{gate['jacobian_condition']:.3e}`",
            f"- 接口连续：`{gate['continuity_norm']:.3e}`",
            f"- 载荷传播比例：`{gate['max_interface_force_jump_over_rigid_force']:.3e}`",
            f"- 末端位置误差：`{gate['position_error_m']:.3e} m`",
            "",
            "## 诊断结果",
            "",
            *table,
            "",
            "## 结论",
            "",
            "单纯弱化末端边界不能消除载荷传播失败；单纯重标定接口载荷可以压低表观载荷比例，但并未改变原物理模型中的载荷链。联合白化能够作为数值诊断降低块尺度支配，但它改变的是残差度量，不是物理闭合条件。因此，n1 的下一步不应继续做权重扫描，而应进入物理闭合模型修复：显式描述刚性磁体段两端的力矩传递，或重写接口处的载荷连续性残差，使其与磁体中心受载假设一致。",
            "",
            "## 下一步计划",
            "",
            "1. 暂停 n1 末端位置灵敏度发布尝试，保留为不可发布候选。",
            "2. 开展 P2c87：构造刚性磁体段端点载荷传播的显式残差，与当前中心载荷近似作同工况对比。",
            "3. 若显式端点载荷模型仍不能使载荷传播比例降至 1.5 以下，则给出当前单点配置的工程不可发布边界。",
            "",
        ]
    )


def main() -> None:
    gate, blocks, cosines = _n1_data()
    rows = _candidate_rows(gate, blocks, cosines)
    _write_csv(OUT_DIR / "summary.csv", rows)
    summary = {
        "experiment": "P2c86_n1_boundary_relaxation_load_balance",
        "source": str(P2C84_DIR),
        "status": "diagnostic_only_not_publishable",
        "main_conclusion": "n1 失败不能由末端边界弱化或接口载荷重标定单独修复；需要物理闭合模型修复。",
        "rows": rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            if not row["publishable"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    _plot(rows)
    (OUT_DIR / "record.md").write_text(_record(gate, rows), encoding="utf-8")


if __name__ == "__main__":
    main()
