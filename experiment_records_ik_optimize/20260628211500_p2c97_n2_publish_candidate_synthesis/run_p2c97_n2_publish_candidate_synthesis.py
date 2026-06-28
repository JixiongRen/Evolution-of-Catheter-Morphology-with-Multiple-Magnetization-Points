"""Synthesize n2 branch-conditioned sensitivity evidence."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

OUT_DIR = Path(__file__).resolve().parent
configure_matplotlib_cjk()

SOURCES = {
    "P2c88": ROOT / "experiment_records_ik_optimize/20260628143800_p2c88_n2_branch_condition_synthesis/summary.json",
    "P2c92": ROOT / "experiment_records_ik_optimize/20260628165500_p2c92_n2_target_radius_100um/summary.json",
    "P2c93": ROOT / "experiment_records_ik_optimize/20260628175300_p2c93_n2_target_combo_100um/summary.json",
    "P2c94": ROOT / "experiment_records_ik_optimize/20260628183000_p2c94_n2_target_random_100um/summary.json",
    "P2c95": ROOT / "experiment_records_ik_optimize/20260628192500_p2c95_n2_eps010_trusted_sensitivity/summary.json",
    "P2c96": ROOT / "experiment_records_ik_optimize/20260628205000_p2c96_n2_eps0125_trusted_sensitivity/summary.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _target_row(name: str, data: dict[str, Any], label: str) -> dict[str, Any]:
    """Summarize a target-continuation experiment."""
    summaries = data["summaries"]
    return {
        "experiment": name,
        "category": "目标闭环",
        "label": label,
        "passed": sum(1 for row in summaries if row.get("return_ok")),
        "total": len(summaries),
        "max_primary_metric": max(row.get("return_state_drift", 0.0) for row in summaries),
        "primary_metric_name": "最大返回状态漂移",
        "max_stationarity": max(row.get("return_stationarity_norm", 0.0) for row in summaries),
        "max_force_ratio": max(row.get("return_force_ratio", 0.0) for row in summaries),
        "wall_s": data.get("wall_s"),
        "status": "通过" if all(row.get("return_ok") for row in summaries) else "未通过",
    }


def _sensitivity_row(name: str, data: dict[str, Any], label: str) -> dict[str, Any]:
    """Summarize a current-perturbation sensitivity experiment."""
    rows = data["audit_rows"]
    return {
        "experiment": name,
        "category": "扰动重求解",
        "label": label,
        "eps": data.get("eps"),
        "passed": sum(1 for row in rows if row.get("status") == "publishable"),
        "total": len(rows),
        "max_primary_metric": max(row.get("vector_rel_err", 0.0) for row in rows),
        "primary_metric_name": "最大线性化相对误差",
        "max_stationarity": data.get("branch_row", {}).get("stationarity_norm"),
        "max_force_ratio": data.get("branch_row", {}).get("force_ratio"),
        "task_condition": data.get("branch_row", {}).get("task_condition"),
        "wall_s": data.get("wall_s"),
        "status": "通过" if all(row.get("status") == "publishable" for row in rows) else "未通过",
    }


def _plot(rows: list[dict[str, Any]]) -> None:
    """Plot the updated n2 evidence chain."""
    labels = [row["experiment"] for row in rows]
    pass_ratio = [row["passed"] / row["total"] for row in rows]
    primary = [row["max_primary_metric"] for row in rows]
    wall = [row["wall_s"] for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].bar(labels, pass_ratio, color="#2a9d8f")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("通过比例")
    axes[1].bar(labels, primary, color="#457b9d")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("主指标最大值")
    axes[2].bar(labels, wall, color="#e76f51")
    axes[2].set_ylabel("耗时/s")
    axes[2].tick_params(axis="x", rotation=25)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("P2c97 n2 分支条件灵敏度证据链")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_publish_candidate_synthesis_cn.png", dpi=180)
    plt.close(fig)


def _record(rows: list[dict[str, Any]]) -> str:
    """Render the Chinese synthesis report."""
    table = [
        "| 实验 | 类型 | 结论 | 通过数 | 主指标 | 主指标最大值 | 一阶最优性 | 载荷比 | 耗时/s |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        table.append(
            f"| {row['experiment']} | {row['category']} | {row['status']} | {row['passed']}/{row['total']} | "
            f"{row['primary_metric_name']} | {row['max_primary_metric']:.3e} | "
            f"{row.get('max_stationarity', float('nan')):.3e} | {row.get('max_force_ratio', float('nan')):.3e} | "
            f"{row.get('wall_s', float('nan')):.2f} |"
        )
    total_wall = sum(float(row.get("wall_s") or 0.0) for row in rows)
    return "\n".join(
        [
            "# P2c97 n2 分支条件三维末端位置灵敏度阶段性综合",
            "",
            "## 实验目的",
            "",
            "本轮不重新求解导管平衡，而是综合 P2c88 至 P2c96 的现有证据，更新两点磁控 n2 工况的阶段性发布边界。重点区分三类结论：末端拟合成功、局部分支可跟踪、灵敏度可线性化。",
            "",
            "## 证据来源",
            "",
            "- P2c88：旧证据综合，指出 n2 只能作为受信任分支条件候选，不能初始化无关发布。",
            "- P2c92-P2c94：100 微米目标点闭环，覆盖坐标轴、组合方向和固定随机方向。",
            "- P2c95-P2c96：`eps=0.010` 与 `eps=0.0125` 电流扰动重求解一致性复核。",
            "",
            "## 汇总结果",
            "",
            *table,
            "",
            "## 阶段性判定",
            "",
            "n2 当前已经具备较强的“受信任分支条件下三维末端位置灵敏度”证据。该结论并不依赖单次末端拟合表象，而是同时得到两类独立支持：其一，目标点闭环显示该分支在 100 微米局部任务邻域内可连续跟踪；其二，电流扰动重求解显示局部导数在 `eps=0.005`、`0.010` 和 `0.0125` 尺度上均满足发布级线性化误差门禁。",
            "",
            "该结论尚不能升级为初始化无关的稳健灵敏度。P2c88 之前已经给出明确证据：零初值、半初值和普通电流延拓不能稳定回到同一物理解。因此，任何表述都必须加上“受信任分支条件”。",
            "",
            "## 可发布范围",
            "",
            "- 可发布候选：`n2_z_pair_rand12, alpha=0.10, force=4, reg=3e-4` 工况下，受信任分支条件的三维末端位置局部灵敏度。",
            "- 可用任务：position-only，即末端三维位置。完整末端姿态仍不可发布。",
            "- 可用尺度：当前证据覆盖 100 微米目标点邻域，以及至少 `eps=0.0125` 的电流扰动重求解尺度。",
            "- 必要限定：多平衡区域内的分支条件导数，不能解释为全局灵敏度。",
            "",
            "## 不可发布范围",
            "",
            "- 不能发布初始化无关结论。",
            "- 不能默认接入外层 IK 作为生产灵敏度。",
            "- 不能发布完整末端位姿灵敏度。",
            "- 不能把状态层面的病态完整灵敏度作为物理可解释量；当前可信对象是末端位置任务空间投影后的局部导数。",
            "",
            "## 失败机制与工程边界",
            "",
            "当前主要失败机制是初始化敏感和多平衡分支竞争，而不是末端位置任务维度本身不可线性化。任务空间条件数约为 `3.23e2`，说明在受信任分支内三维末端位置是可辨识的；但普通初值路径会落入不同物理解或载荷链异常解，导致不能初始化无关发布。",
            "",
            "工程上建议先将 n2 作为“分支条件局部导数诊断模块”使用。若要接入外层 IK，必须额外实现可靠分支选择机制，或者证明至少两条工程可复现路径能回到同一物理解。",
            "",
            "## 下一步计划",
            "",
            "1. 做随机电流方向批量复核，确认 `eps=0.0125` 结论不依赖 P2c79 固定方向集合。",
            "2. 同步维护 n1 不可发布机制归因，避免把 n2 结论外推到单点磁控。",
            "3. 在可信性不下降后，再做缓存分支和批量门禁等效率优化。",
            "",
            f"- 证据链总运行耗时：`{total_wall:.2f} s`",
            "",
        ]
    )


def main() -> None:
    data = {name: _read_json(path) for name, path in SOURCES.items()}
    rows = [
        _target_row("P2c92", data["P2c92"], "坐标轴 100 微米目标闭环"),
        _target_row("P2c93", data["P2c93"], "组合方向 100 微米目标闭环"),
        _target_row("P2c94", data["P2c94"], "随机方向 100 微米目标闭环"),
        _sensitivity_row("P2c95", data["P2c95"], "eps=0.010 电流扰动"),
        _sensitivity_row("P2c96", data["P2c96"], "eps=0.0125 电流扰动"),
    ]
    payload = {
        "experiment": OUT_DIR.name,
        "status": "branch_conditioned_publish_candidate",
        "sources": {name: str(path.relative_to(ROOT)) for name, path in SOURCES.items()},
        "rows": rows,
        "decision": {
            "n2_position_only_branch_conditioned": "publish_candidate",
            "n2_initialization_independent": "not_publishable",
            "n2_full_pose": "not_publishable",
            "recommended_next": "random_current_direction_batch_audit",
        },
    }
    _write_csv(OUT_DIR / "summary.csv", rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "failure_events.jsonl").write_text("", encoding="utf-8")
    _plot(rows)
    (OUT_DIR / "record.md").write_text(_record(rows), encoding="utf-8")
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
