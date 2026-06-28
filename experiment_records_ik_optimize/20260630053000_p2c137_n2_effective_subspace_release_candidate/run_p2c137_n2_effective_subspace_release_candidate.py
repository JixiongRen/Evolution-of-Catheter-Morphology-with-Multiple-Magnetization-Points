"""Build the n2 effective-subspace release-candidate package."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
P2C104_DIR = ROOT / "experiment_records_ik_optimize" / "20260629000000_p2c104_n2_cached_branch_audit"
P2C134_DIR = ROOT / "experiment_records_ik_optimize" / "20260630023000_p2c134_n2_trusted_branch_basin_radius"
P2C135_DIR = ROOT / "experiment_records_ik_optimize" / "20260630033000_p2c135_n2_basin_breakpoint_bracket"
P2C136_DIR = ROOT / "experiment_records_ik_optimize" / "20260630043000_p2c136_n2_trust_region_current_reopt"
for path in (ROOT, P2C104_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt
import numpy as np

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c104_n2_cached_branch_audit as p2c104


configure_matplotlib_cjk()
OUT_DIR = Path(__file__).resolve().parent


def jsonable(value: Any) -> Any:
    """Convert arrays and scalar objects to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def direction_rows(summary136: dict[str, Any], S: np.ndarray) -> list[dict[str, Any]]:
    """Summarize direction-level release status."""
    rows = summary136["rows"]
    by_direction = sorted({row["direction"] for row in rows})
    U, singular_values, _Vt = np.linalg.svd(S, full_matrices=False)
    principal_map = {
        "主灵敏方向1": U[:, 0],
        "主灵敏方向2": U[:, 1],
        "主灵敏方向3": U[:, 2],
        "x方向": np.array([1.0, 0.0, 0.0]),
        "y方向": np.array([0.0, 1.0, 0.0]),
        "z方向": np.array([0.0, 0.0, 1.0]),
    }
    rows_out: list[dict[str, Any]] = []
    for name in by_direction:
        group = [row for row in rows if row["direction"] == name]
        passes = [row for row in group if row["status"] == "pass"]
        max_pass_radius = max((row["radius_A"] for row in passes), default=None)
        conservative_pass = [row for row in passes if row["radius_A"] <= 1.0e-4]
        release_class = "不可发布"
        reason = "未通过可信门禁"
        if conservative_pass and name == "主灵敏方向1":
            release_class = "优势方向可发布"
            reason = "保守电流半径内通过局部线性化、状态信赖域、中心线漂移和载荷传播门禁"
        elif passes:
            release_class = "条件诊断"
            reason = "存在通过样本，但不满足保守全局方向发布条件"
        if all(row["failure_class"] == "末端位移低于可靠判读阈值" for row in group if row["status"] != "pass"):
            reason = "末端位移幅值过低，无法稳定判读局部线性化"
        vector = principal_map[name]
        singular_index = None
        singular_value = None
        if name.startswith("主灵敏方向"):
            singular_index = int(name[-1])
            singular_value = float(singular_values[singular_index - 1])
        rows_out.append(
            {
                "direction": name,
                "release_class": release_class,
                "reason": reason,
                "max_pass_radius_A": max_pass_radius,
                "conservative_pass_count": len(conservative_pass),
                "tested_count": len(group),
                "pass_count": len(passes),
                "min_linear_rel_err": min(row["linear_rel_err"] for row in group),
                "max_state_radius": max(row["state_radius"] for row in group),
                "max_force_ratio": max(row["max_interface_force_jump_over_rigid_force"] for row in group),
                "direction_vector_x": float(vector[0]),
                "direction_vector_y": float(vector[1]),
                "direction_vector_z": float(vector[2]),
                "singular_index": singular_index,
                "singular_value_m_per_A": singular_value,
            }
        )
    return rows_out


def plot_release(rows: list[dict[str, Any]], singular_values: np.ndarray) -> None:
    """Plot release classes and singular spectrum."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    color_map = {"优势方向可发布": "#2a9d8f", "条件诊断": "#e9c46a", "不可发布": "#b24b38"}
    labels = [row["direction"] for row in rows]
    axes[0].bar(labels, [row["pass_count"] for row in rows], color=[color_map[row["release_class"]] for row in rows])
    axes[0].set_title("方向级门禁通过数量")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar([f"s{i+1}" for i in range(len(singular_values))], singular_values, color="#457b9d")
    axes[1].set_yscale("log")
    axes[1].set_title("三维位置灵敏度奇异值")
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(OUT_DIR / "n2_effective_subspace_release_candidate_cn.png", dpi=220)
    plt.close(fig)


def write_record(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write the release-candidate report."""
    table = "\n".join(
        f"| {row['direction']} | {row['release_class']} | {row['max_pass_radius_A']} | {row['reason']} |"
        for row in rows
    )
    record = f"""# P2c137 两点磁控有效子空间发布候选包

## 实验目的

本轮不再进行新的数值求解，而是综合 P2c104、P2c134、P2c135 和 P2c136 的证据，形成两点磁控 n2 的局部任务灵敏度发布边界。重点区分三类结论：优势方向可发布、条件诊断、不可发布。

## 证据来源

- P2c104：建立同一可信缓存分支，并给出基准任务灵敏度矩阵。
- P2c134：有效子空间初值扰动半径 `1e-4` 内三方向稳定回到同一分支。
- P2c135：`2e-4` 起出现方向相关分支失败，`1e-3` 三方向全部失败，严格稳定半径位于 `1e-4` 至 `2e-4`。
- P2c136：在电流扰动真实重求解中，只有第一主灵敏方向在保守半径内满足局部线性化和物理门禁；弱方向主要受位移幅值过低和方向增益不足限制。

## 数学表述

设基准电流为 \(I_0\)，末端位置为 \(p(I)\)，在可信分支 \(\mathcal{{B}}\) 上定义局部线性映射：

\\[
\\Delta p \\approx S\\Delta I, \\quad S = \\left.\\frac{{\\partial p}}{{\\partial I}}\\right|_{{I_0,\\mathcal{{B}}}} .
\\]

该映射只在同时满足物理解分支、状态信赖域、载荷传播和局部线性化门禁时有效。若扰动离开分支 \(\mathcal{{B}}\)，所得差分不再代表同一物理解的局部导数。

## 方向级结论

| 方向 | 发布类别 | 最大通过电流半径/A | 依据 |
| --- | --- | ---: | --- |
{table}

## 发布边界

- 可发布内容：分支条件下第一主灵敏方向的局部末端位置导数。
- 不可发布内容：完整三维末端位置灵敏度、完整末端位姿灵敏度、无分支条件的全局灵敏度。
- 工程信赖域：有效子空间状态半径不超过 `1e-4`；若使用电流扰动，应额外检查重求解状态半径、中心线漂移、载荷传播比例和局部线性化误差。
- 失败主因：有效驱动方向高度各向异性，弱方向末端位移低于可靠判读阈值；不是当前证据中的载荷传播失配。

## 工程建议

1. 当前 n2 工况可作为优势方向局部调节或诊断接口，不应接入三维外层逆运动学闭环。
2. 若任务要求三维末端位置控制，应更换基准电流、目标工作点或磁化配置，使三维奇异值更均衡。
3. 外层控制若必须使用该工况，应投影到第一主灵敏方向，并设置状态信赖域检查；越界后必须重新恢复可信分支。
4. 弱方向只能报告为不可可靠辨识，不能用相对误差失败直接解释为模型物理错误。

## 结论

P2c137 给出 n2 的阶段性可发布候选：仅限“分支条件下的优势方向局部导数”。完整三维末端位置灵敏度尚不满足发布要求。

## 下一步计划

下一步应回到 n1 单点磁控主线，生成同样的有效子空间发布或不可发布边界；同时可在 n2 上尝试更换工作点，以寻找三维奇异值更均衡的候选。
"""
    (OUT_DIR / "record.md").write_text(record, encoding="utf-8")


def main() -> None:
    """Build the release-candidate artifacts."""
    params, _I0 = p2c104.p2c79.p2c57.p2c24._params_from_saved_case(
        p2c104.p2c79.p2c57.CASE_NAME,
        p2c104.p2c79.p2c57.ALPHA,
    )
    branch = p2c104._load_cached_branch(params)
    S = np.asarray(branch["S"], dtype=float)
    singular_values = np.linalg.svd(S, compute_uv=False)
    summary134 = load_json(P2C134_DIR / "summary.json")
    summary135 = load_json(P2C135_DIR / "summary.json")
    summary136 = load_json(P2C136_DIR / "summary.json")
    rows = direction_rows(summary136, S)
    release_rows = [row for row in rows if row["release_class"] == "优势方向可发布"]
    payload = {
        "experiment_id": OUT_DIR.name,
        "case": p2c104.p2c79.p2c57.CASE_NAME,
        "alpha": p2c104.p2c79.p2c57.ALPHA,
        "release_status": "effective_subspace_candidate" if release_rows else "no_publishable_direction",
        "full_3d_position_publishable": False,
        "full_pose_publishable": False,
        "outer_ik_allowed": False,
        "release_direction_count": len(release_rows),
        "release_directions": release_rows,
        "direction_rows": rows,
        "singular_values_m_per_A": singular_values.tolist(),
        "condition": float(singular_values[0] / max(singular_values[-1], np.finfo(float).tiny)),
        "trusted_state_radius_gate": 1.0e-4,
        "all_direction_break_interval": summary135.get("all_direction_break_interval"),
        "evidence": {
            "p2c134_summary": str(P2C134_DIR / "summary.json"),
            "p2c135_summary": str(P2C135_DIR / "summary.json"),
            "p2c136_summary": str(P2C136_DIR / "summary.json"),
            "p2c134_largest_all_direction_pass_radius": summary134.get("largest_all_direction_pass_radius"),
            "p2c136_pass_count": summary136.get("pass_count"),
            "p2c136_low_signal_failure_count": summary136.get("low_signal_failure_count"),
        },
    }
    write_csv(OUT_DIR / "direction_release_rows.csv", rows)
    write_csv(
        OUT_DIR / "summary.csv",
        [
            {
                "release_status": payload["release_status"],
                "release_direction_count": payload["release_direction_count"],
                "full_3d_position_publishable": payload["full_3d_position_publishable"],
                "outer_ik_allowed": payload["outer_ik_allowed"],
                "condition": payload["condition"],
            }
        ],
    )
    (OUT_DIR / "summary.json").write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "failure_events.jsonl").write_text(
        json.dumps(
            {
                "failure_class": "完整三维位置灵敏度不可发布",
                "reason": "仅第一主灵敏方向满足保守发布条件，弱方向位移幅值过低且不可可靠辨识。",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_release(rows, singular_values)
    write_record(payload, rows)
    print(json.dumps(jsonable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
