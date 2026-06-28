"""Self-test the guarded n2 sensitivity interface candidate."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "experiment_records_ik_optimize"
for path in (ROOT, EXPERIMENT_ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt

import plotting_fonts
from n2_guarded_sensitivity_interface import N2GuardedSensitivityInterface


OUT_DIR = Path(__file__).resolve().parent


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    """Return JSON-safe values."""
    if hasattr(value, "__dict__"):
        return _jsonable(value.__dict__)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _validation_cases(interface: N2GuardedSensitivityInterface) -> list[dict[str, Any]]:
    """Validate accepted and rejected interface requests."""
    required = interface.as_dict()["required_request"]
    cases = {
        "exact_position_request": required,
        "full_pose_request": {**required, "terminal_quantity": "full_pose"},
        "wrong_task_dimension": {**required, "task_dimension": 6},
        "wrong_branch": {**required, "branch_label": "other_branch"},
        "wrong_eps": {**required, "eps": 0.01},
        "wrong_task_name": {**required, "task_name": "n2_full_pose"},
    }
    rows: list[dict[str, Any]] = []
    for name, request in cases.items():
        decision = interface.validate_request(request)
        rows.append(
            {
                "case": name,
                "accepted": decision.accepted,
                "reasons": ";".join(decision.reasons) or "none",
            }
        )
    return rows


def _plot(evidence: dict[str, Any], validation_rows: list[dict[str, Any]]) -> None:
    """Plot evidence and request validation results."""
    plotting_fonts.configure_matplotlib_cjk()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    gate_labels = ["一阶最优性", "载荷传播", "条件数/1e7", "线性化误差"]
    gate_values = [
        evidence["stationarity_norm"] / 1.0e-5,
        evidence["force_ratio"] / 1.5,
        evidence["task_condition"] / 1.0e7,
        evidence["guarded_repeat_max_rel_err"] / 0.05,
    ]
    axes[0].bar(gate_labels, gate_values, color="#2a9d8f")
    axes[0].axhline(1.0, color="#e76f51", linestyle="--", label="门禁")
    axes[0].set_ylabel("归一化门禁比例")
    axes[0].set_title("发布门禁余量")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    labels = [row["case"] for row in validation_rows]
    values = [1 if row["accepted"] else 0 for row in validation_rows]
    colors = ["#457b9d" if value else "#e76f51" for value in values]
    axes[1].bar(labels, values, color=colors)
    axes[1].set_ylim(-0.05, 1.15)
    axes[1].set_ylabel("是否允许")
    axes[1].set_title("接口请求边界")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("P2c107 两点磁控灵敏度接口候选")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_guarded_interface_candidate_cn.png", dpi=180)
    plt.close(fig)


def _record(summary: dict[str, Any]) -> str:
    """Render the Chinese experiment report."""
    evidence = summary["snapshot"]["evidence"]
    accepted = [row for row in summary["validation_rows"] if row["accepted"]]
    rejected = [row for row in summary["validation_rows"] if not row["accepted"]]
    return "\n".join(
        [
            "# P2c107 n2 契约保护灵敏度接口候选",
            "",
            "## 实验目的",
            "",
            "P2c104 到 P2c106 已证明两点磁控三维末端位置灵敏度在受信任分支条件下可快速复核。本轮将证据链整理为只读接口候选，明确工程可用范围和禁止范围。",
            "",
            "## 方法",
            "",
            "- 读取 P2c105 缓存契约和 P2c106 重复复核结果。",
            "- 构造接口请求校验函数，要求任务口径、扰动幅值、分支标签、任务维度和终端量完全一致。",
            "- 用正例和负例自检接口边界。",
            "",
            "## 证据摘要",
            "",
            f"- 状态：`{evidence['status']}`",
            f"- 一阶最优性：`{evidence['stationarity_norm']:.3e}`",
            f"- 载荷传播比例：`{evidence['force_ratio']:.3e}`",
            f"- 条件数：`{evidence['task_condition']:.3e}`",
            f"- 随机方向复核：`{evidence['random_direction_passed']}/{evidence['random_direction_total']}`",
            f"- 契约保护重复复核：`{evidence['guarded_repeat_passed']}/{evidence['guarded_repeat_total']}`",
            f"- 最大线性化相对误差：`{evidence['guarded_repeat_max_rel_err']:.3e}`",
            "",
            "## 接口边界",
            "",
            f"- 允许范围：{summary['snapshot']['allowed_scope']}",
            f"- 禁止范围：{summary['snapshot']['forbidden_scope']}",
            f"- 默认外层 IK 是否允许：`{summary['snapshot']['outer_ik_default_allowed']}`",
            f"- 请求正例通过数：`{len(accepted)}`",
            f"- 请求负例拒绝数：`{len(rejected)}`",
            "",
            "## 结论",
            "",
            "P2c107 形成了两点磁控三维末端位置局部灵敏度的契约保护只读接口候选。该接口只表达同一物理解分支附近的局部导数，不表达全局灵敏度，也不表达完整末端位姿灵敏度。",
            "",
            "## 下一步计划",
            "",
            "1. 后续若接入外层优化，只能以显式实验开关调用该接口，禁止默认启用。",
            "2. 继续评估首次分支恢复成本，优先考虑隐式灵敏度或更稳定的分支恢复方法。",
            "3. 单点磁控主线仍需回到刚性磁体段载荷传播失配问题，不能用该 n2 接口替代。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    interface = N2GuardedSensitivityInterface(ROOT)
    snapshot = interface.as_dict()
    validation_rows = _validation_cases(interface)
    failures = [
        {"failure": "unexpected_request_decision", "evidence": row}
        for row in validation_rows
        if row["accepted"] != (row["case"] == "exact_position_request")
    ]
    summary = {
        "experiment": OUT_DIR.name,
        "snapshot": snapshot,
        "validation_rows": validation_rows,
        "wall_s": float(time.perf_counter() - t0),
    }
    _write_csv(OUT_DIR / "validation_rows.csv", validation_rows)
    _write_csv(OUT_DIR / "summary.csv", validation_rows)
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(_jsonable(item), ensure_ascii=False) + "\n" for item in failures),
        encoding="utf-8",
    )
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(snapshot["evidence"], validation_rows)
    (OUT_DIR / "record.md").write_text(_record(summary), encoding="utf-8")
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
