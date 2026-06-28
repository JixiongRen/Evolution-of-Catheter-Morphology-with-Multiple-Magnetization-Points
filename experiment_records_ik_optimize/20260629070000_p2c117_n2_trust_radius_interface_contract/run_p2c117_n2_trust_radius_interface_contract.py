"""Validate the n2 trust-radius guarded interface contract."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk
from n2_trust_radius_sensitivity_interface import N2TrustRadiusSensitivityInterface


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write CSV rows."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_json(path: Path, data: dict) -> None:
    """Write stable JSON output."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_requests(interface: N2TrustRadiusSensitivityInterface) -> list[tuple[str, dict]]:
    """Build positive and negative contract requests."""
    base = interface.as_dict()["required_request"]
    radius = interface.max_linear_current_step_A()
    unit = np.zeros(8, dtype=float)
    unit[0] = 1.0
    requests = [
        ("valid_half_radius", {**base, "step_type": "current_increment", "dI_A": (0.5 * radius * unit).tolist()}),
        ("valid_boundary_radius", {**base, "step_type": "current_increment", "dI_A": (radius * unit).tolist()}),
        ("too_large_radius", {**base, "step_type": "current_increment", "dI_A": (1.01 * radius * unit).tolist()}),
        ("physical_budget_misuse", {**base, "step_type": "current_increment", "dI_A": (5.0 * unit).tolist(), "uses_physical_budget_as_linear_radius": True}),
        ("full_pose_request", {**base, "task_dimension": 6, "terminal_quantity": "full_pose", "step_type": "current_increment", "dI_A": (0.5 * radius * unit).tolist()}),
        ("wrong_branch", {**base, "branch_label": "untrusted", "step_type": "current_increment", "dI_A": (0.5 * radius * unit).tolist()}),
        ("target_displacement_request", {**base, "step_type": "task_displacement", "dI_A": (0.5 * radius * unit).tolist()}),
        ("bad_shape", {**base, "step_type": "current_increment", "dI_A": [radius]}),
    ]
    return requests


def plot_rows(rows: list[dict]) -> None:
    """Plot contract decisions."""
    configure_matplotlib_cjk()
    names = [row["case"] for row in rows]
    norms = [row["current_step_norm_A"] for row in rows]
    colors = ["#207a3c" if row["accepted"] else "#b33a3a" for row in rows]
    radius = max(row["max_linear_current_step_A"] for row in rows)
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    ax.bar(names, norms, color=colors)
    ax.axhline(radius, color="black", linestyle="--", linewidth=1.0, label=f"线性信赖域 {radius:g} A")
    ax.set_yscale("log")
    ax.set_ylabel("请求电流步长范数 (A)")
    ax.set_title("n2 信赖域接口契约验证")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.savefig(OUT_DIR / "n2_trust_radius_interface_contract_cn.png", dpi=220)
    plt.close(fig)


def main() -> None:
    """Run interface-contract validation."""
    t0 = time.perf_counter()
    interface = N2TrustRadiusSensitivityInterface(ROOT)
    snapshot = interface.as_dict()
    rows: list[dict] = []
    failures: list[dict] = []
    for name, request in build_requests(interface):
        decision = interface.validate_step_request(request)
        expected_accept = name in {"valid_half_radius", "valid_boundary_radius"}
        expectation_ok = decision.accepted == expected_accept
        if not expectation_ok:
            failures.append(
                {
                    "case": name,
                    "failure_class": "contract_expectation_mismatch",
                    "accepted": decision.accepted,
                    "reasons": decision.reasons,
                }
            )
        rows.append(
            {
                "case": name,
                "accepted": decision.accepted,
                "expected_accept": expected_accept,
                "expectation_ok": expectation_ok,
                "reasons": ";".join(decision.reasons),
                "current_step_norm_A": decision.current_step_norm_A,
                "max_linear_current_step_A": interface.max_linear_current_step_A(),
                "physical_current_step_budget_A": interface.physical_current_step_budget_A(),
            }
        )
    accepted_count = sum(row["accepted"] for row in rows)
    summary = {
        "experiment_id": OUT_DIR.name,
        "status": "contract_pass" if not failures else "contract_failed",
        "accepted_count": accepted_count,
        "total_cases": len(rows),
        "max_linear_current_step_A": interface.max_linear_current_step_A(),
        "physical_current_step_budget_A": interface.physical_current_step_budget_A(),
        "outer_ik_default_allowed": interface.can_use_for_default_outer_ik(),
        "trust_radius_evidence": interface.trust_radius_evidence(),
        "interface_snapshot": snapshot,
        "wall_s": time.perf_counter() - t0,
        "interpretation": "该接口将线性信赖域半径与物理电流预算分离，防止外层 IK 使用 5 A 级单步线性外推。",
    }
    write_csv(OUT_DIR / "validation_rows.csv", rows)
    write_csv(OUT_DIR / "summary.csv", [{k: v for k, v in summary.items() if k not in {"trust_radius_evidence", "interface_snapshot"}}])
    write_json(OUT_DIR / "summary.json", summary)
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in failures),
        encoding="utf-8",
    )
    plot_rows(rows)
    record = f"""# P2c117 n2 信赖域接口契约验证

## 实验目的

P2c115 证明 5 A 级单步线性外推不可用，P2c116 证明 0.0125 A 小电流半径内真实重求解闭环通过。本轮将该边界固化为只读接口契约，避免外层逆运动学误把物理电流预算当作线性信赖域。

## 接口规则

- 任务口径仍限定为两点磁控导管三维末端位置。
- 只允许同一模型、同一目标点、同一基准电流和同一可信分支。
- 最大线性电流步长半径：`{interface.max_linear_current_step_A():.6g} A`。
- 物理电流预算 `5 A` 只作为工程限幅参考，不能作为单步线性信赖域。
- 完整末端位姿、错误分支、错误任务维度和任务位移直接请求均被拒绝。

## 结果

- 验证请求数：{len(rows)}。
- 接受请求数：{accepted_count}。
- 期望不一致数：{len(failures)}。
- 默认接入外层 IK：`{interface.can_use_for_default_outer_ik()}`。
- 墙钟耗时：{summary["wall_s"]:.6f} s。

## 结论

若 `status=contract_pass`，说明接口已能阻止 5 A 级大步长误用，并把当前 n2 发布边界收缩为“可信分支条件下、三维末端位置、单步电流半径不超过 0.0125 A 的局部导数”。这仍不是全局灵敏度，也不是完整末端位姿灵敏度。

## 下一步计划

下一轮应基于该接口做小步迭代式外层 IK 原型：每步不超过 0.0125 A，执行后重新检查分支漂移和线性误差，禁止一次性大步跳转。
"""
    (OUT_DIR / "record.md").write_text(record, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
