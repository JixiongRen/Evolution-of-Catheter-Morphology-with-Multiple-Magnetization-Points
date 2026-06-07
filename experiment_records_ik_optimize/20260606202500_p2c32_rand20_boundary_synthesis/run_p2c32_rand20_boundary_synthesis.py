"""Synthesize rand20 sensitivity boundary evidence from prior experiments."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mcrs_mpl_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk


OUT_DIR = Path(__file__).resolve().parent
P2C16 = ROOT / "experiment_records_ik_optimize" / "20260605132600_p2c16_alpha_boundary_phase_map"
P2C30 = ROOT / "experiment_records_ik_optimize" / "20260606194500_p2c30_rand20_multi_eps_recheck"
P2C31 = ROOT / "experiment_records_ik_optimize" / "20260606201000_p2c31_rand20_eps_boundary_refine"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    configure_matplotlib_cjk()
    alpha_rows_raw = _read_csv(P2C16 / "summary.csv")
    eps_rows_raw = _read_csv(P2C31 / "combined_p2c30_p2c31_boundary_summary.csv")

    alpha_rows: list[dict[str, Any]] = []
    for row in alpha_rows_raw:
        alpha_rows.append(
            {
                "source": "P2c16",
                "alpha": float(row["alpha"]),
                "eps_alpha": float(row["eps_alpha"]),
                "classification": row["classification"],
                "publishable": _as_bool(row["publishable_gate"]),
                "stationarity": float(row["max_stage_stationarity"]),
                "branch_rms": float(row["branch_rms_max"]),
                "slack_force_ratio": float(row["max_slack_force_over_rigid_force"]),
                "linearization_error": float(row["max_linearization_error"]),
                "sensitivity_norm": float(row["sensitivity_norm_m_per_alpha"]),
            }
        )

    eps_rows: list[dict[str, Any]] = []
    for row in eps_rows_raw:
        eps_rows.append(
            {
                "source": "P2c30/P2c31",
                "alpha": 0.4,
                "eps_alpha": float(row["eps_alpha"]),
                "classification": "可发布" if _as_bool(row["overall_publishable"]) else "分支漂移诊断",
                "publishable": _as_bool(row["overall_publishable"]),
                "stationarity": float(row["max_perturb_stationarity"]),
                "branch_rms": float(row["branch_rms_max"]),
                "linearization_error": float(row["linearization_error_max"]),
                "sensitivity_x": float(row["sens_x"]),
                "sensitivity_y": float(row["sens_y"]),
                "sensitivity_z": float(row["sens_z"]),
            }
        )

    _write_csv(OUT_DIR / "alpha_evidence_summary.csv", alpha_rows)
    _write_csv(OUT_DIR / "eps_boundary_summary.csv", eps_rows)

    failures: list[dict[str, Any]] = []
    for row in alpha_rows:
        if not row["publishable"]:
            failures.append(
                {
                    "source": row["source"],
                    "alpha": row["alpha"],
                    "eps_alpha": row["eps_alpha"],
                    "classification": row["classification"],
                    "dominant_failure": "载荷传播失配或路径依赖" if row["slack_force_ratio"] > 2.0 else "路径依赖或证据不足",
                    "evidence": row,
                }
            )
    for row in eps_rows:
        if not row["publishable"]:
            failures.append(
                {
                    "source": row["source"],
                    "alpha": row["alpha"],
                    "eps_alpha": row["eps_alpha"],
                    "classification": row["classification"],
                    "dominant_failure": "分支漂移",
                    "evidence": row,
                }
            )
    with (OUT_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for row in failures:
            f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 8.0))
    color_by_class = {"可信工况": "#2ca02c", "临界工况": "#ff7f0e", "不可用工况": "#d62728"}
    colors = [color_by_class.get(row["classification"], "#7f7f7f") for row in alpha_rows]
    axes[0].scatter([r["alpha"] for r in alpha_rows], [r["sensitivity_norm"] for r in alpha_rows], c=colors, s=72)
    axes[0].plot([r["alpha"] for r in alpha_rows], [r["sensitivity_norm"] for r in alpha_rows], color="0.35", lw=1)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("加载比例 alpha")
    axes[0].set_ylabel("敏感度模长 / (m/alpha)")
    axes[0].set_title("不同加载程度下的可信性分类")
    axes[0].grid(alpha=0.25)
    for label, color in color_by_class.items():
        axes[0].scatter([], [], c=color, label=label)
    axes[0].legend(ncol=3)

    eps = [r["eps_alpha"] for r in eps_rows]
    colors = ["#2ca02c" if r["publishable"] else "#d62728" for r in eps_rows]
    axes[1].scatter(eps, [r["branch_rms"] for r in eps_rows], c=colors, s=72)
    axes[1].plot(eps, [r["branch_rms"] for r in eps_rows], color="0.35", lw=1)
    axes[1].axhline(1.0e-3, color="black", ls="--", lw=1, label="分支漂移门限")
    axes[1].axvspan(0.01, 0.0125, color="#ffcc66", alpha=0.25, label="已定位临界区间")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("扰动半径 eps")
    axes[1].set_ylabel("轴线分支漂移 / m")
    axes[1].set_title("alpha=0.4 附近扰动半径失效边界")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rand20_boundary_synthesis.png", dpi=180)
    plt.close(fig)

    summary = {
        "experiment": OUT_DIR.name,
        "sources": [str(P2C16), str(P2C30), str(P2C31)],
        "alpha_evidence": {
            "trusted_alpha_interval_observed": [0.4, 0.425],
            "failed_low_alpha_observed": [0.35, 0.375],
            "critical_high_alpha_observed": [0.45],
        },
        "eps_boundary_at_alpha_0p4": {
            "publishable_eps_max_observed": 0.01,
            "failed_eps_min_observed": 0.0125,
            "critical_interval": [0.01, 0.0125],
            "dominant_failure": "branch drift",
        },
        "failure_count": len(failures),
        "recommendations": [
            "Do not publish sensitivities for n3_mixed_rand20 at alpha=0.4 with eps greater than 0.01.",
            "Use alpha=0.40-0.425 as the current local trusted loading window, pending same-protocol replication.",
            "Run the next experiment with the P2c30/P2c31 refinement protocol across alpha values to remove protocol mismatch.",
        ],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
