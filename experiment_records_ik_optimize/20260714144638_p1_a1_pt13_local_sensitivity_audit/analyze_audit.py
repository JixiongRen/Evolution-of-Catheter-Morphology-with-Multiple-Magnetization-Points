"""Summarize the local sensitivity audit for A1-pt13."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "audit_output"
FD_PATH = AUDIT / "finite_difference_audit.csv"
SENS_PATH = AUDIT / "sensitivity_modes.csv"
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def _p90(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[int(math.ceil(0.9 * len(values))) - 1]


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    fd_rows = _read_csv(FD_PATH)
    sens_rows = _read_csv(SENS_PATH)
    modes = sorted({row["mode"] for row in fd_rows})
    mode_rows: list[dict[str, object]] = []
    for mode in modes:
        rows = [row for row in fd_rows if row["mode"] == mode]
        used = [row for row in rows if _bool(row.get("used_in_stats", ""))]
        rel = [float(row["relative_error"]) for row in used]
        angle = [float(row["angle_deg"]) for row in used]
        mode_rows.append(
            {
                "mode": mode,
                "total_rows": len(rows),
                "branch_consistent_rows": sum(_bool(row.get("branch_consistent", "")) for row in rows),
                "used_rows": len(used),
                "relative_error_median": _median(rel),
                "relative_error_p90": _p90(rel),
                "angle_deg_median": _median(angle),
                "angle_deg_p90": _p90(angle),
            }
        )

    spec_rows: list[dict[str, object]] = []
    for key in sorted({(row["kind"], row["channel"], row["step"]) for row in fd_rows}):
        rows = [row for row in fd_rows if (row["kind"], row["channel"], row["step"]) == key]
        first = rows[0]
        best = next((row for row in rows if row["mode"] == "gn_svd"), first)
        spec_rows.append(
            {
                "kind": key[0],
                "channel": key[1],
                "step": key[2],
                "fd_displacement_mm": float(first["fd_displacement_mm"]),
                "roundtrip_ok": _bool(first["roundtrip_ok"]),
                "weak_gate_triggered": _bool(first["weak_gate_triggered"]),
                "used_in_stats": _bool(first["used_in_stats"]),
                "back_plus_terminal_mm": float(first["back_plus_terminal_mm"]),
                "back_minus_terminal_mm": float(first["back_minus_terminal_mm"]),
                "gn_svd_relative_error": float(best["relative_error"]),
                "gn_svd_angle_deg": float(best["angle_deg"]),
            }
        )

    summary = {
        "point": "A1-pt13",
        "base_selector": "best_accept",
        "base_error_mm": float(sens_rows[0]["base_err_mm"]) if sens_rows else None,
        "terminal_reload_error_mm": float(sens_rows[0]["terminal_reload_error_mm"]) if sens_rows else None,
        "centerline_reload_max_error_mm": float(sens_rows[0]["centerline_reload_max_error_mm"]) if sens_rows else None,
        "mode_summary": mode_rows,
        "spec_summary": spec_rows,
        "interpretation": {
            "best_mode": "gn_svd",
            "strict_gate_passed": False,
            "main_reason": "gn_svd is directionally useful but p90 relative error is above the 10% gate; several finite-difference perturbations fail the branch roundtrip gate.",
            "safe_local_channels": "I0 <= 0.125 A, I6 <= 0.25 A, x <= 0.5 mm are the clearest same-branch candidates in this source state.",
            "unsafe_or_diagnostic_channels": "I4 fails roundtrip at all tested steps; I5 is weak and fails at larger steps; I3 is weak at smaller steps.",
        },
    }
    (ROOT / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(ROOT / "mode_summary.csv", mode_rows)
    _write_csv(ROOT / "channel_step_summary.csv", spec_rows)

    font = fm.FontProperties(fname=str(FONT_PATH)) if FONT_PATH.exists() else None
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    labels = [row["mode"] for row in mode_rows]
    med = [row["relative_error_median"] or 0.0 for row in mode_rows]
    p90 = [row["relative_error_p90"] or 0.0 for row in mode_rows]
    x = range(len(labels))
    axes[0].bar([v - 0.18 for v in x], med, width=0.36, label="中位数")
    axes[0].bar([v + 0.18 for v in x], p90, width=0.36, label="90分位")
    axes[0].axhline(0.05, color="green", linestyle="--", linewidth=1, label="目标中位数5%")
    axes[0].axhline(0.10, color="red", linestyle="--", linewidth=1, label="目标90分位10%")
    axes[0].set_xticks(list(x), labels, rotation=20)
    axes[0].set_ylabel("相对误差", fontproperties=font)
    axes[0].set_title("三种灵敏度公式误差对比", fontproperties=font)
    axes[0].legend(prop=font)

    plot_rows = spec_rows
    labels2 = [f"{row['kind']}{row['channel']}\\n{row['step']}" for row in plot_rows]
    disp = [float(row["fd_displacement_mm"]) for row in plot_rows]
    colors = [
        "#2ca02c" if row["roundtrip_ok"] and not row["weak_gate_triggered"] else "#ffbf00" if row["roundtrip_ok"] else "#d62728"
        for row in plot_rows
    ]
    axes[1].bar(range(len(plot_rows)), disp, color=colors)
    axes[1].axhline(0.25, color="gray", linestyle="--", linewidth=1, label="弱响应门限0.25mm")
    axes[1].set_xticks(range(len(plot_rows)), labels2, rotation=60, ha="right")
    axes[1].set_ylabel("单侧末端位移(mm)", fontproperties=font)
    axes[1].set_title("通道扰动有效性与回环状态", fontproperties=font)
    axes[1].legend(prop=font)
    fig.suptitle("A1-pt13局部灵敏度审计", fontproperties=font, fontsize=15)
    fig.tight_layout()
    fig.savefig(ROOT / "sensitivity_audit_cn.png", dpi=180)
    fig.savefig(ROOT / "sensitivity_audit_cn.svg")


if __name__ == "__main__":
    main()
