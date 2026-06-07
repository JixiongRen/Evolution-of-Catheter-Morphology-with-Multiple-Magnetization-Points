"""Audit the scale dependence of interface load-propagation gates."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
P2C53_DIR = ROOT / "experiment_records_ik_optimize" / "20260607185000_p2c53_load_propagation_formula_audit"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

configure_matplotlib_cjk()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(ROOT), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc!r}"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        out = []
        for row in csv.DictReader(f):
            out.append({k: _coerce(v) for k, v in row.items()})
        return out


def _coerce(value: str) -> Any:
    try:
        return float(value)
    except ValueError:
        return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    current = [row for row in rows if row["formula"] == "当前刚段传播"]
    out = []
    for row in current:
        force_jump = float(row["jump_force_norm"])
        rigid_force = float(row["rigid_force_norm"])
        out.append(
            {
                "interface": int(row["interface"]),
                "force_jump_norm": force_jump,
                "rigid_force_norm": rigid_force,
                "force_ratio": float(row["force_ratio"]),
                "normalized_jump_norm": float(row["normalized_jump_norm"]),
                "ratio_if_force_floor_1e-6": force_jump / max(rigid_force, 1.0e-6),
                "ratio_if_force_floor_1e-5": force_jump / max(rigid_force, 1.0e-5),
                "ratio_if_force_floor_1e-4": force_jump / max(rigid_force, 1.0e-4),
                "ratio_if_force_floor_1e-3": force_jump / max(rigid_force, 1.0e-3),
            }
        )
    return out


def _plot(rows: list[dict[str, Any]]) -> None:
    labels = [f"接口 {int(row['interface'])}" for row in rows]
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9, 5))
    keys = [
        ("force_ratio", "原始相对比"),
        ("ratio_if_force_floor_1e-5", "分母下限 1e-5 N"),
        ("ratio_if_force_floor_1e-4", "分母下限 1e-4 N"),
        ("ratio_if_force_floor_1e-3", "分母下限 1e-3 N"),
    ]
    for i, (key, label) in enumerate(keys):
        ax.bar(x + (i - 1.5) * width, [row[key] for row in rows], width=width, label=label)
    ax.axhline(1.5, color="black", ls="--", lw=1.0)
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("接口力跳变比")
    ax.set_title("载荷传播门禁的分母尺度敏感性")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "load_gate_scale_audit_cn.png", dpi=180)
    plt.close(fig)


def _record(rows: list[dict[str, Any]]) -> str:
    table = [
        "| 接口 | 力跳变范数/N | 刚段外力范数/N | 原始力跳变比 | 分母下限1e-4后的比值 | 归一化跳变范数 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        table.append(
            f"| {int(row['interface'])} | {row['force_jump_norm']:.3e} | {row['rigid_force_norm']:.3e} | "
            f"{row['force_ratio']:.3e} | {row['ratio_if_force_floor_1e-4']:.3e} | {row['normalized_jump_norm']:.3e} |"
        )
    max_raw = max(row["force_ratio"] for row in rows)
    max_floor_1e4 = max(row["ratio_if_force_floor_1e-4"] for row in rows)
    max_floor_1e3 = max(row["ratio_if_force_floor_1e-3"] for row in rows)
    return "\n".join(
        [
            "# P2c54 载荷传播门禁尺度审计记录",
            "",
            "## 实验目的",
            "",
            "P2c53 排除了简单刚段外载符号错误，但仍显示接口力跳变比超过门禁。本轮不重新求解，只审计该比值对刚段外力分母尺度的敏感性，判断当前失败是否由小分母放大。",
            "",
            "## 数据来源",
            "",
            "使用 P2c53 中 alpha=0.10、P2c50 同源分支、当前刚段传播公式的接口数据。",
            "",
            "## 结果",
            "",
            *table,
            "",
            "## 结论",
            "",
            f"原始最大接口力跳变比为 `{max_raw:.3e}`；若对刚段外力分母加入 `1e-4 N` 下限，最大比值仍为 `{max_floor_1e4:.3e}`，说明当前刚段外力并非数值意义上的极小分母。若采用更接近接口绝对跳变量级的 `1e-3 N` 下限，最大比值降为 `{max_floor_1e3:.3e}`，接近但仍略高于 `1.5` 门限。由于原硬门禁仍未通过，该工况不能发布；失败机理应从“载荷传播公式错误”细化为“当前公式方向基本合理，但接口绝对跳变约为 1e-3 N，超过刚段外力数倍，且接口连续残差略高于门限”。",
            "",
            "## 下一步计划",
            "",
            "1. 保留原硬门禁用于发布判定，不放宽当前样本。",
            "2. 下一轮测试组合门禁：绝对跳变、相对外载跳变、相对内力跳变三者同时记录。",
            "3. 若组合门禁显示绝对闭合可接受，再考虑小扰动重求解诊断；否则转入接口约束尺度修复。",
            "",
        ]
    )


def main() -> None:
    rows = _read_rows(P2C53_DIR / "interface_formula_rows.csv")
    summary_rows = _summarize(rows)
    _plot(summary_rows)
    _write_csv(OUT_DIR / "summary.csv", summary_rows)
    payload = {
        "experiment": OUT_DIR.name,
        "source": str(P2C53_DIR),
        "rows": summary_rows,
        "git": {
            "branch": _run(["git", "branch", "--show-current"]),
            "commit": _run(["git", "rev-parse", "HEAD"]),
            "status_short": _run(["git", "status", "--short"]),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "failure_events.jsonl").write_text(
        json.dumps(_jsonable({"failure": "载荷传播门禁小分母敏感", "evidence": summary_rows}), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "record.md").write_text(_record(summary_rows), encoding="utf-8")
    print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
