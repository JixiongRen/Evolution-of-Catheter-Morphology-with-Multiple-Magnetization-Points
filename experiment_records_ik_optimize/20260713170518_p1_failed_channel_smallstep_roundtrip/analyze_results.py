"""Aggregate failed-channel sensitivity audit outputs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent


def _float(value: str, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def _point_label(point_dir: str) -> str:
    parts = Path(point_dir).parts
    catheter = ""
    point = Path(point_dir).name
    for idx, item in enumerate(parts):
        if item in ("A1", "A2"):
            catheter = item
            break
    return f"{catheter}-{point}" if catheter else point


def _classification(row: dict[str, str]) -> str:
    if not _bool(row.get("roundtrip_ok", "")) or not _bool(row.get("fd_valid", "")):
        return "回环失败"
    if _bool(row.get("weak_channel", "")):
        return "弱响应"
    rel = _float(row.get("relative_error", ""))
    if rel <= 0.10:
        return "通过"
    return "灵敏度偏差"


def _load_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(ROOT.glob("*/finite_difference_audit.csv")):
        case = path.parent.name
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                item = dict(row)
                item["case"] = case
                item["point"] = _point_label(item.get("point_dir", ""))
                item["classification"] = _classification(item)
                rows.append(item)
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _font_properties() -> fm.FontProperties | None:
    fallback_paths = [
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
        Path("/usr/share/fonts/truetype/unifont/unifont.ttf"),
    ]
    for path in fallback_paths:
        if path.exists():
            return fm.FontProperties(fname=str(path))
    preferred = [
        "Noto Sans CJK SC",
        "Noto Sans CJK",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "SimHei",
        "Microsoft YaHei",
    ]
    available = {font.name: font.fname for font in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            return fm.FontProperties(fname=available[name])
    return None


def _plot(rows: list[dict[str, str]]) -> None:
    font = _font_properties()
    plot_rows = sorted(rows, key=lambda r: (r["point"], int(float(r["channel"])), int(float(r["roundtrip_segments"])), _float(r["step"])))
    labels = [
        f"{r['point']} 通道{r['channel']} 段{r['roundtrip_segments']} 步{_float(r['step']):g}"
        for r in plot_rows
    ]
    rel = [_float(r.get("relative_error", "")) * 100.0 for r in plot_rows]
    colors = []
    for row in plot_rows:
        cls = row["classification"]
        if cls == "通过":
            colors.append("#168a2f")
        elif cls == "弱响应":
            colors.append("#dd8a00")
        elif cls == "回环失败":
            colors.append("#d92525")
        else:
            colors.append("#6b3fa0")

    fig, ax = plt.subplots(figsize=(13.5, max(6.0, 0.28 * len(labels))), dpi=170)
    y = list(range(len(labels)))
    ax.barh(y, rel, color=colors, alpha=0.85)
    ax.axvline(10.0, color="#202020", lw=1.1, ls="--", label="10% 门禁")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontproperties=font, fontsize=8)
    ax.set_xlabel("相对误差 (%)", fontproperties=font)
    ax.set_title("失败通道小步与分段回环审计", fontproperties=font, fontsize=13)
    ax.legend(prop=font)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROOT / "failed_channel_audit_summary.png", bbox_inches="tight")
    fig.savefig(ROOT / "failed_channel_audit_summary.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = _load_rows()
    compact_rows: list[dict[str, object]] = []
    for row in rows:
        compact_rows.append(
            {
                "case": row["case"],
                "point": row["point"],
                "channel": int(float(row["channel"])),
                "step": _float(row["step"]),
                "roundtrip_segments": int(float(row["roundtrip_segments"])),
                "svd_rcond": _float(row["svd_rcond"]),
                "fd_displacement_mm": _float(row["fd_displacement_mm"]),
                "roundtrip_ok": _bool(row["roundtrip_ok"]),
                "fd_valid": _bool(row["fd_valid"]),
                "weak_channel": _bool(row["weak_channel"]),
                "relative_error": _float(row["relative_error"]),
                "angle_deg": _float(row["angle_deg"]),
                "back_plus_terminal_mm": _float(row["back_plus_terminal_mm"]),
                "back_minus_terminal_mm": _float(row["back_minus_terminal_mm"]),
                "classification": row["classification"],
            }
        )

    _write_csv(ROOT / "aggregate_failed_channel_audit.csv", compact_rows)
    counts: dict[str, int] = {}
    for row in compact_rows:
        cls = str(row["classification"])
        counts[cls] = counts.get(cls, 0) + 1
    summary = {
        "row_count": len(compact_rows),
        "classification_counts": counts,
        "figure": str((ROOT / "failed_channel_audit_summary.png").resolve()),
    }
    (ROOT / "aggregate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
