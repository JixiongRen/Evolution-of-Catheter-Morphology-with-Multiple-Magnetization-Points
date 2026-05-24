"""Run multi-condition FK validation cases in one Python process."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.efficiency_optimization.fk_under_60s_driver import build_parser, run_continuation


OUT_DIR = Path(__file__).resolve().parent

MAGNET_CONFIGS = {
    "single_tip_z": "0,0,0; 0,0,0; 0,0,0.005301",
    "double_z_pair": "0,0,-0.005301; 0,0,0.005301; 0,0,0",
    "triple_mixed": "0,0,-0.005301; 0,0,0.005301; 0.005301,0,0",
}

CURRENTS = {
    "i15_c6": "0,0,0,0,0,15,0,0",
    "i50_c6": "0,0,0,0,0,50,0,0",
    "mixed": "10,-8,6,-4,2,-6,8,-10",
}

M_LISTS = {
    "M322": "3,2,2",
    "M555": "5,5,5",
}


def make_args(case_name: str, *, currents: str, magnets: str, m_list: str):
    base = [
        "--out_dir",
        str(OUT_DIR / case_name),
        "--case_name",
        case_name,
        "--mode",
        "continuation",
        "--classification",
        "multi-condition validation",
        "--coil_currents",
        currents,
        "--m_body_list",
        magnets,
        "--M_list",
        m_list,
        "--scale_mode",
        "optimized",
        "--scale_include_external",
        "--warmup",
        "--tol",
        "1e-5",
        "--lm_damping",
        "0.1",
        "--damping_mode",
        "identity",
        "--dense_solve_method",
        "pos",
        "--current_factors",
        "0,1.0",
        "--stage_max_iter",
        "20",
        "--stage_tol",
        "1e-3",
        "--final_max_iter",
        "1450",
        "--final_tol",
        "1e-5",
    ]
    return build_parser().parse_args(base)


def run_cases() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    (OUT_DIR / "batch_metadata.json").write_text(
        json.dumps(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "repo_root": str(REPO_ROOT),
                "jax_backend": jax.default_backend(),
                "jax_devices": [str(device) for device in jax.devices()],
                "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", ""),
                "case_count": len(MAGNET_CONFIGS) * len(CURRENTS) * len(M_LISTS),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    for m_name, m_list in M_LISTS.items():
        for mag_name, magnets in MAGNET_CONFIGS.items():
            for current_name, currents in CURRENTS.items():
                case_name = f"{m_name}_{mag_name}_{current_name}"
                case_dir = OUT_DIR / case_name
                case_dir.mkdir(parents=True, exist_ok=True)
                case_config = {
                    "case_name": case_name,
                    "M_list": m_list,
                    "magnet_config": mag_name,
                    "m_body_list": magnets,
                    "current_config": current_name,
                    "coil_currents": currents,
                }
                (case_dir / "case_config.json").write_text(json.dumps(case_config, ensure_ascii=False, indent=2), encoding="utf-8")
                start = time.perf_counter()
                try:
                    summary = run_continuation(make_args(case_name, currents=currents, magnets=magnets, m_list=m_list), case_dir)
                    summary["process_wall_s"] = float(time.perf_counter() - start)
                    summary["case_config"] = case_config
                    (case_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                    row = {
                        **case_config,
                        "ok": bool(summary["ok"]),
                        "wall_s": float(summary["wall_s"]),
                        "first_call_compile_s": float(summary["first_call_compile_s"]),
                        "n_iter": summary["n_iter"],
                        "final_normE": float(summary["final_normE"]),
                        "max_abs_E": float(summary["max_abs_E"]),
                        "end_x": summary["centerline_end_m"][0],
                        "end_y": summary["centerline_end_m"][1],
                        "end_z": summary["centerline_end_m"][2],
                    }
                except Exception as exc:
                    row = {
                        **case_config,
                        "ok": False,
                        "wall_s": None,
                        "first_call_compile_s": None,
                        "n_iter": None,
                        "final_normE": None,
                        "max_abs_E": None,
                        "end_x": None,
                        "end_y": None,
                        "end_z": None,
                        "error": repr(exc),
                    }
                    (case_dir / "error.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    return rows


def write_summary(rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row.keys()})
    with (OUT_DIR / "multi_condition_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "multi_condition_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_summary(rows: list[dict[str, object]]) -> None:
    valid = [row for row in rows if row.get("wall_s") is not None]
    labels = [str(row["case_name"]) for row in valid]
    wall = [float(row["wall_s"]) for row in valid]
    norm = [float(row["final_normE"]) for row in valid]

    fig, ax = plt.subplots(figsize=(13, 6))
    colors = ["tab:green" if row.get("ok") else "tab:red" for row in valid]
    ax.bar(np.arange(len(valid)), wall, color=colors)
    ax.axhline(60.0, color="black", linestyle="--", linewidth=1.2, label="60s target")
    ax.set_xticks(np.arange(len(valid)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("steady solve (s)")
    ax.set_title("Multi-condition FK runtime")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "multi_condition_runtime.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(np.arange(len(valid)), norm, color=colors)
    ax.axhline(1e-5, color="black", linestyle="--", linewidth=1.2, label="tol=1e-5")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(valid)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("final normE")
    ax.set_title("Multi-condition FK residual")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "multi_condition_residual.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    for row in valid:
        centerline_path = OUT_DIR / str(row["case_name"]) / "centerline.npy"
        if not centerline_path.exists():
            continue
        pts = np.load(centerline_path)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=1.2, alpha=0.8, label=str(row["case_name"]))
    ax.set_title("Multi-condition FK centerlines")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "multi_condition_centerlines.png", dpi=180)
    plt.close(fig)


def main() -> int:
    rows = run_cases()
    write_summary(rows)
    plot_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
