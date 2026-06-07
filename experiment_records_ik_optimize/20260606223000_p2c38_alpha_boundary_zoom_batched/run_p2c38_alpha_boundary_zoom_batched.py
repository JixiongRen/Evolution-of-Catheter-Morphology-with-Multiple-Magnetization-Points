"""Batch alpha-boundary zoom with isolated worker processes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk


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


def _run_git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(ROOT), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc!r}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def _run_worker(alpha: float, eps: str, python_bin: str, failures: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    alpha_dir = OUT_DIR / f"alpha_{alpha:.3f}"
    alpha_dir.mkdir(parents=True, exist_ok=True)
    log = alpha_dir / "stdout_stderr.log"
    cmd = [
        python_bin,
        str(OUT_DIR / "run_p2c38_single_alpha_worker.py"),
        "--alpha",
        f"{alpha:.6f}",
        "--eps",
        eps,
        "--out_dir",
        str(alpha_dir),
    ]
    env = dict(os.environ)
    env.update(
        {
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "JAX_PLATFORMS": "cpu",
            "JAX_PLATFORM_NAME": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
            "MPLCONFIGDIR": "/tmp/mcrs_mpl_cache",
        }
    )
    t0 = time.perf_counter()
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
    wall = time.perf_counter() - t0
    worker_summary = {
        "alpha": alpha,
        "returncode": proc.returncode,
        "wall_s": wall,
        "alpha_dir": str(alpha_dir),
        "log": str(log),
    }
    summary_path = alpha_dir / "summary.json"
    if proc.returncode != 0 or not summary_path.exists():
        _append_jsonl(failures, {"failure": "worker_failed", **worker_summary})
        return [], worker_summary
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = summary.get("eps_rows", [])
    for row in rows:
        row["worker_wall_s"] = wall
        row["alpha_dir"] = str(alpha_dir)
    worker_summary["publishable_eps_count"] = summary.get("overall", {}).get("publishable_eps_count")
    worker_summary["diagnostic_eps_count"] = summary.get("overall", {}).get("diagnostic_eps_count")
    return rows, worker_summary


def _plot(rows: list[dict[str, Any]]) -> None:
    configure_matplotlib_cjk()
    rows = sorted(rows, key=lambda r: float(r["alpha"]))
    alphas = [float(row["alpha"]) for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 9.0), sharex=True)
    axes[0].plot(alphas, [row["branch_rms_max"] for row in rows], marker="o")
    axes[0].axhline(1.0e-3, color="black", ls="--", lw=1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("分支漂移 / m")
    axes[1].plot(alphas, [row["max_perturb_stationarity"] for row in rows], marker="o")
    axes[1].axhline(1.0e-5, color="black", ls="--", lw=1)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("扰动后一阶最优性")
    axes[2].plot(alphas, [1 if row["overall_publishable"] else 0 for row in rows], marker="o")
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_ylabel("是否可发布")
    axes[2].set_xlabel("加载比例 alpha")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("eps=0.0125 可发布边界加密扫描")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alpha_0420_0425_boundary_zoom.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", default="0.421,0.422,0.423,0.424")
    parser.add_argument("--eps", default="0.0125")
    parser.add_argument("--python_bin", default="/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python")
    args = parser.parse_args()

    failures = OUT_DIR / "failure_events.jsonl"
    failures.write_text("", encoding="utf-8")
    t0 = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    worker_rows: list[dict[str, Any]] = []
    for alpha in [float(x.strip()) for x in args.alphas.split(",") if x.strip()]:
        rows, worker = _run_worker(alpha, args.eps, args.python_bin, failures)
        all_rows.extend(rows)
        worker_rows.append(worker)
    _write_csv(OUT_DIR / "boundary_zoom_eps_summary.csv", all_rows)
    _write_csv(OUT_DIR / "worker_summary.csv", worker_rows)
    if all_rows:
        _plot(all_rows)
    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": float(time.perf_counter() - t0),
        "args": vars(args),
        "git": {
            "branch": _run_git(["git", "branch", "--show-current"]),
            "commit": _run_git(["git", "rev-parse", "HEAD"]),
            "status_short": _run_git(["git", "status", "--short"]),
        },
        "overall": {
            "alpha_count": len(worker_rows),
            "completed_alpha_count": sum(row["returncode"] == 0 for row in worker_rows),
            "eps_count": len(all_rows),
            "publishable_eps_count": sum(row.get("overall_publishable", False) for row in all_rows),
            "diagnostic_eps_count": sum(not row.get("overall_publishable", False) for row in all_rows),
            "max_branch_rms": max((row["branch_rms_max"] for row in all_rows), default=None),
        },
        "worker_rows": worker_rows,
        "eps_rows": all_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload["overall"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
