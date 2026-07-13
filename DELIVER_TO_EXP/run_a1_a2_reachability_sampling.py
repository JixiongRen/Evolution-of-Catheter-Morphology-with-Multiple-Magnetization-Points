"""Sample A1/A2 control space for supervised IK reachability diagnosis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from forward_kinematics_optimized.fk_engine import ForwardKinematicsEngine
from inverse_kinematics.ik_diff import extract_terminal_p_dim
from DELIVER_TO_EXP.run_a1_a2_diagnostic_closure import (
    _build_engine_from_run_config,
    _configure_fonts,
    _font,
    _history_states,
    _read_json,
    _write_csv,
)


jax.config.update("jax_enable_x64", True)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_RUN = (
    ROOT
    / "formal_experiments"
    / "20260713063500_A1_A2_5mm_supervised_IK_no_gravity_merged_complete40"
)
DEFAULT_OUT_ROOT = ROOT / "formal_diagnostics" / "reachability_sampling"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def _jax_env() -> dict[str, Any]:
    devices = [str(dev) for dev in jax.devices()]
    platforms = sorted({getattr(dev, "platform", "unknown") for dev in jax.devices()})
    return {"devices": devices, "platforms": platforms, "has_gpu": "gpu" in platforms}


def _point_dir(source_run: Path, catheter: str, index: int) -> Path:
    return source_run / catheter / "formal_20pt" / f"pt_{index:02d}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _halton_value(index: int, base: int) -> float:
    f = 1.0
    r = 0.0
    i = int(index)
    while i > 0:
        f /= float(base)
        r += f * (i % base)
        i //= base
    return r


def _halton(index: int, dim: int) -> np.ndarray:
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    return np.asarray([_halton_value(index + 1, primes[i]) for i in range(dim)], dtype=float)


def _control_from_unit(unit: np.ndarray, *, x_min: float, x_max: float) -> tuple[np.ndarray, float]:
    I = -20.0 + 40.0 * np.asarray(unit[:8], dtype=float)
    x = float(x_min + (x_max - x_min) * float(unit[8]))
    return I, x


def _target_rows(source_run: Path) -> list[dict[str, Any]]:
    rows = []
    for row in _read_csv(source_run / "summary.csv"):
        catheter = row["catheter_type"]
        index = int(row["target_index"])
        result = _read_json(_point_dir(source_run, catheter, index) / "result.json", {})
        target = np.asarray(result.get("p_des", [math.nan, math.nan, math.nan]), dtype=float)
        rows.append({"catheter_type": catheter, "target_index": index, "target_m": target})
    return rows


def _nearest_history_controls(
    targets: list[dict[str, Any]],
    history_states: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    by_catheter: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in history_states:
        if state.get("I_A") is None or not math.isfinite(float(state.get("x_m", math.nan))):
            continue
        by_catheter[str(state["catheter_type"])].append(state)

    out: dict[tuple[str, int], dict[str, Any]] = {}
    for target in targets:
        catheter = str(target["catheter_type"])
        target_p = np.asarray(target["target_m"], dtype=float)
        best_state = None
        best_d = math.inf
        for state in by_catheter.get(catheter, []):
            d = float(np.linalg.norm(np.asarray(state["terminal_m"], dtype=float) - target_p) * 1000.0)
            if d < best_d:
                best_d = d
                best_state = state
        if best_state is not None:
            out[(catheter, int(target["target_index"]))] = {
                "distance_mm": best_d,
                "I_A": np.asarray(best_state["I_A"], dtype=float),
                "x_m": float(best_state["x_m"]),
                "source_target_index": int(best_state["source_target_index"]),
                "source_iter": int(best_state["iter"]),
            }
    return out


def _load_engine(source_run: Path, catheter: str) -> tuple[ForwardKinematicsEngine, tuple[float, float]]:
    point = _point_dir(source_run, catheter, 0)
    run_config = _read_json(point / "run_config.json", {})
    engine = _build_engine_from_run_config(run_config)
    bounds = run_config.get("x_bounds", [0.003001, 0.203])
    return engine, (float(bounds[0]), float(bounds[1]))


def _solve_sample(
    engine: ForwardKinematicsEngine,
    *,
    I_A: np.ndarray,
    x_m: float,
) -> tuple[np.ndarray, str, float, bool]:
    z, params, _meshes, ok, stats = engine.solve_with_stats(
        coil_currents=jnp.asarray(I_A, dtype=jnp.float64),
        L_protrude=float(x_m),
        warm_start=True,
        return_stats=True,
        stats_include_jacobian=False,
    )
    p = np.asarray(jax.device_get(extract_terminal_p_dim(z, M_list=params.M_list, scales=params.scales)), dtype=float)
    norm_e = float(stats.normE) if stats is not None else math.nan
    cls = "GOOD" if bool(ok) and norm_e <= 1e-5 else ("WEAK" if norm_e <= 1e-2 else "BAD")
    return p, cls, norm_e, bool(ok)


def _build_sample_plan(
    *,
    source_run: Path,
    global_samples_per_catheter: int,
    local_samples_per_uncovered: int,
    local_current_radius_A: float,
    local_x_radius_m: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    targets, history = _history_states(source_run)
    target_rows = _target_rows(source_run)
    nearest = _nearest_history_controls(target_rows, history)
    history_reachability = []
    for target in targets:
        key = (target["catheter_type"], int(target["target_index"]))
        near = nearest.get(key, {})
        d = float(near.get("distance_mm", math.inf))
        if bool(target["success_5mm"]):
            conclusion = "目标自身已经达标"
        elif d <= 10.0:
            conclusion = "历史近可达"
        elif d <= 30.0:
            conclusion = "历史边界可达"
        else:
            conclusion = "历史未覆盖"
        history_reachability.append(
            {
                "catheter_type": key[0],
                "target_index": key[1],
                "history_nearest_mm": d,
                "history_conclusion": conclusion,
            }
        )

    plan: list[dict[str, Any]] = []
    for catheter in ("A1", "A2"):
        _, (x_min, x_max) = _load_engine(source_run, catheter)
        for i in range(int(global_samples_per_catheter)):
            I, x = _control_from_unit(_halton(i, 9), x_min=x_min, x_max=x_max)
            plan.append(
                {
                    "sample_type": "global_halton",
                    "catheter_type": catheter,
                    "target_index": "",
                    "sample_index": i,
                    "I_A": I,
                    "x_m": x,
                    "source_history_distance_mm": "",
                }
            )

    local_seed = 0
    for row in history_reachability:
        if row["history_conclusion"] != "历史未覆盖":
            continue
        key = (row["catheter_type"], int(row["target_index"]))
        near = nearest.get(key)
        if near is None:
            continue
        _, (x_min, x_max) = _load_engine(source_run, key[0])
        for j in range(int(local_samples_per_uncovered)):
            u = _halton(local_seed, 9) - 0.5
            local_seed += 1
            I = np.clip(np.asarray(near["I_A"], dtype=float) + 2.0 * local_current_radius_A * u[:8], -20.0, 20.0)
            x = float(np.clip(float(near["x_m"]) + 2.0 * local_x_radius_m * u[8], x_min, x_max))
            plan.append(
                {
                    "sample_type": "local_uncovered",
                    "catheter_type": key[0],
                    "target_index": key[1],
                    "sample_index": j,
                    "I_A": I,
                    "x_m": x,
                    "source_history_distance_mm": near["distance_mm"],
                }
            )
    return plan, history_reachability


def _evaluate_samples(source_run: Path, plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    engines: dict[str, ForwardKinematicsEngine] = {}
    rows = []
    for sample_id, sample in enumerate(plan):
        catheter = str(sample["catheter_type"])
        if catheter not in engines:
            engines[catheter], _bounds = _load_engine(source_run, catheter)
        t0 = time.perf_counter()
        p, cls, norm_e, ok = _solve_sample(
            engines[catheter],
            I_A=np.asarray(sample["I_A"], dtype=float),
            x_m=float(sample["x_m"]),
        )
        rows.append(
            {
                "sample_id": sample_id,
                "sample_type": sample["sample_type"],
                "catheter_type": catheter,
                "target_index": sample["target_index"],
                "sample_index": sample["sample_index"],
                "x_m": float(sample["x_m"]),
                "I_A_json": json.dumps([float(v) for v in np.asarray(sample["I_A"]).reshape(-1)]),
                "terminal_x_m": float(p[0]),
                "terminal_y_m": float(p[1]),
                "terminal_z_m": float(p[2]),
                "cls": cls,
                "normE": norm_e,
                "ok": bool(ok),
                "wall_s": time.perf_counter() - t0,
                "source_history_distance_mm": sample["source_history_distance_mm"],
            }
        )
        print(f"[D4] {catheter} sample {sample_id + 1}/{len(plan)} {sample['sample_type']} cls={cls} wall={rows[-1]['wall_s']:.1f}s", flush=True)
    return rows


def _classify_targets(
    target_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_catheter: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in sample_rows:
        by_catheter[str(sample["catheter_type"])].append(sample)
    hist_by_key = {(row["catheter_type"], int(row["target_index"])): row for row in history_rows}

    out = []
    for target in target_rows:
        key = (target["catheter_type"], int(target["target_index"]))
        target_p = np.asarray(target["target_m"], dtype=float)
        best_sample = None
        best_d = math.inf
        for sample in by_catheter.get(key[0], []):
            p = np.asarray([sample["terminal_x_m"], sample["terminal_y_m"], sample["terminal_z_m"]], dtype=float)
            d = float(np.linalg.norm(p - target_p) * 1000.0)
            if d < best_d:
                best_d = d
                best_sample = sample
        hist = hist_by_key.get(key, {})
        hist_d = float(hist.get("history_nearest_mm", math.inf))
        if best_d <= 5.0 or hist_d <= 5.0:
            conclusion = "采样或历史证据可达"
        elif best_d <= 10.0 or hist_d <= 10.0:
            conclusion = "采样或历史证据近可达"
        elif best_d <= 30.0 or hist_d <= 30.0:
            conclusion = "边界可达或优化不足"
        else:
            conclusion = "采样仍未覆盖"
        out.append(
            {
                "catheter_type": key[0],
                "target_index": key[1],
                "history_nearest_mm": hist_d,
                "sample_nearest_mm": best_d,
                "nearest_sample_id": "" if best_sample is None else int(best_sample["sample_id"]),
                "nearest_sample_type": "" if best_sample is None else str(best_sample["sample_type"]),
                "d4_conclusion": conclusion,
            }
        )
    return out


def _plot_sampling(run_dir: Path, target_rows: list[dict[str, Any]], sample_rows: list[dict[str, Any]]) -> None:
    font = _font()
    for catheter in ("A1", "A2"):
        targets = [row for row in target_rows if row["catheter_type"] == catheter]
        samples = [row for row in sample_rows if row["catheter_type"] == catheter]
        if not targets:
            continue
        fig = plt.figure(figsize=(8.0, 6.8), dpi=180)
        ax = fig.add_subplot(111, projection="3d")
        t = np.asarray([row["target_m"] for row in targets], dtype=float) * 1000.0
        ax.plot(t[:, 0], t[:, 1], t[:, 2], color="#333333", lw=1.2, label="目标圆环")
        ax.scatter(t[:, 0], t[:, 1], t[:, 2], color="#d92525", s=35, marker="x", label="目标点")
        if samples:
            s = np.asarray([[row["terminal_x_m"], row["terminal_y_m"], row["terminal_z_m"]] for row in samples], dtype=float) * 1000.0
            colors = ["#168a2f" if row["sample_type"] == "local_uncovered" else "#4263eb" for row in samples]
            ax.scatter(s[:, 0], s[:, 1], s[:, 2], c=colors, s=28, alpha=0.85, label="新增采样末端")
        ax.set_xlabel("x (mm)", fontproperties=font)
        ax.set_ylabel("y (mm)", fontproperties=font)
        ax.set_zlabel("z (mm)", fontproperties=font)
        ax.set_title(f"{catheter} 新增控制采样可达性", fontproperties=font)
        ax.legend(prop=font)
        fig.tight_layout()
        fig.savefig(run_dir / f"{catheter}_sampled_reachability.png", bbox_inches="tight")
        fig.savefig(run_dir / f"{catheter}_sampled_reachability.svg", bbox_inches="tight")
        plt.close(fig)


def run_sampling(args: argparse.Namespace) -> Path:
    _configure_fonts()
    source_run = Path(args.source_run).resolve()
    run_dir = Path(args.out_root).resolve() / f"{datetime.now().strftime('%Y%m%d%H%M%S')}_A1_A2_reachability_sampling"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = _jax_env()
    if not env["has_gpu"]:
        raise RuntimeError(f"GPU is required for this diagnostic, got devices: {env}")

    target_rows = _target_rows(source_run)
    plan, history_rows = _build_sample_plan(
        source_run=source_run,
        global_samples_per_catheter=int(args.global_samples_per_catheter),
        local_samples_per_uncovered=int(args.local_samples_per_uncovered),
        local_current_radius_A=float(args.local_current_radius_A),
        local_x_radius_m=float(args.local_x_radius_m),
    )
    _write_csv(
        run_dir / "sample_plan.csv",
        [
            {
                **{k: v for k, v in row.items() if k != "I_A"},
                "I_A_json": json.dumps([float(x) for x in np.asarray(row["I_A"]).reshape(-1)]),
            }
            for row in plan
        ],
    )
    sample_rows = _evaluate_samples(source_run, plan)
    target_summary = _classify_targets(target_rows, sample_rows, history_rows)
    _write_csv(run_dir / "sample_results.csv", sample_rows)
    _write_csv(run_dir / "target_reachability_summary.csv", target_summary)

    np.savez_compressed(
        run_dir / "reachability_samples.npz",
        sample_terminal_points_m=np.asarray(
            [[row["terminal_x_m"], row["terminal_y_m"], row["terminal_z_m"]] for row in sample_rows],
            dtype=float,
        ),
        target_points_m=np.asarray([row["target_m"] for row in target_rows], dtype=float),
    )
    _plot_sampling(run_dir, target_rows, sample_rows)

    summary = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_run": str(source_run),
            "git_commit": _git_commit(),
            "python": sys.executable,
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "jax": env,
        },
        "sample_count": len(sample_rows),
        "sample_type_counts": dict(Counter(row["sample_type"] for row in sample_rows)),
        "target_conclusion_counts": dict(Counter(row["d4_conclusion"] for row in target_summary)),
        "by_catheter": {
            catheter: dict(Counter(row["d4_conclusion"] for row in target_summary if row["catheter_type"] == catheter))
            for catheter in ("A1", "A2")
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_record(run_dir, source_run, summary)
    command = (
        f"XLA_PYTHON_CLIENT_PREALLOCATE=false {sys.executable} {Path(__file__).resolve()} "
        f"--source-run {source_run} --out-root {Path(args.out_root).resolve()} "
        f"--global-samples-per-catheter {args.global_samples_per_catheter} "
        f"--local-samples-per-uncovered {args.local_samples_per_uncovered} "
        f"--local-current-radius-A {args.local_current_radius_A} "
        f"--local-x-radius-m {args.local_x_radius_m}"
    )
    (run_dir / "commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n", encoding="utf-8")
    print(f"sampling_dir: {run_dir}", flush=True)
    return run_dir


def _write_record(run_dir: Path, source_run: Path, summary: dict[str, Any]) -> None:
    sample_type_zh = {
        "global_halton": "全局低差异采样",
        "local_uncovered": "历史未覆盖点局部加密采样",
    }
    lines = [
        "# A1/A2 有监督逆运动学 D4 可达性采样报告",
        "",
        "## 一、目的",
        "",
        "本报告补充独立控制空间采样和局部加密采样，用于判断失败目标是优化器未找到、边界可达，还是当前采样仍未覆盖。该采样不能证明严格不可达。",
        "",
        "## 二、边界条件",
        "",
        f"- 源实验目录：`{source_run}`",
        f"- 输出目录：`{run_dir}`",
        f"- git 版本：`{summary['metadata']['git_commit']}`",
        f"- JAX 设备：`{summary['metadata']['jax']}`",
        "- 基线保持无重力、`M_list=5`、FK 容差 `1e-5`、电流范围 `[-20,20] A`、推进范围 `[0.003001,0.203] m`。",
        "",
        "## 三、采样规模",
        "",
        f"- 新增样本数：{summary['sample_count']}。",
    ]
    for key, value in summary["sample_type_counts"].items():
        lines.append(f"- {sample_type_zh.get(key, key)}：{value} 个。")
    lines.extend(["", "## 四、D4 分类结论", ""])
    for catheter in ("A1", "A2"):
        lines.append(f"### {catheter}")
        for key, value in summary["by_catheter"].get(catheter, {}).items():
            lines.append(f"- {key}：{value} 点。")
        lines.append("")
    lines.extend(
        [
            "## 五、解释",
            "",
            "1. 若目标被新增采样或历史轨迹覆盖到 5 mm 内，归为已有可达证据；若进入 10 mm，归为近可达；若进入 30 mm，归为边界可达或优化不足。",
            "2. 采样仍未覆盖的点不能写成物理不可达，只能说明当前样本规模不足或目标处于更难到达的区域。",
            "3. 该 D4 结果应与 D1 灵敏度质量和 D3 分支可复现性一起解释，不能单独作为 IK 正确性结论。",
            "",
            "## 六、文件索引",
            "",
            "- `sample_plan.csv`：新增控制样本。",
            "- `sample_results.csv`：每个样本的 FK 结果。",
            "- `target_reachability_summary.csv`：每个目标的 D4 分类。",
            "- `reachability_samples.npz`：新增样本点和目标点。",
            "- `A1_sampled_reachability.png`、`A2_sampled_reachability.png`：三维采样可达性图。",
        ]
    )
    (run_dir / "record.md").write_text("\n".join(lines), encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--global-samples-per-catheter", type=int, default=2)
    parser.add_argument("--local-samples-per-uncovered", type=int, default=1)
    parser.add_argument("--local-current-radius-A", type=float, default=2.0)
    parser.add_argument("--local-x-radius-m", type=float, default=0.01)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    t0 = time.perf_counter()
    run_dir = run_sampling(args)
    print(f"record: {run_dir / 'record.md'}", flush=True)
    print(f"wall_s: {time.perf_counter() - t0:.3f}", flush=True)


if __name__ == "__main__":
    main()
