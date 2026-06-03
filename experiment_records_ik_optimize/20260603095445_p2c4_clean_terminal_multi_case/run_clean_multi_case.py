"""Run clean terminal-pose shooting across several magnetic/current cases."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
P2C2_DIR = ROOT / "experiment_records_ik_optimize" / "20260602103750_p2c2_terminal_pose_constraint"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(P2C2_DIR) not in sys.path:
    sys.path.insert(0, str(P2C2_DIR))

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from forward_kinematics_optimized.fk_engine import ForwardKinematicsEngine

import run_p2c2_terminal_pose as p2c2

jax.config.update("jax_enable_x64", True)
OUT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CaseSpec:
    """One clean shooting validation case."""

    name: str
    m_body_list: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    currents: tuple[float, float, float, float, float, float, float, float]


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(ROOT), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc!r}"


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


def _build_engine(case: CaseSpec, *, max_iter: int, tol: float) -> ForwardKinematicsEngine:
    calib = ROOT / "calib" / "mpem_calibration_file_sp=40_order=1.yaml"
    return ForwardKinematicsEngine(
        flex_lengths_tail=(0.03, 0.03),
        rigid_lengths=(0.003, 0.003, 0.003),
        M_list=(1, 1, 1),
        L1_min=1e-6,
        L_protrude_max=0.11,
        p0_dim=(0.0, 0.0, 0.0),
        Q0_wxyz=(0.70710678, 0.0, 0.70710678, 0.0),
        axis_body=(0.0, 0.0, 1.0),
        enable_gravity=True,
        enable_magnetics=True,
        calib_file=str(calib),
        m_body_list=case.m_body_list,
        max_iter=int(max_iter),
        tol=float(tol),
        lm_damping=1e-1,
        jac_method="fwd",
        linear_solver="dense_fused_cholesky",
        scale_include_external=True,
    )


def _cases() -> list[CaseSpec]:
    m = 0.005301
    return [
        CaseSpec(
            "n3_mixed_base_i4",
            ((0.0, 0.0, -m), (0.0, 0.0, m), (m, 0.0, 0.0)),
            (0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0),
        ),
        CaseSpec(
            "n3_mixed_rand20",
            ((0.0, 0.0, -m), (0.0, 0.0, m), (m, 0.0, 0.0)),
            (12.0, -7.5, 4.0, -15.0, 9.0, 18.0, -3.5, 6.0),
        ),
        CaseSpec(
            "n2_z_pair_rand12",
            ((0.0, 0.0, -m), (0.0, 0.0, m), (0.0, 0.0, 0.0)),
            (-8.0, 5.0, 11.0, -4.0, 0.0, 9.5, -6.0, 3.0),
        ),
        CaseSpec(
            "n1_tip_z_rand15",
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, m)),
            (3.0, -12.0, 6.0, 0.0, 14.0, -9.0, 4.5, -2.0),
        ),
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _plot_case(case_dir: Path, oracle: dict[str, Any], engine: Any, outputs: dict[str, dict[str, np.ndarray]]) -> None:
    oracle_cl = np.asarray(oracle["centerline"], dtype=float)
    p_target, _q_target = p2c2._oracle_terminal_pose(oracle, engine)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(oracle_cl[:, 0], oracle_cl[:, 1], oracle_cl[:, 2], "k-", lw=2.0, label="FK oracle")
    for name, out in outputs.items():
        shoot = p2c2.p2c1._arc_resample(out["centerline"], len(oracle_cl))
        ax.plot(shoot[:, 0], shoot[:, 1], shoot[:, 2], "--", lw=2.0, label=name)
        ax.scatter([out["tip_p"][0]], [out["tip_p"][1]], [out["tip_p"][2]], s=35)
    ax.scatter([p_target[0]], [p_target[1]], [p_target[2]], c="green", s=60, label="target terminal")
    ax.scatter([oracle_cl[0, 0]], [oracle_cl[0, 1]], [oracle_cl[0, 2]], c="blue", s=35, label="base")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(case_dir.name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(case_dir / "oracle_vs_clean_shooting.png", dpi=180)
    plt.close(fig)


def _plot_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    labels = [f"{r['case']}\\n{r['init_mode']}" for r in rows]
    x = np.arange(len(rows))
    fig, ax1 = plt.subplots(figsize=(max(10, len(rows) * 1.25), 5))
    ax1.bar(x - 0.2, [r["terminal_position_error"] * 1000.0 for r in rows], width=0.4, label="terminal pos (mm)")
    ax1.set_ylabel("position error (mm)")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, [r["terminal_so3_error_deg"] for r in rows], width=0.4, color="tab:orange", label="SO3 (deg)")
    ax2.set_ylabel("orientation error (deg)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.grid(axis="y", alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "terminal_error_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    case_payloads = []
    for case in _cases():
        case_dir = OUT_DIR / case.name
        case_dir.mkdir(parents=True, exist_ok=True)
        engine = _build_engine(case, max_iter=120, tol=1e-5)
        oracle = p2c2.p2c1.p2c._solve_oracle(engine, np.asarray(case.currents, dtype=float), z0=None)
        length = float(np.sum(engine.rigid_lengths) + sum(engine._compute_flex_lengths(L_protrude=0.11)))
        spec = p2c2.PoseCandidateSpec(
            "clean_terminal_pose",
            max_nfev=16,
            centerline_weight=0.0,
            tip_weight=120.0,
            so3_weight=2.0,
            rigid_weight=0.0,
            boundary_weight=0.02,
        )
        y0_oracle = p2c2.p2c1._y_guess_from_oracle(oracle["z"], oracle["params"], "segment_wrench")
        init_modes = {
            "oracle_y0": y0_oracle,
            "zero_y0": np.zeros_like(y0_oracle),
        }
        outputs: dict[str, dict[str, np.ndarray]] = {}
        for init_mode, y0 in init_modes.items():
            result = p2c2._solve_pose_candidate(spec, oracle, engine, steps_per_flex=4, y0=y0)
            metrics = p2c2._terminal_metrics(oracle, engine, result.out)
            row = {
                "case": case.name,
                "init_mode": init_mode,
                "magnet_count_nonzero": int(sum(np.linalg.norm(np.asarray(v, dtype=float)) > 0.0 for v in case.m_body_list)),
                "currents": ",".join(f"{x:.3g}" for x in case.currents),
                "m_body_list": "; ".join(",".join(f"{v:.6g}" for v in m) for m in case.m_body_list),
                "pose_stage": p2c2._pose_stage(metrics, length),
                "success": bool(result.success),
                "wall_s": float(result.wall_s),
                "nfev": int(result.nfev),
                "cost": float(result.cost),
                "data_leakage": False,
                "uses_oracle_y0": bool(init_mode == "oracle_y0"),
                **metrics,
            }
            rows.append(row)
            outputs[init_mode] = result.out
            np.save(case_dir / f"{init_mode}_centerline.npy", result.out["centerline"])
            np.save(case_dir / f"{init_mode}_state_y.npy", result.y)
            np.save(case_dir / f"{init_mode}_residual_history.npy", np.asarray(result.residual_history, dtype=float))
        np.save(case_dir / "oracle_centerline.npy", np.asarray(oracle["centerline"], dtype=float))
        _plot_case(case_dir, oracle, engine, outputs)
        case_payloads.append(
            {
                "case": case.name,
                "oracle_ok": bool(oracle["ok"]),
                "oracle_normE": float(oracle["stats"].normE),
                "oracle_wall_s": float(oracle["wall_s"]),
                "currents": case.currents,
                "m_body_list": case.m_body_list,
            }
        )
    _plot_summary(OUT_DIR, rows)
    payload = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "M_list": [1, 1, 1],
            "clean_residuals": ["terminal_position", "terminal_so3", "physical_boundary"],
            "excluded_residuals": ["centerline", "rigid_midpoint"],
            "init_modes": ["oracle_y0", "zero_y0"],
            "total_wall_s": float(time.perf_counter() - t0),
        },
        "cases": case_payloads,
        "rows": rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "summary.csv", rows)
    print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
