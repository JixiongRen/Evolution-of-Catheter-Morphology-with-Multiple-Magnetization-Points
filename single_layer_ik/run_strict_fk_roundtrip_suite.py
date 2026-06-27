from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "experiment_records_ik_optimize" / "20260618_single_layer_ik_strict_fk_roundtrip"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    label: str
    target: str
    mode: str
    prior: str
    appendix_only: bool = False


MAIN_CASES = [
    CaseSpec("nomag_gravity", "无磁场重力", "0.100,0.000,-0.060", "direct_nomag", "无磁场"),
    CaseSpec("t2_coil0_prior", "T2 0号线圈先验", "0.058,0.000,-0.118", "direct_mag_c0", "0号线圈约15A"),
    CaseSpec("t2_auto_multicoil", "T2 多线圈先验", "0.058,0.000,-0.118", "auto_mag", "自动多线圈"),
    CaseSpec("t1_auto_multicoil", "T1 多线圈先验", "0.050,0.000,-0.123", "auto_mag", "自动多线圈"),
    CaseSpec("t4_auto_multicoil", "T4 多线圈先验", "0.064,0.000,-0.115", "auto_mag", "自动多线圈"),
    CaseSpec("appendix_t2_coil4_prior", "T2 4号线圈先验", "0.058,0.000,-0.118", "direct_mag_c4", "4号线圈约15A", True),
]


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(cmd: list[str], *, cwd: Path, log_path: Path, skip_if_done: Path | None = None) -> int:
    if skip_if_done is not None and skip_if_done.exists():
        print(f"[suite] skip existing: {skip_if_done}")
        return 0
    print("[suite] run:", " ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(cwd), env=_child_env(), stdout=log, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        print(f"[suite] command failed rc={proc.returncode}; log={log_path}")
    return int(proc.returncode)


def _base_direct_args(*, out_dir: Path, target: str, run_mode: str, m_body_list: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "single_layer_ik.run_joint_ik",
        f"--run_mode={run_mode}",
        f"--p_des={target}",
        "--flex_lengths_tail=0.03,0.03",
        "--rigid_lengths=0.003,0.003,0.003",
        "--M_list=3,2,2",
        "--x_min=0.06",
        "--x_max=0.18",
        "--L_protrude_max=0.18",
        "--L1_min=1e-6",
        "--enforce_feasible_x_bounds",
        "--x_init=0.1",
        "--init_mode=equilibrium",
        "--init_fk_max_iter=400",
        "--init_fk_tol=1e-4",
        "--init_fk_lm_damping=0.01",
        "--p0_dim=0,0,-0.05",
        "--Q0_wxyz=1,0,0,0",
        "--axis_body=1,0,0",
        "--enable_gravity",
        "--g_world=0,0,-9.81",
        "--I_max=50",
        "--I_scale=1.0",
        "--w_E=100",
        "--sigma_p=0.005",
        "--w_I=1.0",
        "--w_x=0.001",
        "--x_ref=0.1",
        "--w_smooth=0.2",
        "--w_smooth_baseline=0.2",
        "--w_gap=2.0",
        "--sigma_gap=0.0005",
        "--centerline_rigid_samples=2",
        "--max_iter=220",
        "--tol=1e-6",
        "--dist_tol_mm=1.0",
        "--lm_damping=0.01",
        "--lam_max=1e10",
        "--gtol=1e-12",
        "--xtol=1e-12",
        "--stagnation_patience=100",
        "--reject_patience=30",
        "--step_norm_clip=50",
        "--backtrack_max=6",
        "--backtrack_factor=0.5",
        "--block_clip_dz=20",
        "--block_clip_dI=10",
        "--block_clip_dx=0.015",
        "--accept_kappa_E=1.05",
        "--accept_eps_E=1e-4",
        "--accept_eps_smooth=0.0",
        "--stage1_sigma_scale=2.5",
        "--stage1_w_smooth_scale=0.8",
        "--stage1_block_clip_scale=0.75",
        "--stage1_max_iter=70",
        "--trace_block_topk=0",
        "--trace_block_every=0",
        "--pose_save_every=20",
        f"--m_body_list={m_body_list}",
        f"--actuation_table_pkl=supiee_auto_diff/offline_interpolation_data/actuation_tables/actuation_table.pkl",
        f"--out_dir={out_dir}",
    ]


def _direct_nomag_args(*, out_dir: Path, target: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "single_layer_ik.run_joint_ik",
        "--run_mode=v1_staged",
        f"--p_des={target}",
        "--flex_lengths_tail=0.03,0.03",
        "--rigid_lengths=0.003,0.003,0.003",
        "--M_list=3,2,2",
        "--x_min=0.06",
        "--x_max=0.18",
        "--L_protrude_max=0.18",
        "--L1_min=1e-6",
        "--enforce_feasible_x_bounds",
        "--x_init=0.1",
        "--init_mode=geometry",
        "--p0_dim=0,0,-0.05",
        "--Q0_wxyz=1,0,0,0",
        "--axis_body=1,0,0",
        "--enable_gravity",
        "--g_world=0,0,-9.81",
        "--I_max=5",
        "--I_init=0,0,0,0,0,0,0,0",
        "--w_E=30",
        "--sigma_p=0.01",
        "--w_I=0",
        "--w_x=0.001",
        "--x_ref=0.1",
        "--w_smooth=5",
        "--w_smooth_baseline=0",
        "--max_iter=80",
        "--tol=1e-6",
        "--dist_tol_mm=1.0",
        "--lm_damping=0.01",
        "--stage1_max_iter=24",
        "--stage1_sigma_scale=4",
        "--trace_block_topk=0",
        "--trace_block_every=0",
        "--pose_save_every=10",
        f"--out_dir={out_dir}",
    ]


def _auto_args(*, out_dir: Path, target: str, m_body_list: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "single_layer_ik.auto_run_joint_ik",
        f"--p_des={target}",
        "--run_mode=v1_staged",
        "--show_plot=false" if False else "--plot_max_lines=200",
        f"--out_dir={out_dir}",
        "--flex_lengths_tail=0.03,0.03",
        "--rigid_lengths=0.003,0.003,0.003",
        "--M_list=3,2,2",
        "--x_min=0.06",
        "--x_max=0.18",
        "--L_protrude_max=0.18",
        "--L1_min=1e-6",
        "--enforce_feasible_x_bounds",
        "--x_init=0.1",
        "--init_mode=equilibrium",
        "--init_fk_max_iter=400",
        "--init_fk_tol=1e-4",
        "--init_fk_lm_damping=0.01",
        "--Q0_wxyz=1,0,0,0",
        "--axis_body=1,0,0",
        "--enable_gravity",
        "--enable_magnetics",
        "--g_world=0,0,-9.81",
        f"--m_body_list={m_body_list}",
        "--I_magnitude=15",
        "--I_max=50",
        "--min_coil_fraction=0.05",
        "--I_ref_scale=0.5",
        "--max_attempts=1",
        "--w_E=100",
        "--sigma_p=0.005",
        "--w_I=1.0",
        "--w_x=0.001",
        "--x_ref=0.1",
        "--I_scale=1.0",
        "--w_smooth=0.2",
        "--w_smooth_baseline=0.2",
        "--w_gap=2.0",
        "--sigma_gap=0.0005",
        "--centerline_rigid_samples=2",
        "--max_iter=220",
        "--tol=1e-6",
        "--dist_tol_mm=1.0",
        "--lm_damping=0.01",
        "--stage1_max_iter=70",
        "--trace_block_topk=0",
        "--trace_block_every=0",
        "--pose_save_every=20",
    ]


def _case_command(case: CaseSpec, run_dir: Path, m_body_list: str) -> list[str]:
    if case.mode == "direct_nomag":
        return _direct_nomag_args(out_dir=run_dir, target=case.target)
    if case.mode == "direct_mag_c0":
        return _base_direct_args(out_dir=run_dir, target=case.target, run_mode="v1_staged", m_body_list=m_body_list) + [
            "--enable_magnetics",
            "--I_init=15,0,0,0,0,0,0,0",
            "--I_ref=15,0,0,0,0,0,0,0",
        ]
    if case.mode == "direct_mag_c4":
        return _base_direct_args(out_dir=run_dir, target=case.target, run_mode="v1_staged", m_body_list=m_body_list) + [
            "--enable_magnetics",
            "--I_init=0,0,0,0,15,0,0,0",
            "--I_ref=0,0,0,0,15,0,0,0",
        ]
    if case.mode == "auto_mag":
        return _auto_args(out_dir=run_dir, target=case.target, m_body_list=m_body_list)
    raise ValueError(f"Unknown case mode: {case.mode}")


def _load_trace(run_dir: Path) -> list[dict[str, Any]]:
    trace_path = run_dir / "trace.jsonl"
    if not trace_path.exists():
        return []
    return [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _top_current(I: list[float]) -> str:
    arr = np.asarray(I, dtype=float).reshape(-1)
    idx = np.argsort(np.abs(arr))[::-1][:2]
    return "; ".join(f"I{int(i)}={arr[i]:.3g}A" for i in idx if abs(arr[i]) > 1e-6)


def _collect_summary(out_dir: Path, cases: list[CaseSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        run_dir = out_dir / case.name
        result_path = run_dir / "result.json"
        rt_path = run_dir / "fk_roundtrip_summary.json"
        if not result_path.exists():
            continue
        result = _read_json(result_path)
        rt = _read_json(rt_path) if rt_path.exists() else {"classification": "missing_roundtrip", "rows": []}
        row_state = next((r for r in rt.get("rows", []) if r.get("mode") == "state_inherited"), {})
        row_cold = next((r for r in rt.get("rows", []) if r.get("mode") == "cold_start"), {})
        max_rms = max([float(r.get("ik_fk_centerline_rms_mm", float("nan"))) for r in rt.get("rows", [])] or [float("nan")])
        rows.append(
            {
                "case": case.name,
                "label": case.label,
                "prior": case.prior,
                "appendix_only": case.appendix_only,
                "dominant_current": _top_current(result.get("I_A", [])),
                "x_mm": float(result.get("x_m", float("nan"))) * 1e3,
                "ik_target_err_mm": float(result.get("diag", {}).get("pos_err_norm_mm", float("nan"))),
                "state_fk_target_err_mm": float(row_state.get("fk_target_err_mm", float("nan"))),
                "cold_fk_target_err_mm": float(row_cold.get("fk_target_err_mm", float("nan"))),
                "max_ik_fk_centerline_rms_mm": max_rms,
                "classification": rt.get("classification", "missing_roundtrip"),
            }
        )
    return rows


def _write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    if rows:
        with (out_dir / "strict_fk_roundtrip_results.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    md_lines = [
        "| 工况 | 先验 | 主导电流 | 插入深度/mm | IK目标误差/mm | 状态继承FK目标误差/mm | 冷启动FK目标误差/mm | 最大中心线RMS/mm | 判定 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['label']} | {r['prior']} | {r['dominant_current']} | {r['x_mm']:.3f} | "
            f"{r['ik_target_err_mm']:.4g} | {r['state_fk_target_err_mm']:.4g} | {r['cold_fk_target_err_mm']:.4g} | "
            f"{r['max_ik_fk_centerline_rms_mm']:.4g} | {r['classification']} |"
        )
    (out_dir / "strict_fk_roundtrip_results.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def _plot_controls(out_dir: Path, cases: list[CaseSpec]) -> None:
    plot_cases = [c for c in cases if not c.appendix_only]
    fig, axes = plt.subplots(len(plot_cases), 3, figsize=(13.5, 2.6 * len(plot_cases)), constrained_layout=True)
    if len(plot_cases) == 1:
        axes = np.asarray([axes])
    for row_idx, case in enumerate(plot_cases):
        trace = _load_trace(out_dir / case.name)
        if not trace:
            continue
        it = np.asarray([r["iter"] for r in trace], dtype=float)
        I = np.asarray([r["I_A"] for r in trace], dtype=float)
        x = np.asarray([r["x_m"] for r in trace], dtype=float) * 1e3
        dist = np.asarray([r["dist_to_target_mm"] for r in trace], dtype=float)
        ax = axes[row_idx, 0]
        for i in range(8):
            ax.plot(it, I[:, i], lw=1.0, alpha=0.75, label=f"I{i}" if row_idx == 0 else None)
        ax.set_ylabel(case.label)
        ax.grid(True, alpha=0.25)
        if row_idx == 0:
            ax.legend(ncols=4, fontsize=8, frameon=False)
        ax = axes[row_idx, 1]
        ax.plot(it, x, color="#D55E00", lw=1.8)
        ax.set_ylabel("x (mm)")
        ax.grid(True, alpha=0.25)
        ax = axes[row_idx, 2]
        ax.plot(it, dist, color="#0072B2", lw=1.8)
        ax.set_yscale("log")
        ax.set_ylabel("末端误差/mm")
        ax.grid(True, alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("迭代步")
    fig.savefig(out_dir / "fig_strict_ik_controls.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig_strict_ik_controls.svg", bbox_inches="tight")
    plt.close(fig)


def _plot_roundtrip_pose(out_dir: Path, cases: list[CaseSpec]) -> None:
    plot_cases = [c for c in cases if not c.appendix_only]
    fig, axes = plt.subplots(1, len(plot_cases), figsize=(3.25 * len(plot_cases), 3.2), constrained_layout=True)
    if len(plot_cases) == 1:
        axes = np.asarray([axes])
    for ax, case in zip(axes, plot_cases):
        run_dir = out_dir / case.name
        cl_path = run_dir / "fk_roundtrip_centerlines.npz"
        state_path = run_dir / "state_final.npz"
        if not cl_path.exists() or not state_path.exists():
            continue
        cls = np.load(cl_path)
        state = np.load(state_path)
        target = state["p_des"] * 1e3
        for key, color, label in [
            ("ik_centerline", "0.2", "IK"),
            ("state_inherited_centerline", "#0072B2", "状态继承FK"),
            ("cold_start_centerline", "#D55E00", "冷启动FK"),
        ]:
            if key in cls.files:
                c = cls[key] * 1e3
                ax.plot(c[:, 0], c[:, 2], lw=1.8, color=color, label=label)
        ax.scatter(target[0], target[2], marker="*", s=70, color="#C51B29", label="目标")
        ax.set_title(case.label, fontsize=9)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm)")
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8, frameon=False)
    fig.savefig(out_dir / "fig_strict_fk_roundtrip_pose.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig_strict_fk_roundtrip_pose.svg", bbox_inches="tight")
    plt.close(fig)


def _plot_metrics(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    rows = [r for r in rows if not r["appendix_only"]]
    if not rows:
        return
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    vals = {
        "状态继承目标误差": [r["state_fk_target_err_mm"] for r in rows],
        "冷启动目标误差": [r["cold_fk_target_err_mm"] for r in rows],
        "最大中心线RMS": [r["max_ik_fk_centerline_rms_mm"] for r in rows],
    }
    fig, ax = plt.subplots(figsize=(10.8, 4.2), constrained_layout=True)
    width = 0.25
    for j, (label, data) in enumerate(vals.items()):
        ax.bar(x + (j - 1) * width, data, width=width, label=label)
    ax.axhline(0.1, color="0.35", ls="--", lw=1.2, label="姿态一致性阈值 0.1 mm")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("指标值 (mm)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out_dir / "fig_strict_fk_roundtrip_metrics.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig_strict_fk_roundtrip_metrics.svg", bbox_inches="tight")
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run strict single-layer IK + FK round-trip verification suite.")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--m_body_list", type=str, default="0,0,0.005301; 0,0,0.005301; 0,0,0.005301")
    p.add_argument("--include_appendix_c4", action="store_true")
    p.add_argument("--skip_ik", action="store_true")
    p.add_argument("--skip_fk", action="store_true")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--plot_only", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = MAIN_CASES if bool(args.include_appendix_c4) else [c for c in MAIN_CASES if not c.appendix_only]
    (out_dir / "suite_config.json").write_text(
        json.dumps(
            {
                "m_body_list": args.m_body_list,
                "cases": [case.__dict__ for case in cases],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if not args.plot_only:
        for case in cases:
            run_dir = out_dir / case.name
            run_dir.mkdir(parents=True, exist_ok=True)
            if not args.skip_ik:
                _run(
                    _case_command(case, run_dir, str(args.m_body_list)),
                    cwd=REPO_ROOT,
                    log_path=run_dir / "ik_run.log",
                    skip_if_done=(run_dir / "state_final.npz") if args.skip_existing else None,
                )
            if not args.skip_fk:
                _run(
                    [
                        sys.executable,
                        "-m",
                        "single_layer_ik.fk_roundtrip_verify",
                        "--run_dir",
                        str(run_dir),
                        "--max_iter",
                        "6000",
                        "--tol",
                        "1e-6",
                        "--lm_damping",
                        "0.01",
                        "--perturb_repeats",
                        "3",
                        "--short_circuit_on_fail",
                    ],
                    cwd=REPO_ROOT,
                    log_path=run_dir / "fk_roundtrip.log",
                    skip_if_done=(run_dir / "fk_roundtrip_summary.csv") if args.skip_existing else None,
                )

    rows = _collect_summary(out_dir, cases)
    if rows:
        _write_summary(out_dir, rows)
        _plot_controls(out_dir, cases)
        _plot_roundtrip_pose(out_dir, cases)
        _plot_metrics(out_dir, rows)
    print(f"[suite] wrote: {out_dir}")


if __name__ == "__main__":
    main()
