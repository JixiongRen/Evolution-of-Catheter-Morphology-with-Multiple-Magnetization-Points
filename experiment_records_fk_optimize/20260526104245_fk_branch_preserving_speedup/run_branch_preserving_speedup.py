from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[2]
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(FK_OPT_DIR) not in sys.path:
    sys.path.insert(0, str(FK_OPT_DIR))

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jspl
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from equilibrium_solver_nondim import residual_bar, with_L1_dim, with_coil_currents
from experiments.efficiency_optimization.fk_under_60s_driver import _extract_centerline
from fk_engine import ForwardKinematicsEngine
from rod_mesh_nondim import build_uniform_mesh
from utils_nondim import make_initial_guess_multi_bar_jax


TARGET_CURRENTS = (0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0)
TARGET_M_BODY = (
    (0.0, 0.0, -0.005301),
    (0.0, 0.0, 0.005301),
    (0.005301, 0.0, 0.0),
)
GOLDEN_DIR = ROOT / "experiment_records_fk_optimize" / "20260525094651_fk_new_tol_sweep" / "tol_1e-05"


@dataclass(frozen=True)
class Candidate:
    name: str
    mode: str
    linear_solver: str = "dense"
    dense_solve_method: str = "solve"
    damping_mode: str = "identity"
    step_scale: float = 1.0
    current_factors: tuple[float, ...] = (1.0,)
    stage_max_iter: int = 0
    stage_tol: float = 1e-3
    final_max_iter: int = 100000
    classification_hint: str = "branch-preserving result"


CANDIDATES = (
    Candidate(
        name="A1_cached_direct_solve",
        mode="direct",
        dense_solve_method="solve",
        classification_hint="branch-preserving result",
    ),
    Candidate(
        name="B1_cached_direct_pos",
        mode="direct",
        dense_solve_method="pos",
        classification_hint="branch-preserving result",
    ),
    Candidate(
        name="B2_cached_direct_cholesky",
        mode="direct",
        dense_solve_method="cholesky",
        classification_hint="branch-preserving result",
    ),
    Candidate(
        name="F1_fused_cholesky_direct",
        mode="fused_direct",
        dense_solve_method="cholesky",
        classification_hint="branch-preserving fused-jit result",
    ),
    Candidate(
        name="F2_engine_fused_cholesky_direct",
        mode="direct",
        linear_solver="dense_fused_cholesky",
        dense_solve_method="cholesky",
        classification_hint="production fused-jit result",
    ),
    Candidate(
        name="E1_direct_cholesky_scale125",
        mode="direct",
        dense_solve_method="cholesky",
        step_scale=1.25,
        classification_hint="branch-preserving overrelaxation result",
    ),
    Candidate(
        name="D1_short_continuation_solve",
        mode="continuation",
        dense_solve_method="solve",
        current_factors=(0.0, 1.0),
        stage_max_iter=20,
        stage_tol=1e-3,
        final_max_iter=3000,
        classification_hint="high-risk continuation result",
    ),
    Candidate(
        name="D2_short_continuation_pos",
        mode="continuation",
        dense_solve_method="pos",
        current_factors=(0.0, 1.0),
        stage_max_iter=20,
        stage_tol=1e-3,
        final_max_iter=3000,
        classification_hint="high-risk continuation result",
    ),
)


def block(value: Any) -> Any:
    if hasattr(value, "block_until_ready"):
        return value.block_until_ready()
    return value


def repo_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip()


def rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def get_calib_file() -> tuple[Path, str]:
    requested = Path("/abs/path/calibration.json")
    if requested.exists():
        return requested, "requested_path_exists"
    return ROOT / "calib" / "mpem_calibration_file_sp=40_order=1.yaml", "requested_path_missing_used_repo_fallback"


def build_engine(candidate: Candidate, calib_file: Path, *, max_iter: int, tol: float) -> ForwardKinematicsEngine:
    return ForwardKinematicsEngine(
        flex_lengths_tail=(0.03, 0.03),
        rigid_lengths=(0.003, 0.003, 0.003),
        M_list=(5, 5, 5),
        L1_min=1e-6,
        flex_d_outer=(0.0015,),
        flex_E=(1.8e6,),
        flex_G=(0.6e6,),
        flex_rho=(970.0,),
        rigid_d_outer=(0.0015,),
        rigid_rho=(7500.0,),
        p0_dim=(0.0, 0.0, 0.0),
        Q0_wxyz=(0.70710678, 0.0, 0.70710678, 0.0),
        axis_body=(0.0, 0.0, 1.0),
        enable_gravity=True,
        enable_magnetics=True,
        calib_file=str(calib_file),
        actuation_table_pkl=str(
            ROOT / "supiee_auto_diff" / "offline_interpolation_data" / "actuation_tables" / "actuation_table.pkl"
        ),
        m_body_list=TARGET_M_BODY,
        max_iter=max_iter,
        tol=tol,
        lm_damping=1e-1,
        jac_method="fwd",
        linear_solver=candidate.linear_solver,
        damping_mode=candidate.damping_mode,
        dense_solve_method=candidate.dense_solve_method,
        step_scale=float(candidate.step_scale),
        scale_mode="optimized",
        scale_include_external=True,
        L_protrude_max=0.11,
    )


def pointwise_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    count = min(len(reference), len(candidate))
    ref = np.asarray(reference[:count], dtype=np.float64)
    cand = np.asarray(candidate[:count], dtype=np.float64)
    delta = np.linalg.norm(ref - cand, axis=1)
    max_err = float(np.max(delta))
    return {
        "point_count": int(count),
        "max_point_error_m": max_err,
        "rms_point_error_m": float(np.sqrt(np.mean(delta * delta))),
        "mean_point_error_m": float(np.mean(delta)),
        "end_error_m": float(delta[-1]),
        "max_rel_to_L_ref": float(max_err / 0.11),
        "end_rel_to_L_ref": float(delta[-1] / 0.11),
    }


def make_initial_state(engine: ForwardKinematicsEngine, currents: jnp.ndarray) -> tuple[Any, Any]:
    engine._ensure_static_initialized(L_protrude=0.11, coil_currents=currents)
    assert engine._params_static_base is not None
    assert engine._meshes_ref is not None
    flex_lengths = engine._compute_flex_lengths(L_protrude=0.11)
    params = with_coil_currents(engine._params_static_base, currents)
    params = with_L1_dim(params, jnp.asarray(float(flex_lengths[0]), dtype=jnp.float64))
    meshes = list(engine._meshes_ref)
    meshes[0] = build_uniform_mesh(float(flex_lengths[0]), int(engine.M_list[0]))
    z0_bar, *_ = make_initial_guess_multi_bar_jax(
        flex_segs=list(params.flex),
        meshes=meshes,
        rigid_segs=list(params.rigid),
        scales=params.scales,
        p0_dim=engine.p0_dim,
        Q0=engine.Q0,
        axis_body=engine.axis_body,
    )
    return z0_bar, params


@jax.jit
def fused_cholesky_lm(z0_bar: Any, params: Any, max_iter: Any, tol: Any, lm_damping: Any):
    """Run the dense LM loop inside one XLA while loop."""

    def residual_fn(z_bar: Any) -> Any:
        return residual_bar(z_bar, params)

    def jacobian_fn(z_bar: Any) -> Any:
        return jax.jacfwd(residual_fn)(z_bar)

    z = jnp.asarray(z0_bar, dtype=jnp.float64)
    E = residual_fn(z)
    normE = jnp.linalg.norm(E)
    cost = 0.5 * normE * normE
    n = z.shape[0]
    eye = jnp.eye(n, dtype=z.dtype)
    max_iter_i = jnp.asarray(max_iter, dtype=jnp.int32)
    tol_f = jnp.asarray(tol, dtype=z.dtype)
    lam0 = jnp.asarray(lm_damping, dtype=z.dtype)
    init = (
        jnp.asarray(0, dtype=jnp.int32),
        z,
        E,
        normE,
        cost,
        lam0,
        cost,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )

    def cond_fun(carry: Any) -> Any:
        it, _z, _E, current_norm, _cost, lam, _best, no_improve, rejects = carry
        return (
            (it < max_iter_i)
            & (current_norm >= tol_f)
            & (no_improve < jnp.asarray(20, dtype=jnp.int32))
            & (rejects < jnp.asarray(10, dtype=jnp.int32))
            & (lam < jnp.asarray(1e10, dtype=z.dtype))
        )

    def body_fun(carry: Any) -> Any:
        it, z_cur, E_cur, norm_cur, cost_cur, lam, best_cost, no_improve, rejects = carry
        J = jacobian_fn(z_cur)
        H = jnp.transpose(J) @ J
        g = jnp.transpose(J) @ E_cur
        A = 0.5 * (H + lam * eye + jnp.transpose(H + lam * eye))
        dz = jspl.cho_solve(jspl.cho_factor(A, lower=True, check_finite=False), -g, check_finite=False)
        z_trial = z_cur + dz
        E_trial = residual_fn(z_trial)
        norm_trial = jnp.linalg.norm(E_trial)
        cost_trial = 0.5 * norm_trial * norm_trial
        finite_trial = jnp.isfinite(norm_trial) & jnp.isfinite(cost_trial)
        act_red = cost_cur - cost_trial
        pred_red = 0.5 * (dz @ (lam * dz - g))
        rho = act_red / (pred_red + jnp.asarray(1e-30, dtype=z.dtype))
        accept = finite_trial & (act_red > 0.0)

        z_next = jnp.where(accept, z_trial, z_cur)
        E_next = jnp.where(accept, E_trial, E_cur)
        norm_next = jnp.where(accept, norm_trial, norm_cur)
        cost_next = jnp.where(accept, cost_trial, cost_cur)
        improved = cost_next < best_cost - jnp.maximum(
            jnp.asarray(1e-18, dtype=z.dtype),
            jnp.asarray(1e-12, dtype=z.dtype) * best_cost,
        )
        best_next = jnp.where(accept & improved, cost_next, best_cost)
        no_improve_next = jnp.where(
            accept,
            jnp.where(improved, jnp.asarray(0, dtype=jnp.int32), no_improve + 1),
            no_improve,
        )
        rejects_next = jnp.where(accept, jnp.asarray(0, dtype=jnp.int32), rejects + 1)
        lam_accept = jnp.where(
            rho > 0.75,
            lam * jnp.asarray(0.3, dtype=z.dtype),
            jnp.where(rho < 0.25, lam * jnp.asarray(10.0, dtype=z.dtype), lam),
        )
        lam_next = jnp.where(accept, lam_accept, lam * jnp.asarray(10.0, dtype=z.dtype))
        lam_next = jnp.minimum(jnp.maximum(lam_next, jnp.asarray(1e-30, dtype=z.dtype)), jnp.asarray(1e10, dtype=z.dtype))
        return it + 1, z_next, E_next, norm_next, cost_next, lam_next, best_next, no_improve_next, rejects_next

    return jax.lax.while_loop(cond_fun, body_fun, init)


def classify(summary: dict[str, Any], metrics: dict[str, Any]) -> str:
    if not summary["ok"] or summary["final_normE"] > 1e-5:
        return "failed result"
    if metrics["max_point_error_m"] > 0.011:
        return "speed-only result"
    if metrics["max_point_error_m"] > 0.002:
        return "branch-drift-risk result"
    if summary["speedup_vs_golden"] >= 3.0 and metrics["max_point_error_m"] <= 0.011:
        return "production-candidate result"
    return "branch-preserving result"


def run_solve_stage(
    engine: ForwardKinematicsEngine,
    *,
    currents: jnp.ndarray,
    warm_start: bool,
    max_iter: int,
    tol: float,
) -> tuple[Any, Any, dict[str, Any]]:
    engine.max_iter = int(max_iter)
    engine.tol = float(tol)
    start = time.perf_counter()
    z_star, params, _meshes, ok, stats = engine.solve_with_stats(
        coil_currents=currents,
        L_protrude=0.11,
        warm_start=warm_start,
        return_stats=True,
        stats_include_jacobian=False,
        verbose=False,
    )
    block(z_star)
    wall_s = time.perf_counter() - start
    E = residual_bar(z_star, params)
    block(E)
    row = {
        "wall_s": float(wall_s),
        "ok": bool(ok and float(jnp.linalg.norm(E)) < float(tol)),
        "stop_reason": str(stats.stop_reason),
        "n_iter": int(stats.n_iter),
        "normE": float(jnp.linalg.norm(E)),
        "max_abs_E": float(jnp.max(jnp.abs(E))),
        "max_iter": int(max_iter),
        "tol": float(tol),
    }
    return z_star, params, row


def sample_convergence(
    engine: ForwardKinematicsEngine,
    out_dir: Path,
    *,
    currents: jnp.ndarray,
    max_iter: int,
    tol: float,
) -> None:
    engine.reset_warm_start()
    engine.max_iter = int(max_iter)
    engine.tol = float(tol)
    engine._ensure_static_initialized(L_protrude=0.11, coil_currents=currents)
    z0, params, _meshes, _ok, _stats = engine.solve_with_stats(
        coil_currents=currents,
        L_protrude=0.11,
        warm_start=False,
        return_stats=True,
        stats_include_jacobian=False,
        verbose=False,
    )
    centerline = _extract_centerline(z0, params, rigid_sample_count=10)
    np.savez_compressed(out_dir / "convergence_sample_final_only.npz", centerline=centerline)


def run_candidate(candidate: Candidate, out_dir: Path, golden_centerline: np.ndarray, golden_wall_s: float) -> dict[str, Any]:
    case_dir = out_dir / candidate.name
    case_dir.mkdir(parents=True, exist_ok=True)
    calib_file, _reason = get_calib_file()
    target = jnp.asarray(TARGET_CURRENTS, dtype=jnp.float64)
    engine = build_engine(candidate, calib_file, max_iter=100000, tol=1e-5)

    warmup = {} if candidate.mode == "fused_direct" else engine.warm_up(coil_currents=target, L_protrude=0.11)
    stages: list[dict[str, Any]] = []

    if candidate.mode == "direct":
        z_star, params, row = run_solve_stage(
            engine,
            currents=target,
            warm_start=False,
            max_iter=candidate.final_max_iter,
            tol=1e-5,
        )
        row["stage"] = "direct"
        row["current_factor"] = 1.0
        stages.append(row)
    elif candidate.mode == "fused_direct":
        z0_bar, params = make_initial_state(engine, target)
        compile_start = time.perf_counter()
        warmup_result = fused_cholesky_lm(
            z0_bar,
            params,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(1e-5, dtype=jnp.float64),
            jnp.asarray(1e-1, dtype=jnp.float64),
        )
        block(warmup_result[1])
        fused_compile_s = float(time.perf_counter() - compile_start)

        solve_start = time.perf_counter()
        result = fused_cholesky_lm(
            z0_bar,
            params,
            jnp.asarray(candidate.final_max_iter, dtype=jnp.int32),
            jnp.asarray(1e-5, dtype=jnp.float64),
            jnp.asarray(1e-1, dtype=jnp.float64),
        )
        block(result[1])
        wall_s = float(time.perf_counter() - solve_start)
        n_iter, z_star, E_final, norm_final, _cost, lam_final, _best, no_improve, rejects = result
        row = {
            "wall_s": wall_s,
            "ok": bool(float(norm_final) < 1e-5),
            "stop_reason": "converged" if float(norm_final) < 1e-5 else "fused_loop_stop",
            "n_iter": int(n_iter),
            "normE": float(norm_final),
            "max_abs_E": float(jnp.max(jnp.abs(E_final))),
            "max_iter": int(candidate.final_max_iter),
            "tol": 1e-5,
            "stage": "fused_direct",
            "current_factor": 1.0,
            "lam": float(lam_final),
            "no_improve": int(no_improve),
            "rejects": int(rejects),
        }
        stages.append(row)
        warmup = {"fused_compile_s": fused_compile_s, "total_s": fused_compile_s}
    elif candidate.mode == "continuation":
        z_star = params = None
        for idx, factor in enumerate(candidate.current_factors):
            z_star, params, row = run_solve_stage(
                engine,
                currents=target * float(factor),
                warm_start=(idx > 0),
                max_iter=candidate.stage_max_iter,
                tol=candidate.stage_tol,
            )
            row["stage"] = f"continuation_{idx:02d}"
            row["current_factor"] = float(factor)
            stages.append(row)
        z_star, params, row = run_solve_stage(
            engine,
            currents=target,
            warm_start=True,
            max_iter=candidate.final_max_iter,
            tol=1e-5,
        )
        row["stage"] = "final_refine"
        row["current_factor"] = 1.0
        stages.append(row)
    else:
        raise ValueError(f"Unsupported mode: {candidate.mode}")

    if z_star is None or params is None:
        raise RuntimeError(f"{candidate.name} did not produce a solution.")

    E = residual_bar(z_star, params)
    block(E)
    centerline = _extract_centerline(z_star, params, rigid_sample_count=10)
    np.save(case_dir / "centerline.npy", centerline)
    np.savez_compressed(
        case_dir / "solution.npz",
        z_bar=np.asarray(jax.device_get(z_star), dtype=np.float64),
        E=np.asarray(jax.device_get(E), dtype=np.float64),
        centerline=centerline,
    )
    metrics = pointwise_metrics(golden_centerline, centerline)
    total_wall_s = float(sum(float(item["wall_s"]) for item in stages))
    final_stage = stages[-1]
    summary = {
        "case_name": candidate.name,
        "mode": candidate.mode,
        "dense_solve_method": candidate.dense_solve_method,
        "linear_solver": candidate.linear_solver,
        "damping_mode": candidate.damping_mode,
        "step_scale": candidate.step_scale,
        "classification_hint": candidate.classification_hint,
        "ok": bool(final_stage["ok"]),
        "stop_reason": str(final_stage["stop_reason"]),
        "n_iter": int(final_stage["n_iter"]),
        "wall_s": total_wall_s,
        "steady_solve_s": total_wall_s,
        "first_call_compile_s": float(warmup.get("total_s", 0.0)),
        "wall_s_including_warmup": float(total_wall_s + float(warmup.get("total_s", 0.0))),
        "speedup_vs_golden": float(golden_wall_s / total_wall_s),
        "final_normE": float(jnp.linalg.norm(E)),
        "max_abs_E": float(jnp.max(jnp.abs(E))),
        "centerline_start_m": centerline[0].tolist(),
        "centerline_end_m": centerline[-1].tolist(),
        "golden_metrics": metrics,
        "warmup": warmup,
        "stages": stages,
        "rss_mb": rss_mb(),
    }
    summary["classification"] = classify(summary, metrics)
    (case_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def plot_final_overlay(out_dir: Path, rows: list[dict[str, Any]], golden_centerline: np.ndarray) -> None:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot(golden_centerline[:, 0], golden_centerline[:, 1], golden_centerline[:, 2], "k--", linewidth=2.5, label="golden")
    for row in rows:
        centerline = np.load(out_dir / row["case_name"] / "centerline.npy")
        label = f"{row['case_name']} {row['wall_s']:.1f}s {row['classification']}"
        ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], linewidth=1.5, label=label)
        ax.scatter(centerline[-1, 0], centerline[-1, 1], centerline[-1, 2], s=20)
    ax.scatter(golden_centerline[0, 0], golden_centerline[0, 1], golden_centerline[0, 2], color="black", s=45, label="start")
    ax.set_title("Branch-preserving FK speedup candidates")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(fontsize=7)
    ax.set_box_aspect([1, 1, 1])
    fig.tight_layout()
    fig.savefig(out_dir / "final_pose_overlay.png", dpi=180)
    plt.close(fig)


def write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows = []
    for row in rows:
        metrics = row["golden_metrics"]
        flat_rows.append(
            {
                "case_name": row["case_name"],
                "classification": row["classification"],
                "mode": row["mode"],
                "linear_solver": row.get("linear_solver", "dense"),
                "dense_solve_method": row["dense_solve_method"],
                "step_scale": row.get("step_scale", 1.0),
                "ok": row["ok"],
                "stop_reason": row["stop_reason"],
                "wall_s": row["wall_s"],
                "speedup_vs_golden": row["speedup_vs_golden"],
                "n_iter": row["n_iter"],
                "final_normE": row["final_normE"],
                "max_abs_E": row["max_abs_E"],
                "max_point_error_m": metrics["max_point_error_m"],
                "rms_point_error_m": metrics["rms_point_error_m"],
                "end_error_m": metrics["end_error_m"],
                "max_rel_to_L_ref": metrics["max_rel_to_L_ref"],
            }
        )
    fields = list(flat_rows[0].keys())
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)
    (out_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def write_record(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    best = min((row for row in rows if row["ok"]), key=lambda row: row["wall_s"], default=None)
    trusted = [
        row
        for row in rows
        if row["ok"] and row["final_normE"] <= 1e-5 and row["golden_metrics"]["max_point_error_m"] <= 0.011
    ]
    best_trusted = min(trusted, key=lambda row: row["wall_s"], default=None)
    lines = [
        "# FK 分支保持提速实验记录",
        "",
        "## 实验目的",
        "",
        "在固定 FK case 和 `tol=1e-5` 下，验证 cached/no-output、正定线性求解器和 continuation 是否能在不明显偏离当前 golden 姿态的前提下实现 3-4x 提速。",
        "",
        "## 实验方案",
        "",
        "- golden baseline：`experiment_records_fk_optimize/20260525094651_fk_new_tol_sweep/tol_1e-05/`。",
        "- 固定 `tol=1e-5`、`max_iter=100000`、`lm_damping=1e-1`、`jac_method=fwd`。",
        "- 每个候选保存最终中心线，并计算相对 golden 的逐点误差。",
        "- `max_point_error <= 0.011m` 为目标 10% 门禁；同时报告 0.5mm、1mm、2mm 更严格参考。",
        "",
        "## 实验过程记录",
        "",
        f"- {time.strftime('%Y-%m-%d %H:%M:%S')}：完成候选矩阵运行和汇总。",
        "",
        "## 实验结果",
        "",
        "| case | class | wall(s) | speedup | iter | normE | max point err(mm) | end err(mm) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        metrics = row["golden_metrics"]
        lines.append(
            f"| `{row['case_name']}` | {row['classification']} | {row['wall_s']:.2f} | "
            f"{row['speedup_vs_golden']:.2f}x | {row['n_iter']} | {row['final_normE']:.3e} | "
            f"{metrics['max_point_error_m'] * 1000:.3f} | {metrics['end_error_m'] * 1000:.3f} |"
        )
    lines.extend(["", "## 数据分析", ""])
    if best_trusted is not None:
        m = best_trusted["golden_metrics"]
        lines.append(
            f"最快可信候选为 `{best_trusted['case_name']}`，耗时 `{best_trusted['wall_s']:.2f}s`，"
            f"相对 golden 提速 `{best_trusted['speedup_vs_golden']:.2f}x`，最大逐点误差 `{m['max_point_error_m']*1000:.3f}mm`。"
        )
    if best is not None and (best_trusted is None or best["case_name"] != best_trusted["case_name"]):
        m = best["golden_metrics"]
        lines.append(
            f"最快收敛候选为 `{best['case_name']}`，但分类为 `{best['classification']}`，"
            f"最大逐点误差 `{m['max_point_error_m']*1000:.3f}mm`。"
        )
    lines.extend(
        [
            "",
            "## 实验结论",
            "",
            "见上表。若没有 production-candidate，说明当前候选尚未同时满足 3-4x 提速和分支保持门禁，需要继续做 solver 结构级优化。",
            "",
            "## 下一步计划",
            "",
            "- 对最快可信候选做多工况 smoke test。",
            "- 若未达到 3x，继续定位 dense Jacobian 和 normal equation 的结构化稀疏/块求解替代方案。",
        ]
    )
    (out_dir / "record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--rerun_existing", action="store_true")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    golden_summary = json.loads((GOLDEN_DIR / "summary.json").read_text(encoding="utf-8"))
    golden_centerline = np.load(GOLDEN_DIR / "centerline.npy")
    calib_file, calib_reason = get_calib_file()
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": str(ROOT),
        "git_commit": repo_commit(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", ""),
        "calib_file": str(calib_file),
        "calib_reason": calib_reason,
        "golden_summary": golden_summary,
        "candidate_names": [candidate.name for candidate in CANDIDATES],
    }
    (out_dir / "batch_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for candidate in CANDIDATES:
        summary_path = out_dir / candidate.name / "summary.json"
        if summary_path.exists() and not args.rerun_existing:
            row = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append(row)
            print(json.dumps({"case_name": candidate.name, "status": "loaded_existing"}, ensure_ascii=False), flush=True)
            continue
        started = time.perf_counter()
        try:
            row = run_candidate(candidate, out_dir, golden_centerline, float(golden_summary["wall_s"]))
            row["process_wall_s"] = float(time.perf_counter() - started)
        except Exception as exc:
            row = {
                "case_name": candidate.name,
                "mode": candidate.mode,
                "dense_solve_method": candidate.dense_solve_method,
                "classification": "failed result",
                "ok": False,
                "stop_reason": type(exc).__name__,
                "wall_s": float(time.perf_counter() - started),
                "speedup_vs_golden": 0.0,
                "n_iter": 0,
                "final_normE": float("nan"),
                "max_abs_E": float("nan"),
                "golden_metrics": {
                    "point_count": 0,
                    "max_point_error_m": float("inf"),
                    "rms_point_error_m": float("inf"),
                    "mean_point_error_m": float("inf"),
                    "end_error_m": float("inf"),
                    "max_rel_to_L_ref": float("inf"),
                    "end_rel_to_L_ref": float("inf"),
                },
                "error": str(exc),
            }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    write_summary(out_dir, rows)
    plot_final_overlay(out_dir, rows, golden_centerline)
    write_record(out_dir, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
