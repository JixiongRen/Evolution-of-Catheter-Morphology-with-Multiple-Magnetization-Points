"""Run closure diagnostics for A1/A2 supervised IK failures."""

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

import forward_kinematics_optimized.fk as fk_cli
from forward_kinematics_optimized.fk_engine import ForwardKinematicsEngine
from inverse_kinematics.ik_diff import (
    compute_tip_sensitivities_via_lm_adjoint,
    extract_terminal_p_dim,
)


jax.config.update("jax_enable_x64", True)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parent
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
DEFAULT_SOURCE_RUN = (
    ROOT
    / "formal_experiments"
    / "20260713063500_A1_A2_5mm_supervised_IK_no_gravity_merged_complete40"
)
DEFAULT_OUT_ROOT = ROOT / "formal_diagnostics" / "closure"


def _font() -> FontProperties:
    if FONT_PATH.exists():
        return FontProperties(fname=str(FONT_PATH))
    return FontProperties()


def _configure_fonts() -> None:
    if FONT_PATH.exists():
        plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def _jax_env() -> dict[str, Any]:
    try:
        devices = [str(dev) for dev in jax.devices()]
        platforms = sorted({getattr(dev, "platform", "unknown") for dev in jax.devices()})
        return {"devices": devices, "platforms": platforms, "has_gpu": "gpu" in platforms}
    except Exception as exc:
        return {"devices": [], "platforms": [], "has_gpu": False, "error": repr(exc)}


def _vec(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def _float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _point_dir(source_run: Path, catheter: str, index: int) -> Path:
    return source_run / catheter / "formal_20pt" / f"pt_{index:02d}"


def _history_states(source_run: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    targets: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    for row in _read_csv(source_run / "summary.csv"):
        catheter = row["catheter_type"]
        index = int(row["target_index"])
        point = _point_dir(source_run, catheter, index)
        result = _read_json(point / "result.json", {})
        history = _read_json(point / "history.json", {"outer": []})
        p_des = _vec(result.get("p_des", [math.nan, math.nan, math.nan]))
        p_final = _vec(result.get("p_tip_true", result.get("p_tip", [math.nan, math.nan, math.nan])))
        targets.append(
            {
                "catheter_type": catheter,
                "target_index": index,
                "target_m": p_des,
                "final_terminal_m": p_final,
                "final_err_mm": float(np.linalg.norm(p_final - p_des) * 1000.0),
                "success_5mm": bool(float(np.linalg.norm(p_final - p_des) * 1000.0) <= 5.0),
            }
        )
        for item in history.get("outer", []):
            p = item.get("p_tip_true_m") or item.get("p_tip_true") or item.get("p_tip_m")
            if p is None:
                continue
            states.append(
                {
                    "catheter_type": catheter,
                    "source_target_index": index,
                    "iter": int(item.get("iter", -1)),
                    "accept": bool(item.get("accept", False)),
                    "terminal_m": _vec(p),
                    "I_A": item.get("I_A"),
                    "x_m": _float(item.get("x_m")),
                    "cls": item.get("cls"),
                    "normE": _float(item.get("normE")),
                }
            )
    return targets, states


def _reachability_from_history(targets: list[dict[str, Any]], states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    states_by_catheter: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in states:
        states_by_catheter[str(state["catheter_type"])].append(state)

    for target in targets:
        catheter = str(target["catheter_type"])
        candidates = states_by_catheter.get(catheter, [])
        distances = []
        for state in candidates:
            d_mm = float(np.linalg.norm(np.asarray(state["terminal_m"]) - np.asarray(target["target_m"])) * 1000.0)
            distances.append((d_mm, state))
        distances.sort(key=lambda item: item[0])
        own = [item for item in distances if int(item[1]["source_target_index"]) == int(target["target_index"])]
        nearest_d, nearest = distances[0] if distances else (math.nan, {})
        own_d, own_state = own[0] if own else (math.nan, {})
        if target["success_5mm"]:
            conclusion = "目标自身已经达标"
        elif math.isfinite(nearest_d) and nearest_d <= 5.0:
            conclusion = "历史经验可达但本目标优化失败"
        elif math.isfinite(nearest_d) and nearest_d <= 10.0:
            conclusion = "历史经验近可达"
        elif math.isfinite(nearest_d) and nearest_d <= 30.0:
            conclusion = "历史经验边界可达"
        else:
            conclusion = "历史经验未覆盖"
        rows.append(
            {
                "catheter_type": catheter,
                "target_index": int(target["target_index"]),
                "success_5mm": bool(target["success_5mm"]),
                "final_err_mm": float(target["final_err_mm"]),
                "nearest_history_err_mm": float(nearest_d),
                "nearest_source_target_index": nearest.get("source_target_index", ""),
                "nearest_iter": nearest.get("iter", ""),
                "nearest_accept": nearest.get("accept", ""),
                "nearest_x_m": nearest.get("x_m", ""),
                "own_best_history_err_mm": float(own_d),
                "own_best_iter": own_state.get("iter", ""),
                "reachability_conclusion": conclusion,
            }
        )
    return rows


def _build_engine_from_run_config(run_config: dict[str, Any]) -> ForwardKinematicsEngine:
    argv = dict(run_config.get("argv", {}))
    m_body_list = None
    if argv.get("m_body_list"):
        m_body_list = fk_cli._parse_list_vec3(argv["m_body_list"])
    return ForwardKinematicsEngine(
        flex_lengths_tail=fk_cli._parse_csv_floats(argv["flex_lengths"]),
        rigid_lengths=fk_cli._parse_csv_floats(argv["rigid_lengths"]),
        M_list=fk_cli._parse_csv_ints(argv["M_list"]),
        L1_min=float(argv.get("L1_min", 1e-6)),
        L_protrude_max=float(argv.get("x_max") or run_config.get("x_bounds", [0.0, argv["L_protrude"]])[1]),
        flex_d_outer=fk_cli._parse_csv_floats(argv["flex_d_outer"]),
        flex_E=fk_cli._parse_csv_floats(argv["flex_E"]),
        flex_G=fk_cli._parse_csv_floats(argv["flex_G"]),
        flex_rho=fk_cli._parse_csv_floats(argv["flex_rho"]),
        rigid_d_outer=fk_cli._parse_csv_floats(argv["rigid_d_outer"]),
        rigid_rho=fk_cli._parse_csv_floats(argv["rigid_rho"]),
        p0_dim=jnp.asarray(fk_cli._parse_csv_floats(argv["p0"]), dtype=jnp.float64),
        Q0_wxyz=jnp.asarray(fk_cli._parse_csv_floats(argv["Q0"]), dtype=jnp.float64),
        axis_body=jnp.asarray(fk_cli._parse_csv_floats(argv["axis_body"]), dtype=jnp.float64),
        enable_gravity=bool(argv.get("enable_gravity", False)),
        g_world=jnp.asarray(fk_cli._parse_csv_floats(argv.get("g_world", "0,0,-9.81")), dtype=jnp.float64),
        enable_magnetics=bool(argv.get("enable_magnetics", False)),
        calib_file=argv.get("calib_file"),
        actuation_table_pkl=argv.get("actuation_table_pkl"),
        m_body_list=m_body_list,
        max_iter=int(argv.get("max_iter", 100000)),
        tol=float(argv.get("tol", 1e-5)),
        lm_damping=float(argv.get("lm_damping", 0.1)),
        jac_method=str(argv.get("jac_method", "fwd")),
    )


def _solve_terminal(
    engine: ForwardKinematicsEngine,
    *,
    I_A: np.ndarray,
    x_m: float,
    z0_bar: Any | None,
    include_stats: bool = True,
) -> tuple[Any, Any, Any, np.ndarray, bool]:
    z, params, _meshes, ok, stats = engine.solve_with_stats(
        coil_currents=jnp.asarray(I_A, dtype=jnp.float64),
        L_protrude=float(x_m),
        warm_start=True,
        override_z0_bar=z0_bar,
        return_stats=True,
        stats_include_jacobian=False,
    )
    p = np.asarray(jax.device_get(extract_terminal_p_dim(z, M_list=params.M_list, scales=params.scales)), dtype=float)
    return z, params, stats if include_stats else None, p, bool(ok)


def _replay_point(source_run: Path, catheter: str, index: int) -> dict[str, Any]:
    point = _point_dir(source_run, catheter, index)
    result = _read_json(point / "result.json", {})
    history = _read_json(point / "history.json", {"outer": []})
    run_config = _read_json(point / "run_config.json", {})
    engine = _build_engine_from_run_config(run_config)

    accepted_z = None
    accepted_params = None
    accepted_stats = None
    accepted_p = None
    replay_rows = []
    for item in history.get("outer", []):
        I = np.asarray(item.get("I_A", []), dtype=float)
        x = _float(item.get("x_m"))
        if I.size != 8 or not math.isfinite(x):
            continue
        z, params, stats, p, ok = _solve_terminal(engine, I_A=I, x_m=x, z0_bar=accepted_z)
        rec_p = _vec(item.get("p_tip_true_m") or item.get("p_tip_true") or item.get("p_tip_m"))
        mismatch_mm = float(np.linalg.norm(p - rec_p) * 1000.0)
        replay_rows.append(
            {
                "iter": int(item.get("iter", -1)),
                "accept": bool(item.get("accept", False)),
                "recorded_err_true_mm": _float(item.get("err_true_mm")),
                "replay_to_recorded_terminal_mm": mismatch_mm,
                "ok": bool(ok),
            }
        )
        if bool(item.get("accept", False)):
            accepted_z = z
            accepted_params = params
            accepted_stats = stats
            accepted_p = p

    final_recorded = _vec(result.get("p_tip_true", result.get("p_tip", [math.nan, math.nan, math.nan])))
    final_mismatch_mm = float(np.linalg.norm(np.asarray(accepted_p) - final_recorded) * 1000.0) if accepted_p is not None else math.nan
    return {
        "engine": engine,
        "z": accepted_z,
        "params": accepted_params,
        "stats": accepted_stats,
        "terminal_m": accepted_p,
        "final_recorded_m": final_recorded,
        "final_mismatch_mm": final_mismatch_mm,
        "replay_rows": replay_rows,
        "result": result,
    }


def _sensitivity_audit_for_point(
    source_run: Path,
    catheter: str,
    index: int,
    *,
    fd_eps_A: float,
    fd_eps_x_m: float,
    replay_tol_mm: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    replay = _replay_point(source_run, catheter, index)
    branch_rows = [
        {
            "catheter_type": catheter,
            "target_index": index,
            "trial_type": "history_replay",
            "status": "ok" if replay["final_mismatch_mm"] <= replay_tol_mm else "replay_mismatch",
            "terminal_mismatch_mm": replay["final_mismatch_mm"],
            "max_iter_terminal_mismatch_mm": max(
                [row["replay_to_recorded_terminal_mm"] for row in replay["replay_rows"]] or [math.nan]
            ),
            "conclusion_zh": (
                "可以重放原 IK 平衡分支。"
                if replay["final_mismatch_mm"] <= replay_tol_mm
                else "无法用历史控制量稳定重放原 IK 分支，旧结果不适合做严格分支保持差分。"
            ),
        }
    ]
    if replay["z"] is None or replay["final_mismatch_mm"] > replay_tol_mm:
        return (
            [
                {
                    "catheter_type": catheter,
                    "target_index": index,
                    "audit_scope": "branch_preserving_terminal_sensitivity",
                    "status": "replay_mismatch",
                    "reason_zh": "历史重放未回到原末端分支，灵敏度审计无物理意义。",
                    "branch_mismatch_mm": replay["final_mismatch_mm"],
                }
            ],
            branch_rows,
        )

    engine: ForwardKinematicsEngine = replay["engine"]
    z0 = replay["z"]
    params = replay["params"]
    stats = replay["stats"]
    result = replay["result"]
    I0 = np.asarray(result["I"], dtype=float)
    x0 = float(result["x"])

    sens = compute_tip_sensitivities_via_lm_adjoint(
        z_star_bar=z0,
        params=params,
        lm_stats=stats,
        coil_currents=jnp.asarray(I0, dtype=jnp.float64),
        ridge=None,
        materialize_jacobian=True,
        compute_conditioning=True,
    )
    J_imp = np.asarray(jax.device_get(sens.J_p_I), dtype=float)
    Jx_imp = np.asarray(jax.device_get(sens.J_p_x), dtype=float).reshape(3)

    fd_cols = []
    rows = []
    for j in range(8):
        ej = np.zeros_like(I0)
        ej[j] = 1.0
        zp, pp, _sp, p_plus, okp = _solve_terminal(engine, I_A=I0 + fd_eps_A * ej, x_m=x0, z0_bar=z0)
        zm, pm, _sm, p_minus, okm = _solve_terminal(engine, I_A=I0 - fd_eps_A * ej, x_m=x0, z0_bar=z0)
        col = (p_plus - p_minus) / (2.0 * fd_eps_A)
        fd_cols.append(col)
        denom = max(float(np.linalg.norm(col)), 1e-12)
        rel = float(np.linalg.norm(J_imp[:, j] - col) / denom)
        rows.append(
            {
                "catheter_type": catheter,
                "target_index": index,
                "audit_scope": "branch_preserving_terminal_sensitivity",
                "control": f"I{j}",
                "status": "ok" if rel <= 0.10 else "failed",
                "fd_eps": fd_eps_A,
                "relative_error": rel,
                "implicit_norm": float(np.linalg.norm(J_imp[:, j])),
                "fd_norm": float(np.linalg.norm(col)),
                "ok_plus": bool(okp),
                "ok_minus": bool(okm),
                "sensitivity_status": str(sens.status),
            }
        )

    xp = x0 + fd_eps_x_m
    xm = max(float(params.L1_dim), x0 - fd_eps_x_m)
    _zp, _pp, _sp, p_plus, okp = _solve_terminal(engine, I_A=I0, x_m=xp, z0_bar=z0)
    _zm, _pm, _sm, p_minus, okm = _solve_terminal(engine, I_A=I0, x_m=xm, z0_bar=z0)
    col_x = (p_plus - p_minus) / max(xp - xm, 1e-12)
    rel_x = float(np.linalg.norm(Jx_imp - col_x) / max(float(np.linalg.norm(col_x)), 1e-12))
    rows.append(
        {
            "catheter_type": catheter,
            "target_index": index,
            "audit_scope": "branch_preserving_terminal_sensitivity",
            "control": "x",
            "status": "ok" if rel_x <= 0.10 else "failed",
            "fd_eps": fd_eps_x_m,
            "relative_error": rel_x,
            "implicit_norm": float(np.linalg.norm(Jx_imp)),
            "fd_norm": float(np.linalg.norm(col_x)),
            "ok_plus": bool(okp),
            "ok_minus": bool(okm),
            "sensitivity_status": str(sens.status),
        }
    )
    global_fd = np.column_stack(fd_cols + [col_x])
    global_imp = np.column_stack([J_imp, Jx_imp.reshape(3, 1)])
    rows.append(
        {
            "catheter_type": catheter,
            "target_index": index,
            "audit_scope": "branch_preserving_terminal_sensitivity",
            "control": "global",
            "status": "ok" if float(np.linalg.norm(global_imp - global_fd) / max(np.linalg.norm(global_fd), 1e-12)) <= 0.10 else "failed",
            "fd_eps": "",
            "relative_error": float(np.linalg.norm(global_imp - global_fd) / max(np.linalg.norm(global_fd), 1e-12)),
            "implicit_norm": float(np.linalg.norm(global_imp)),
            "fd_norm": float(np.linalg.norm(global_fd)),
            "ok_plus": "",
            "ok_minus": "",
            "sensitivity_status": str(sens.status),
        }
    )
    return rows, branch_rows


def _plot_reachability(run_dir: Path, targets: list[dict[str, Any]], states: list[dict[str, Any]]) -> None:
    font = _font()
    for catheter in ("A1", "A2"):
        cat_targets = [t for t in targets if t["catheter_type"] == catheter]
        cat_states = [s for s in states if s["catheter_type"] == catheter]
        if not cat_targets:
            continue
        fig = plt.figure(figsize=(8.0, 6.8), dpi=180)
        ax = fig.add_subplot(111, projection="3d")
        if cat_states:
            p = np.asarray([s["terminal_m"] for s in cat_states], dtype=float) * 1000.0
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=8, c="#5c7cfa", alpha=0.25, label="历史探索末端")
        t = np.asarray([x["target_m"] for x in cat_targets], dtype=float) * 1000.0
        success = np.asarray([x["success_5mm"] for x in cat_targets], dtype=bool)
        ax.plot(t[:, 0], t[:, 1], t[:, 2], color="#333333", lw=1.2, label="目标圆环")
        ax.scatter(t[success, 0], t[success, 1], t[success, 2], s=40, c="#168a2f", label="达标目标")
        ax.scatter(t[~success, 0], t[~success, 1], t[~success, 2], s=42, c="#d92525", marker="x", label="未达标目标")
        ax.set_xlabel("x (mm)", fontproperties=font)
        ax.set_ylabel("y (mm)", fontproperties=font)
        ax.set_zlabel("z (mm)", fontproperties=font)
        ax.set_title(f"{catheter} 历史轨迹经验可达集", fontproperties=font)
        ax.legend(prop=font)
        fig.tight_layout()
        fig.savefig(run_dir / f"{catheter}_history_reachability.png", bbox_inches="tight")
        fig.savefig(run_dir / f"{catheter}_history_reachability.svg", bbox_inches="tight")
        plt.close(fig)


def run_closure(args: argparse.Namespace) -> Path:
    _configure_fonts()
    source_run = Path(args.source_run).resolve()
    run_dir = Path(args.out_root).resolve() / f"{datetime.now().strftime('%Y%m%d%H%M%S')}_A1_A2_diagnostic_closure"
    run_dir.mkdir(parents=True, exist_ok=True)

    targets, states = _history_states(source_run)
    reach_rows = _reachability_from_history(targets, states)
    _write_csv(run_dir / "reachability_history_summary.csv", reach_rows)
    np.savez_compressed(
        run_dir / "reachability_samples.npz",
        terminal_points_m=np.asarray([s["terminal_m"] for s in states], dtype=float),
        target_points_m=np.asarray([t["target_m"] for t in targets], dtype=float),
        catheter_type=np.asarray([s["catheter_type"] for s in states]),
        source_target_index=np.asarray([s["source_target_index"] for s in states], dtype=int),
        iter=np.asarray([s["iter"] for s in states], dtype=int),
    )
    _plot_reachability(run_dir, targets, states)

    sens_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    for spec in args.sensitivity_points:
        catheter, idx_s = spec.split(":")
        rows, branches = _sensitivity_audit_for_point(
            source_run,
            catheter,
            int(idx_s),
            fd_eps_A=float(args.fd_eps_A),
            fd_eps_x_m=float(args.fd_eps_x_m),
            replay_tol_mm=float(args.replay_tol_mm),
        )
        sens_rows.extend(rows)
        branch_rows.extend(branches)
        _write_csv(run_dir / "sensitivity_columns.csv", sens_rows)
        _write_csv(run_dir / "branch_trials.csv", branch_rows)

    if not sens_rows:
        _write_csv(run_dir / "sensitivity_columns.csv", [{"status": "not_run", "reason_zh": "未指定灵敏度审计点。"}])
    if not branch_rows:
        _write_csv(run_dir / "branch_trials.csv", [{"status": "not_run", "reason_zh": "未指定分支重放点。"}])

    summary = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_run": str(source_run),
            "git_commit": _git_commit(),
            "python": sys.executable,
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "jax": _jax_env(),
        },
        "reachability_counts": dict(Counter(row["reachability_conclusion"] for row in reach_rows)),
        "sensitivity_counts": dict(Counter(row.get("status", "") for row in sens_rows)),
        "branch_counts": dict(Counter(row.get("status", "") for row in branch_rows)),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "commands.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"XLA_PYTHON_CLIENT_PREALLOCATE=false {sys.executable} {Path(__file__).resolve()} "
        f"--source-run {source_run} --out-root {Path(args.out_root).resolve()} "
        + " ".join(f"--sensitivity-point {p}" for p in args.sensitivity_points)
        + f" --fd-eps-A {args.fd_eps_A} --fd-eps-x-m {args.fd_eps_x_m}\n",
        encoding="utf-8",
    )
    _write_record(run_dir, source_run, summary, reach_rows, sens_rows, branch_rows)
    print(f"closure_dir: {run_dir}", flush=True)
    return run_dir


def _write_record(
    run_dir: Path,
    source_run: Path,
    summary: dict[str, Any],
    reach_rows: list[dict[str, Any]],
    sens_rows: list[dict[str, Any]],
    branch_rows: list[dict[str, Any]],
) -> None:
    by_catheter: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reach_rows:
        by_catheter[str(row["catheter_type"])].append(row)

    lines = [
        "# A1/A2 有监督逆运动学诊断闭环补充报告",
        "",
        "## 一、目的",
        "",
        "本报告补充 D1、D3 和 D4 的关键证据，重点回答失败点是否已有经验可达证据，以及旧结果能否支持分支保持灵敏度审计。",
        "",
        "## 二、数据边界",
        "",
        f"- 源实验目录：`{source_run}`",
        f"- 输出目录：`{run_dir}`",
        f"- git 版本：`{summary['metadata']['git_commit']}`",
        f"- JAX 设备：`{summary['metadata']['jax']}`",
        "",
        "## 三、经验可达性结论",
        "",
    ]
    for catheter in ("A1", "A2"):
        rows = by_catheter.get(catheter, [])
        counts = Counter(row["reachability_conclusion"] for row in rows)
        lines.append(f"### {catheter}")
        lines.append("")
        for key, value in counts.most_common():
            lines.append(f"- {key}：{value} 点。")
        hard = [
            row
            for row in rows
            if (not row["success_5mm"]) and row["reachability_conclusion"] == "历史经验未覆盖"
        ]
        if hard:
            labels = ", ".join(f"pt{int(row['target_index']):02d}" for row in hard)
            lines.append(f"- 历史探索仍未覆盖的失败点：{labels}。这些点不能直接判定不可达，只能判定为现有探索证据不足。")
        lines.append("")

    lines.extend(
        [
            "## 四、分支与灵敏度结论",
            "",
        ]
    )
    if branch_rows:
        for row in branch_rows:
            lines.append(
                f"- {row['catheter_type']}-pt{int(row['target_index']):02d}："
                f"历史重放状态为 `{row['status']}`，末端差异 {float(row['terminal_mismatch_mm']):.3f} mm。"
            )
    else:
        lines.append("- 本轮未执行分支重放。")
    if sens_rows:
        by_point: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in sens_rows:
            if row.get("control") in {"I0", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "x", "global"}:
                by_point[(str(row["catheter_type"]), int(row["target_index"]))].append(row)
        global_rows = [row for row in sens_rows if row.get("control") == "global"]
        for row in global_rows:
            point_rows = by_point[(str(row["catheter_type"]), int(row["target_index"]))]
            current_rows = [x for x in point_rows if str(x.get("control", "")).startswith("I")]
            current_failed = sum(1 for x in current_rows if str(x.get("status")) != "ok")
            x_rows = [x for x in point_rows if x.get("control") == "x"]
            x_status = x_rows[0].get("status") if x_rows else "未审计"
            lines.append(
                f"- {row['catheter_type']}-pt{int(row['target_index']):02d}："
                f"全局灵敏度相对误差 {float(row['relative_error']):.3f}，状态 `{row['status']}`；"
                f"8 个电流通道中 {current_failed} 个未通过 10% 阈值，推进量通道状态为 `{x_status}`。"
            )
    else:
        lines.append("- 本轮未执行灵敏度差分。")

    lines.extend(
        [
            "",
            "## 五、行动结论",
            "",
            "1. 现有历史轨迹可作为经验可达集，但它不是均匀控制空间采样；对“历史经验未覆盖”的点，结论只能是探索不足，不能写成物理不可达。",
            "2. 若历史重放无法回到原分支，旧结果不适合做严格灵敏度审计；下一轮必须在 IK 运行时直接保存平衡状态。",
            "3. 若分支保持差分仍显示灵敏度误差超过 10%，优先修复隐式灵敏度；若灵敏度可用，则优先修复外层信赖域、回溯和多初值分支管理。",
            "",
            "## 六、文件索引",
            "",
            "- `reachability_history_summary.csv`：每个目标到历史经验可达集的最近距离。",
            "- `reachability_samples.npz`：历史外层迭代产生的末端点云。",
            "- `sensitivity_columns.csv`：分支保持灵敏度审计结果。",
            "- `branch_trials.csv`：历史轨迹重放分支检查。",
            "- `A1_history_reachability.png`、`A2_history_reachability.png`：三维经验可达集图。",
        ]
    )
    (run_dir / "record.md").write_text("\n".join(lines), encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--sensitivity-point",
        action="append",
        dest="sensitivity_points",
        default=[],
        help="Point spec such as A1:12. Can be repeated.",
    )
    parser.add_argument("--fd-eps-A", type=float, default=1e-3)
    parser.add_argument("--fd-eps-x-m", type=float, default=1e-4)
    parser.add_argument("--replay-tol-mm", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    t0 = time.perf_counter()
    run_dir = run_closure(args)
    print(f"record: {run_dir / 'record.md'}", flush=True)
    print(f"wall_s: {time.perf_counter() - t0:.3f}", flush=True)


if __name__ == "__main__":
    main()
